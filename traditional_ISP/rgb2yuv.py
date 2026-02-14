import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBtoYUV(nn.Module):
    """
    Преобразование RGB в YUV формат NV12 (4:2:0 subsampling)
    """
    
    def __init__(self):
        super().__init__()
        
        # BT.709 коэффициенты преобразования (full range)
        rgb_to_y = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
        rgb_to_u = torch.tensor([-0.1146, -0.3854, 0.5000], dtype=torch.float32)
        rgb_to_v = torch.tensor([0.5000, -0.4542, -0.0458], dtype=torch.float32)
        
        # Объединяем в матрицу для одного матричного умножения [3, 3]
        rgb2yuv_matrix = torch.stack([rgb_to_y, rgb_to_u, rgb_to_v], dim=0)
        self.register_buffer('rgb2yuv_matrix', rgb2yuv_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Преобразует RGB в YUV NV12 формат
        
        Args:
            x: RGB изображение, shape: [H, W, 3], dtype: float32, range: [0, 1]
        
        Returns:
            torch.Tensor: YUV в формате NV12, 1D тензор uint8
        """
        H, W, _ = x.shape
        
        # Векторизованное преобразование RGB -> YUV
        yuv = x @ self.rgb2yuv_matrix.T
        
        # Выделяем Y, U, V
        Y = yuv[..., 0]
        U = yuv[..., 1]
        V = yuv[..., 2]
        
        # Subsampling 4:2:0 для U и V через avg_pool2d
        # Преобразуем в [1, 1, H, W] для pooling
        U_4d = U.unsqueeze(0).unsqueeze(0)
        V_4d = V.unsqueeze(0).unsqueeze(0)
        
        # Average pooling 2x2
        U_420 = F.avg_pool2d(U_4d, kernel_size=2, stride=2).squeeze()
        V_420 = F.avg_pool2d(V_4d, kernel_size=2, stride=2).squeeze()
        
        # Масштабирование в 8-bit диапазон
        Y_uint8 = (Y * 255.0).clamp_(0, 255).byte()
        U_420_uint8 = (U_420 * 255.0 + 128.0).clamp_(0, 255).byte()
        V_420_uint8 = (V_420 * 255.0 + 128.0).clamp_(0, 255).byte()
        
        # Создаем выходной буфер NV12: [Y plane] + [UV interleaved]
        yuv_size = H * W + 2 * (H // 2) * (W // 2)
        yuv = torch.empty(yuv_size, dtype=torch.uint8, device=x.device)
        
        # Y plane
        yuv[:H * W] = Y_uint8.flatten()
        
        # UV plane (interleaved)
        uv_interleaved = torch.stack([U_420_uint8, V_420_uint8], dim=-1).flatten()
        yuv[H * W:] = uv_interleaved
        
        return yuv
