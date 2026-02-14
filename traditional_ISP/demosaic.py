import torch
import torch.nn as nn
import torch.nn.functional as F


class Demosaic(nn.Module):
    """
    Демозаикинг Bayer изображения в RGB с использованием алгоритма Malvar-He-Cutler
    """

    def __init__(self):
        super().__init__()

        # Malvar-He-Cutler фильтры

        G_at_RB = torch.tensor([[0, 0, -1, 0, 0],
                                [0, 0,  2, 0, 0],
                                [-1, 2, 4, 2, -1],
                                [0, 0, 2, 0, 0],
                                [0, 0, -1, 0, 0]], dtype=torch.float32) / 8

        R_at_G_Rrow = torch.tensor([[0, 0, 0.5, 0, 0],
                                    [0, -1, 0, -1, 0],
                                    [-1, 4, 5, 4, -1],
                                    [0, -1, 0, -1, 0],
                                    [0, 0, 0.5, 0, 0]], dtype=torch.float32) / 8

        R_at_G_Brow = torch.tensor([[0, 0, -1, 0, 0],
                                    [0, -1, 4, -1, 0],
                                    [0.5, 0, 5, 0, 0.5],
                                    [0, -1, 4, -1, 0],
                                    [0, 0, -1, 0, 0]], dtype=torch.float32) / 8

        R_at_B = torch.tensor([[0, 0, -1.5, 0, 0],
                               [0, 2, 0, 2, 0],
                               [-1.5, 0, 6, 0, -1.5],
                               [0, 2, 0, 2, 0],
                               [0, 0, -1.5, 0, 0]], dtype=torch.float32) / 8

        # Объединяем все kernels в один batch
        batched_kernels = torch.stack([G_at_RB.unsqueeze(0),     
                                       R_at_G_Rrow.unsqueeze(0),  
                                       R_at_G_Brow.unsqueeze(0),  
                                       R_at_B.unsqueeze(0)], dim=0)

        self.register_buffer('batched_kernels', batched_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет демозаикинг к Bayer изображению

        Args:
            x: Bayer изображение RGGB, shape: [H, W], dtype: int32, range: [0, 0xFFFFFF]

        Returns:
            torch.Tensor: RGB изображение, shape: [H, W, 3], dtype: int32, range: [0, 0xFFFFFF]
        """
        H, W = x.shape

        x_float = x.float()

        # Создаем каналы R, G, B
        R = torch.zeros_like(x_float)
        G = torch.zeros_like(x_float)
        B = torch.zeros_like(x_float)

        # Извлекаем RGGB
        R[::2, ::2] = x_float[::2, ::2]
        G[::2, 1::2] = x_float[::2, 1::2]
        G[1::2, ::2] = x_float[1::2, ::2]
        B[1::2, 1::2] = x_float[1::2, 1::2]

        # Добавляем batch и channel измерения для conv2d
        x_4d = x_float.unsqueeze(0).unsqueeze(0) 

        # Padding один раз для всех операций
        x_padded = F.pad(x_4d, (2, 2, 2, 2), mode='reflect')

        all_results = F.conv2d(x_padded, self.batched_kernels)

        # Извлекаем результаты каждого kernel
        G_interp = all_results[0, 0:1, :, :] 
        R_g_r = all_results[0, 1:2, :, :]    
        R_g_b = all_results[0, 2:3, :, :]    
        R_b = all_results[0, 3:4, :, :]      

        # Интерполяция зеленого канала
        G[::2, ::2] = G_interp[0, ::2, ::2]
        G[1::2, 1::2] = G_interp[0, 1::2, 1::2]

        # Интерполяция красного канала
        R[::2, 1::2] = R_g_r[0, ::2, 1::2]
        R[1::2, ::2] = R_g_b[0, 1::2, ::2]
        R[1::2, 1::2] = R_b[0, 1::2, 1::2]

        # Интерполяция синего канала
        B[1::2, ::2] = R_g_r[0, 1::2, ::2]
        B[::2, 1::2] = R_g_b[0, ::2, 1::2]
        B[::2, ::2] = R_b[0, ::2, ::2]

        # Формируем RGB изображение
        rgb = torch.stack([R, G, B], dim=-1)

        # Возвращаем в int32, ограничиваем диапазон
        return torch.clamp(rgb, 0, 0xFFFFFF).to(torch.int32)
