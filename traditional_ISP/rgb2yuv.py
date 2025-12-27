import torch


def rgb_to_yuv(frame: torch.Tensor) -> torch.Tensor:
    """
    Преобразует RGB изображение в формат YUV с использованием цветового пространства BT.709 (full range).
    
    Args:
        frame: RGB изображение (H x W x 3), float в диапазоне [0,1]

    Returns:
        torch.Tensor: YUV в формате NV12, uint8, 1D тензор длины H*W*3/2
    """
    H, W, _ = frame.shape
    
    # Разделение цветовых каналов
    R = frame[..., 0]
    G = frame[..., 1]
    B = frame[..., 2]

    # Преобразование RGB → YUV (BT.709 full range)
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    U = -0.1146 * R - 0.3854 * G + 0.5000 * B
    V = 0.5000 * R - 0.4542 * G - 0.0458 * B
    
    # Субдискретизация цветности 4:2:0
    U_420 = 0.25 * (U[0::2, 0::2] + U[0::2, 1::2] + U[1::2, 0::2] + U[1::2, 1::2])

    V_420 = 0.25 * (V[0::2, 0::2] + V[0::2, 1::2] + V[1::2, 0::2] + V[1::2, 1::2])
    
    # Масштабирование в полный диапазон 8 бит
    Y_uint8 = torch.clamp(Y * 255.0, 0, 255).to(torch.uint8)
    U_420_uint8 = torch.clamp(U_420 * 255.0 + 128.0, 0, 255).to(torch.uint8)
    V_420_uint8 = torch.clamp(V_420 * 255.0 + 128.0, 0, 255).to(torch.uint8)

    # Создаем 1D тензор
    yuv = torch.empty(H * W + 2 * (H//2 * W//2), dtype=torch.uint8, device=frame.device)

    # Y plane
    yuv[0:H*W] = Y_uint8.reshape(-1)

    # UV plane (interleaved)
    uv = torch.stack([U_420_uint8, V_420_uint8], dim=-1).reshape(-1)
    yuv[H*W:] = uv

    return yuv
        
        
def save_yuv(yuv_frame: torch.Tensor, filename: str, append: bool = True):
    """
    Сохраняет один YUV кадр в бинарный файл. 
    Если append=True, кадр добавляется к существующему файлу (для видео).
    
    Args:
        yuv_data: YUV тензор (uint8, 1D, размер H*W*3/2)
        filename: имя выходного файла
        append: если True, кадр добавляется к файлу; если False — перезаписывает файл
    """
    mode = 'ab' if append else 'wb'  # append или write
    with open(filename, mode) as f:
        yuv_np = yuv_frame.cpu().numpy()
        f.write(yuv_np.tobytes())