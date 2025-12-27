import torch
import torch.nn.functional as F


def demosaic(frame: torch.Tensor) -> torch.Tensor:
    """
    Применяет демозаик к изображению формата RGGB Bayer с использованием алгоритма Malvar–He–Cutler.

    Args:
        frame: 2D тензор после AWB (H x W)

    Returns:
        torch.Tensor: RGB изображение (H x W x 3)
    """
    device = frame.device
    frame = frame.float()

    R = torch.zeros_like(frame)
    G = torch.zeros_like(frame)
    B = torch.zeros_like(frame)

    # Извлекаем RGGB
    R[::2, ::2] = frame[::2, ::2]
    G[::2, 1::2] = frame[::2, 1::2]
    G[1::2, ::2] = frame[1::2, ::2]
    B[1::2, 1::2] = frame[1::2, 1::2]

    # Malvar–He–Cutler фильтры
    def to_tensor(kernel):
        return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    G_at_RB = to_tensor([[0, 0, -1, 0, 0],
                         [0, 0,  2, 0, 0],
                         [-1, 2, 4, 2, -1],
                         [0, 0, 2, 0, 0],
                         [0, 0, -1, 0, 0]]) / 8

    R_at_G_Rrow = to_tensor([[0, 0, 0.5, 0, 0],
                             [0, -1, 0, -1, 0],
                             [-1, 4, 5, 4, -1],
                             [0, -1, 0, -1, 0],
                             [0, 0, 0.5, 0, 0]]) / 8

    R_at_G_Brow = to_tensor([[0, 0, -1, 0, 0],
                             [0, -1, 4, -1, 0],
                             [0.5, 0, 5, 0, 0.5],
                             [0, -1, 4, -1, 0],
                             [0, 0, -1, 0, 0]]) / 8

    R_at_B = to_tensor([[0, 0, -1.5, 0, 0],
                        [0, 2, 0, 2, 0],
                        [-1.5, 0, 6, 0, -1.5],
                        [0, 2, 0, 2, 0],
                        [0, 0, -1.5, 0, 0]]) / 8

    # Добавляем batch и channel измерения
    frame_4d = frame.unsqueeze(0).unsqueeze(0)
    
    # Интерполяция зеленого
    G_interp = F.conv2d(F.pad(frame_4d, (2, 2, 2, 2), mode='reflect'), G_at_RB.to(device))
    G[::2, ::2] = G_interp[0, 0, ::2, ::2]
    G[1::2, 1::2] = G_interp[0, 0, 1::2, 1::2]

    # Интерполяция красного
    R_g_r = F.conv2d(F.pad(frame_4d, (2, 2, 2, 2), mode='reflect'), R_at_G_Rrow.to(device))
    R_g_b = F.conv2d(F.pad(frame_4d, (2, 2, 2, 2), mode='reflect'), R_at_G_Brow.to(device))
    R_b   = F.conv2d(F.pad(frame_4d, (2, 2, 2, 2), mode='reflect'), R_at_B.to(device))

    R[::2, 1::2] = R_g_r[0, 0, ::2, 1::2]
    R[1::2, ::2] = R_g_b[0, 0, 1::2, ::2]
    R[1::2, 1::2] = R_b[0, 0, 1::2, 1::2]

    # Интерполяция синего
    B[1::2, ::2] = R_g_r[0, 0, 1::2, ::2]  
    B[::2, 1::2] = R_g_b[0, 0, ::2, 1::2]  
    B[::2, ::2] = R_b[0, 0, ::2, ::2]   
    
    # Формируем RGB изображение
    rgb = torch.stack([R, G, B], dim=-1)  

    # Ограничиваем диапазон и конвертируем
    return torch.clamp(rgb, 0, 0xFFFFFF).to(torch.uint32)