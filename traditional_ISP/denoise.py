import torch
import torch.nn.functional as F


def median_filter(channel: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Применяет простую медианную фильтрацию к 2D тензору через unfold.

    Args:
        channel: 2D тензор одного канала изображения
        kernel_size: размер ядра медианного фильтра (нечетное число)

    Returns:
        torch.Tensor: отфильтрованный канал с той же формой
    """
    # Преобразуем к [1, 1, H, W] и float для фильтрации
    c = channel.unsqueeze(0).unsqueeze(0).float()
    
    # Размер паддинга
    pad = kernel_size // 2
    
    # Разворачиваем окно свертки
    c_unfold = F.unfold(F.pad(c, (pad, pad, pad, pad), mode='reflect'), kernel_size)
    
    # Вычисляем медиану по каждому окну
    median, _ = c_unfold.median(dim=1)
    
    # Возвращаем в исходную форму канала
    return median.reshape(channel.shape)


def bayer_denoise(frame: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Применяет шумоподавление к изображению формата RGGB Bayer.
    
    Args:
        frame: 2D тензор после decompand
        kernel_size: размер ядра (нечетное число)
    
    Returns:
        torch.Tensor: денойзированное изображение в том же формате
    """
    # Приводим к int32 для медианной фильтрации
    frame = frame.to(torch.int32)
    
    # Создаем выходной тензор
    denoised_frame = torch.empty_like(frame)
    
    # Извлекаем каналы из изображения Bayer
    r = frame[::2, ::2]     # Красные пиксели
    gr = frame[::2, 1::2]   # Зеленые пиксели (красная строка)
    gb = frame[1::2, ::2]   # Зеленые пиксели (синяя строка)
    b = frame[1::2, 1::2]   # Синие пиксели
    
    # Медианная фильтрация
    r_filtered = median_filter(r, kernel_size=kernel_size)
    gr_filtered = median_filter(gr, kernel_size=kernel_size)
    gb_filtered = median_filter(gb, kernel_size=kernel_size)
    b_filtered = median_filter(b, kernel_size=kernel_size)
    
    # Собираем обратно в Bayer паттерн
    denoised_frame[::2, ::2] = r_filtered
    denoised_frame[::2, 1::2] = gr_filtered
    denoised_frame[1::2, ::2] = gb_filtered
    denoised_frame[1::2, 1::2] = b_filtered
    
    return denoised_frame.to(torch.uint32)