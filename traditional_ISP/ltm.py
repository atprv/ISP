import torch
import torch.nn.functional as F


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Быстрый box filter через интегральное изображение (summed area table).
    
    Args:
        x: входной тензор
        radius: радиус окна
        
    Returns:
        torch.Tensor: отфильтрованное изображение
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    
    B, C, H, W = x.shape
    
    # Размеры окна
    kernel_size = 2 * radius + 1
    
    # Паддинг для обработки краев
    x_padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
    
    # Cumsum по обеим осям для создания интегрального изображения
    cumsum_h = torch.cumsum(x_padded, dim=2)
    cumsum_hw = torch.cumsum(cumsum_h, dim=3)
    integral = F.pad(cumsum_hw, (1, 0, 1, 0))
    
    # Координаты углов окна в интегральном изображении
    br = integral[:, :, kernel_size:H + kernel_size, kernel_size:W + kernel_size]
    bl = integral[:, :, kernel_size:H + kernel_size, :W]
    tr = integral[:, :, :H, kernel_size:W + kernel_size]
    tl = integral[:, :, :H, :W]
    
    # Вычисляем сумму в окне
    box_sum = br - bl - tr + tl
    
    # Нормализуем на размер окна
    return box_sum / (kernel_size * kernel_size)


def guided_filter_gray(I: torch.Tensor, radius: int = 32, eps: float = 1e-3) -> torch.Tensor:
    """
    Применяет guided filter к одноканальному изображению.

    Args:
        I: входное одноканальное изображение (H x W)
        radius: радиус окна усреднения
        eps: регуляризующий параметр для подавления шума

    Returns:
        torch.Tensor: отфильтрованное изображение (H x W)
    """
    # Добавляем batch и channel измерения
    I = I.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    
    # Средние значения
    mean_I = box_filter(I, radius)
    mean_II = box_filter(I * I, radius)
    
    # Дисперсия
    var_I = mean_II - mean_I * mean_I

    # Коэффициенты линейной модели
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I

    # Усреднение коэффициентов
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    # Выходное изображение
    out = mean_a * I + mean_b
    return out.squeeze(0).squeeze(0)


def local_tone_mapping(image: torch.Tensor, a: float = 0.7, b: float = 0.0, radius: int = 32) -> torch.Tensor:
    """
    Применяет Local Tone Mapping к RGB изображению

    Args:
        image: RGB изображение (H x W x 3), float в диапазоне [0, 1]
        a: коэффициент сжатия динамического диапазона
        b: сдвиг яркости в лог-домене
        radius: радиус guided filter

    Returns:
        torch.Tensor: RGB изображение после tone mapping (H x W x 3)
    """
    
    # Разделение цветовых каналов
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    # Вычисление яркости (Rec.709)
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # Защита от нуля перед логарифмом
    eps_log = 1e-4
    Y_safe = torch.clamp(Y, min=eps_log)

    # Переход в логарифмическую область
    Y_log = torch.log2(Y_safe)

    # Разложение на base (крупномасштабные изменения) и detail (текстура)
    Y_base = guided_filter_gray(Y_log, radius=radius, eps=1e-3)
    Y_detail = Y_log - Y_base

    # Tone mapping базовой компоненты
    Y_base_tm = a * Y_base + b

    # Восстановление с деталями
    Y_tm = Y_base_tm + Y_detail
    Y_out = 2.0 ** Y_tm
    
    # Цветовое масштабирование
    eps_scale = 1e-8
    scale = Y_out / (Y + eps_scale)

    # Применяем масштабирование к каждому каналу
    out = torch.empty_like(image)
    out[..., 0] = R * scale
    out[..., 1] = G * scale
    out[..., 2] = B * scale

    return torch.clamp(out, 0.0, 1.0)
