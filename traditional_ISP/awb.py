import torch


def awb_gray_world(frame: torch.Tensor, max_gain: float = 4.0) -> torch.Tensor:
    """
    Применяет баланс белого (Gray World) к изображению формата RGGB Bayer.

    Args:
        frame: 2D тензор после denoise
        max_gain: максимально допустимый коэффициент усиления

    Returns:
        torch.Tensor: изображение в том же формате с применённым AWB 
    """
    awb_frame = frame.float()

    # Извлекаем RGGB каналы
    r  = awb_frame[::2, ::2]
    gr = awb_frame[::2, 1::2]
    gb = awb_frame[1::2, ::2]
    b  = awb_frame[1::2, 1::2]

    # Средние значения каналов
    r_mean = r.mean()
    g_mean = 0.5 * (gr.mean() + gb.mean())
    b_mean = b.mean()

    # Коэффициенты усиления (Gray World)
    eps = 1.0
    r_gain = g_mean / (r_mean + eps)
    b_gain = g_mean / (b_mean + eps)
    
    # Ограничиваем усиление разумными пределами
    r_gain = torch.clamp(r_gain, 1.0 / max_gain, max_gain)
    b_gain = torch.clamp(b_gain, 1.0 / max_gain, max_gain)

    # Применяем AWB
    r *= r_gain
    b *= b_gain
    # G каналы остаются без изменений

    # Ограничиваем диапазон значений и конвертируем в uint32
    return torch.clamp(awb_frame, 0, 0xFFFFFF).to(torch.uint32)


def awb_white_world(frame: torch.Tensor, percentile: float = 99.0, max_gain: float = 4.0) -> torch.Tensor:
    """
    Применяет баланс белого (White World) к изображению формата RGGB Bayer.

    Args:
        frame: 2D тензор после denois
        percentile: процентиль для определения "белого"
        max_gain: максимально допустимый коэффициент усиления

    Returns:
        torch.Tensor: изображение в том же формате с применённым AWB
    """
    awb_frame = frame.float()

    # Извлекаем RGGB каналы
    r  = awb_frame[::2, ::2]
    gr = awb_frame[::2, 1::2]
    gb = awb_frame[1::2, ::2]
    b  = awb_frame[1::2, 1::2]

    # "Белые" значения каналов (robust white)
    r_white = torch.quantile(r, percentile / 100.0)
    g_white = 0.5 * (torch.quantile(gr, percentile / 100.0) + torch.quantile(gb, percentile / 100.0))
    b_white = torch.quantile(b, percentile / 100.0)

    # Коэффициенты усиления (нормализация к G)
    eps = 1.0
    r_gain = g_white / (r_white + eps)
    b_gain = g_white / (b_white + eps)
    
    # Ограничиваем усиление разумными пределами
    r_gain = torch.clamp(r_gain, 1.0 / max_gain, max_gain)
    b_gain = torch.clamp(b_gain, 1.0 / max_gain, max_gain)

    # Применяем AWB
    r *= r_gain
    b *= b_gain
    # G каналы остаются без изменений

    # Ограничиваем диапазон и конвертируем в uint32
    return torch.clamp(awb_frame, 0, 0xFFFFFF).to(torch.uint32)