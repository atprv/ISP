import torch


def gamma_correction(frame: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    Применяет гамма-коррекцию к RGB изображению.
    
    Args:
        frame: RGB изображение (H × W × 3) после LTM, float в диапазоне [0, 1]
        gamma: показатель гаммы (2.2 для sRGB, 2.4 для Rec.709)

    Returns:
        torch.Tensor: гамма-кодированное изображение (H × W × 3), диапазон [0, 1]
    """
    
    # Применение гамма-кривой: out = in^(1/gamma)
    frame_gamma = frame.pow(1.0 / gamma)
    
    return frame_gamma