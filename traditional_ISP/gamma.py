import torch
import torch.nn as nn


class GammaCorrection(nn.Module):
    """
    Гамма-коррекция для RGB изображений
    """
    
    def __init__(self, gamma: float = 2.2):
        """
        Args:
            gamma: показатель гаммы (2.2 для sRGB, 2.4 для Rec.709)
        """
        super().__init__()
        
        inv_gamma = 1.0 / gamma
        self.register_buffer('inv_gamma', torch.tensor(inv_gamma, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет гамма-коррекцию
        
        Args:
            x: RGB изображение, shape: [H, W, 3], dtype: float32, range: [0, 1]
        
        Returns:
            torch.Tensor: гамма-кодированное изображение, shape: [H, W, 3], dtype: float32, range: [0, 1]
        """
        # Применяем степенное преобразование
        output = x.pow(self.inv_gamma)
        
        # Ограничиваем диапазон
        return torch.clamp(output, 0.0, 1.0)
