import torch
import torch.nn as nn


class CCM(nn.Module):
    """
    Color Correction Matrix для преобразования цветового пространства
    """

    def __init__(self, ccm_config: dict):
        """
        Args:
            ccm_config: словарь с параметрами CCM
        """
        super().__init__()

        # Получаем CCM матрицу
        ccm_matrix = ccm_config['ccm_matrix']

        # Транспонируем матрицу
        ccm_transposed = ccm_matrix.T

        # Регистрируем как buffer
        self.register_buffer('ccm', ccm_transposed)

        # Максимальное значение для 24-bit
        self.register_buffer('max_val', torch.tensor(0xFFFFFF, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет CCM к RGB изображению

        Args:
            x: RGB изображение, shape: [H, W, 3], dtype: float32, range: [0, 0xFFFFFF]

        Returns:
            torch.Tensor: RGB изображение после CCM, shape: [H, W, 3], dtype: float32, range: [0, 1]
        """
        H, W, _ = x.shape

        # Нормализуем к [0, 1] для применения CCM
        x_norm = x / self.max_val

        # Применяем CCM
        x_flat = x_norm.reshape(-1, 3)
        x_ccm = x_flat @ self.ccm

        # Возвращаем форму
        x_out = x_ccm.reshape(H, W, 3)
        
        # Возвращаем в диапазоне [0, 1] для следующих этапов
        return torch.clamp(x_out, 0.0, 1.0)
