import torch
import torch.nn as nn


class DecompandBlackLevel(nn.Module):
    """
    Декомпандирует 12-битное изображение в 24-битное линейное и вычитает уровень черного из декомпандированного изображения
    """

    def __init__(self, decompand_config: dict):
        """
        Args:
            decompand_config: словарь с параметрами декомпандирования и black level
        """
        super().__init__()

        # Получаем параметры
        compand_knee = decompand_config['compand_knee']
        compand_lut = decompand_config['compand_lut']
        black_level = int(decompand_config['black_level'])

        # Определяем device из входных тензоров
        device = compand_knee.device

        # Создаем полную LUT таблицу
        lut_full = torch.zeros(4096, dtype=torch.int32, device=device)

        # Заполняем LUT таблицу интерполяцией
        for i in range(len(compand_knee) - 1):
            start_idx = compand_lut[i].item()
            end_idx = compand_lut[i + 1].item()
            start_val = compand_knee[i].item()
            end_val = compand_knee[i + 1].item()

            # Линейная интерполяция
            num_points = end_idx - start_idx
            if num_points > 0:
                segment_values = torch.linspace(start_val, end_val, steps=num_points, dtype=torch.int32, device=device)
                lut_full[start_idx:end_idx] = segment_values

        # Последнее значение
        lut_full[compand_lut[-1].item():] = compand_knee[-1].item()

        # Вычитаем black level
        lut_full = lut_full - black_level

        # Регистрируем объединённую LUT как buffer
        self.register_buffer('lut', lut_full)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет декомпандирование + black level subtraction

        Args:
            x: входной кадр (12-битные данные), shape: [H, W]

        Returns:
            torch.Tensor: обработанный кадр, shape: [H, W], dtype: int32, range: [0, 0xFFFFFF]
        """
        x_int = x.to(torch.int32)

        # Ограничиваем значения диапазоном LUT и применяем lookup
        x_clipped = torch.clamp(x_int, 0, 4095)
        output = self.lut[x_clipped]

        return output
