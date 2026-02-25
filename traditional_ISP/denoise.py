import torch
import torch.nn as nn
import torch.nn.functional as F


class BayerDenoise(nn.Module):
    """
    Шумоподавление для Bayer изображений
    """

    def __init__(self, radius: int = 2, eps: float = 100.0):
        """
        Args:
            radius: радиус фильтра
            eps: регуляризация
        """
        super().__init__()

        self.radius = radius
        self.register_buffer('eps', torch.tensor(eps, dtype=torch.float32))

        # Создаем box filter kernel для быстрого усреднения
        kernel_size = 2 * radius + 1
        box_kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32) / (kernel_size ** 2)
        self.register_buffer('box_kernel', box_kernel)

        self.pad = radius

    def _fast_box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Быстрый box filter через convolution

        Args:
            x: входной тензор [B, 1, H, W]

        Returns:
            torch.Tensor: отфильтрованный тензор [B, 1, H, W]
        """
        # Используем conv2d с предвычисленным kernel
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(x_padded, self.box_kernel)

    def _guided_filter_batch(self, I_batch: torch.Tensor) -> torch.Tensor:
        """
        Guided filter для batch каналов

        Args:
            I_batch: входной batch [B, 1, H, W]

        Returns:
            torch.Tensor: отфильтрованный batch [B, 1, H, W]
        """
        # Вычисляем статистики через box filter
        mean_I = self._fast_box_filter(I_batch)
        mean_II = self._fast_box_filter(I_batch * I_batch)

        # Дисперсия
        var_I = mean_II - mean_I * mean_I

        # Коэффициенты линейной модели
        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I

        # Усредняем коэффициенты
        mean_a = self._fast_box_filter(a)
        mean_b = self._fast_box_filter(b)

        # Выходное изображение
        out = mean_a * I_batch + mean_b

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет fast denoise к Bayer изображению

        Args:
            x: Bayer изображение RGGB, shape: [H, W], dtype: float32, range: [0, 0xFFFFFF]

        Returns:
            torch.Tensor: денойзированное изображение, shape: [H, W], dtype: float32, range: [0, 0xFFFFFF]
        """
        # Извлекаем каналы из Bayer паттерна
        r = x[::2, ::2]
        gr = x[::2, 1::2]
        gb = x[1::2, ::2]
        b = x[1::2, 1::2]

        # Объединяем все 4 канала в batch
        channels = torch.stack([r, gr, gb, b], dim=0).unsqueeze(1)

        # Применяем guided filter ко всем каналам одновременно
        filtered_batch = self._guided_filter_batch(channels)

        # Извлекаем отфильтрованные каналы обратно
        r_filtered = filtered_batch[0, 0]
        gr_filtered = filtered_batch[1, 0]
        gb_filtered = filtered_batch[2, 0]
        b_filtered = filtered_batch[3, 0]

        # Собираем обратно в Bayer паттерн
        output = torch.empty_like(x, dtype=torch.float32)
        output[::2, ::2] = r_filtered
        output[::2, 1::2] = gr_filtered
        output[1::2, ::2] = gb_filtered
        output[1::2, 1::2] = b_filtered

        return output
