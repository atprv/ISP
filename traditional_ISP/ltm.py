import torch
import torch.nn as nn
import torch.nn.functional as F


class LTM(nn.Module):
    """
    Local Tone Mapping с Fast Guided Filter
    """

    def __init__(self, a: float = 0.7, b: float = 0.0, radius: int = 8, eps: float = 1e-3, downsample_factor: float = 0.5):
        """
        Args:
            a: коэффициент сжатия динамического диапазона
            b: сдвиг яркости в лог-домене
            radius: радиус guided filter
            eps: регуляризующий параметр
            downsample_factor: коэффициент downsampling 
        """
        super().__init__()

        self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
        self.radius = radius
        self.downsample_factor = downsample_factor
        self.register_buffer('eps', torch.tensor(eps, dtype=torch.float32))

        # Константы для log domain
        self.register_buffer('eps_log', torch.tensor(1e-6, dtype=torch.float32))
        self.register_buffer('eps_scale', torch.tensor(1e-4, dtype=torch.float32))

        # Предвычисляем separable box filter kernels
        kernel_size = 2 * radius + 1

        # 1D box kernel
        box_1d = torch.ones(1, 1, 1, kernel_size, dtype=torch.float32) / kernel_size
        self.register_buffer('box_h', box_1d)  # Horizontal
        self.register_buffer('box_v', box_1d.transpose(2, 3))  # Vertical

        self.pad = radius

    def _separable_box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast box filter через separable convolution

        Args:
            x: входной тензор [1, 1, H, W]

        Returns:
            torch.Tensor: отфильтрованный тензор [1, 1, H, W]
        """
        # Padding
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        # Separable convolution: horizontal then vertical
        x_h = F.conv2d(x_padded, self.box_h)
        x_hv = F.conv2d(x_h, self.box_v)

        return x_hv

    def _fast_guided_filter(self, I: torch.Tensor) -> torch.Tensor:
        """
        Fast guided filter с separable box filter

        Args:
            I: входное изображение [H, W]

        Returns:
            torch.Tensor: отфильтрованное изображение [H, W]
        """
        # Добавляем batch и channel
        I_4d = I.unsqueeze(0).unsqueeze(0)

        # Средние значения через fast box filter
        mean_I = self._separable_box_filter(I_4d)
        mean_II = self._separable_box_filter(I_4d * I_4d)

        # Дисперсия
        var_I = mean_II - mean_I * mean_I

        # Коэффициенты линейной модели
        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I

        # Усреднение коэффициентов
        mean_a = self._separable_box_filter(a)
        mean_b = self._separable_box_filter(b)

        # Выходное изображение
        out = mean_a * I_4d + mean_b

        return out.squeeze(0).squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет Local Tone Mapping к RGB изображению

        Args:
            x: RGB изображение, shape: [H, W, 3], dtype: float32, range: [0, 1]

        Returns:
            torch.Tensor: tone-mapped изображение, shape: [H, W, 3], dtype: float32, range: [0, 1]
        """
        H, W = x.shape[0], x.shape[1]

        # Разделяем каналы
        R = x[..., 0]
        G = x[..., 1]
        B = x[..., 2]

        # Вычисляем яркость (Rec.709)
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

        # Защита от нуля + log domain
        Y_safe = torch.clamp(Y, min=self.eps_log)
        Y_log = torch.log2(Y_safe)

        if self.downsample_factor < 1.0:
            # Downsample для ускорения guided filter
            Y_log_4d = Y_log.unsqueeze(0).unsqueeze(0)
            Y_log_down = F.interpolate(Y_log_4d,
                                       scale_factor=self.downsample_factor,
                                       mode='bilinear',
                                       align_corners=False)
            Y_log_down_2d = Y_log_down.squeeze(0).squeeze(0) 

            # Guided filter на downsampled версии
            Y_base_down = self._fast_guided_filter(Y_log_down_2d)

            # Upscale обратно к исходному размеру
            Y_base_4d = Y_base_down.unsqueeze(0).unsqueeze(0)  
            Y_base_up = F.interpolate(Y_base_4d,
                                      size=(H, W),
                                      mode='bilinear',
                                      align_corners=False)
            Y_base = Y_base_up.squeeze(0).squeeze(0) 
        else:
            # Без downsampling
            Y_base = self._fast_guided_filter(Y_log)

        # Detail layer
        Y_detail = Y_log - Y_base

        # Tone mapping базовой компоненты
        Y_base_tm = self.a * Y_base + self.b

        # Восстановление с деталями
        Y_tm = Y_base_tm + Y_detail
        Y_out = torch.pow(2.0, Y_tm)

        # Цветовое масштабирование
        scale = Y_out / (Y + self.eps_scale)

        # Применяем к каждому каналу
        output = torch.empty_like(x)
        output[..., 0] = R * scale
        output[..., 1] = G * scale
        output[..., 2] = B * scale

        return torch.clamp(output, 0.0, 1.0)
