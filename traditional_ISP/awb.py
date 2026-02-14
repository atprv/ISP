import torch
import torch.nn as nn


class AWB(nn.Module):
    """
    Auto White Balance для Bayer изображений
    """
    
    def __init__(self, method: str = 'gray_world', max_gain: float = 4.0, percentile: float = 99.0):
        """
        Args:
            method: метод AWB ('gray_world' или 'white_world')
            max_gain: максимально допустимый коэффициент усиления
            percentile: процентиль для white_world метода
        """
        super().__init__()
        
        if method not in ['gray_world', 'white_world']:
            raise ValueError(f"Unknown AWB method: {method}. Use 'gray_world' or 'white_world'")
        
        self.method = method
        self.register_buffer('max_gain', torch.tensor(max_gain, dtype=torch.float32))
        self.register_buffer('percentile', torch.tensor(percentile / 100.0, dtype=torch.float32))
        self.register_buffer('eps', torch.tensor(1.0, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет AWB к Bayer изображению
        
        Args:
            x: Bayer изображение в формате RGGB, shape: [H, W], dtype: int32, range: [0, 0xFFFFFF]
        
        Returns:
            torch.Tensor: изображение с примененным AWB, shape: [H, W], dtype: int32, range: [0, 0xFFFFFF]
        """
        x_float = x.float()
        
        # Извлекаем RGGB каналы
        r = x_float[::2, ::2]
        gr = x_float[::2, 1::2]
        gb = x_float[1::2, ::2]
        b = x_float[1::2, 1::2]
        
        if self.method == 'gray_world':
            # Средние значения каналов
            r_ref = r.mean()
            g_ref = 0.5 * (gr.mean() + gb.mean())
            b_ref = b.mean()
        else:  # white_world
            # Используем percentile как reference
            r_ref = torch.quantile(r.flatten(), self.percentile)
            g_ref = 0.5 * (torch.quantile(gr.flatten(), self.percentile) + torch.quantile(gb.flatten(), self.percentile))
            b_ref = torch.quantile(b.flatten(), self.percentile)
        
        # Вычисляем коэффициенты усиления
        r_gain = g_ref / (r_ref + self.eps)
        b_gain = g_ref / (b_ref + self.eps)
        
        # Ограничиваем усиление
        r_gain = torch.clamp(r_gain, 1.0 / self.max_gain, self.max_gain)
        b_gain = torch.clamp(b_gain, 1.0 / self.max_gain, self.max_gain)
        
        # Применяем AWB
        output = x_float.clone()
        output[::2, ::2] *= r_gain
        output[1::2, 1::2] *= b_gain
        # G каналы остаются без изменений
        
        return torch.clamp(output, 0, 0xFFFFFF).to(torch.int32)
