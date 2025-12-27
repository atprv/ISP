import torch
from decompand import decompand
from black_level import subtract_black_level
from denoise import bayer_denoise
from awb import awb_gray_world, awb_white_world
from demosaic import demosaic
from ccm import color_correction_matrix
from ltm import local_tone_mapping
from gamma import gamma_correction
from rgb2yuv import rgb_to_yuv


class ISPPipeline:
    """
    Полный ISP Pipeline для обработки RAW изображений
    """
    
    def __init__(self, config: dict, **params):
        """
        Инициализация ISP pipeline
        
        Args:
            config: словарь конфигурации камеры
            **params: параметры для ISP блоков:
                - awb_method: метод баланса белого ('gray_world' или 'white_world')
                - awb_max_gain: максимальное усиление для AWB (по умолчанию 4.0)
                - awb_percentile: процентиль для white_world AWB (по умолчанию 99.5)
                - denoise_kernel: размер ядра для denoise (по умолчанию 3)
                - ltm_a: коэффициент сжатия для ltm (по умолчанию 0.7)
                - ltm_b: сдвиг яркости для ltm (по умолчанию 0.0)
                - ltm_radius: радиус guided filter (по умолчанию 32)
                - gamma: значение гаммы (по умолчанию 2.2)
        """
        self.config = config
        
        # Значения по умолчанию
        self.defaults = {'awb_method': 'gray_world',
                         'awb_max_gain': 4.0,
                         'awb_percentile': 99.5,
                         'denoise_kernel': 3,
                         'ltm_a': 0.7,
                         'ltm_b': 0.0,
                         'ltm_radius': 32,
                         'gamma': 2.2}
        
        # Параметры с значениями по умолчанию
        self.params = {**self.defaults, **params}
        
        # Создаем атрибуты для удобного доступа
        for key, value in self.params.items():
            setattr(self, key, value)
        
    def process_frame(self, raw_frame: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Обрабатывает один RAW кадр через полный ISP pipeline
        
        Args:
            raw_frame: RAW кадр в формате Bayer RGGB (H x W), uint16
            verbose: выводить информацию о каждом этапе
            
        Returns:
            torch.Tensor: обработанный YUV кадр в формате NV12 (1D тензор)
        """
        
        # 1. Decompand (12-bit → 24-bit linear)
        frame = decompand(raw_frame, self.config)
        
        # 2. Subtract Black Level
        frame = subtract_black_level(frame, self.config)
        
        # 3. Bayer Denoise
        frame = bayer_denoise(frame, kernel_size=self.denoise_kernel)
        
        # 4. Auto White Balance
        if self.awb_method == 'gray_world':
            frame = awb_gray_world(frame, max_gain=self.awb_max_gain)
        elif self.awb_method == 'white_world':
            frame = awb_white_world(frame, percentile=self.awb_percentile, max_gain=self.awb_max_gain)
        else:
            raise ValueError(f"Unknown AWB method: {self.awb_method}")
        
        # 5. Demosaic (Bayer → RGB)
        frame = demosaic(frame)
        
        # Normalize to [0, 1]
        frame = frame.float() / frame.float().max()
        
        # 6. Color Correction Matrix
        frame = color_correction_matrix(frame, self.config)
        
        # 7. Local Tone Mapping
        frame = local_tone_mapping(frame, a=self.ltm_a, b=self.ltm_b, radius=self.ltm_radius)
        
        # 8. Gamma Correction
        frame = gamma_correction(frame, gamma=self.gamma)
        
        # 9. RGB → YUV 
        yuv_frame = rgb_to_yuv(frame)
        
        return yuv_frame
    
    def set_awb_method(self, method: str):
        """Изменить метод баланса белого"""
        if method not in ['gray_world', 'white_world']:
            raise ValueError(f"Unknown AWB method: {method}")
        self.awb_method = method
        
      
    def update_params(self, **params):
        """
        Обновить параметры ISP на лету
        
        Args:
            **params: любые параметры из __init__
        """
        for key, value in params.items():
            if key in self.defaults:
                self.params[key] = value
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}. Available: {list(self.defaults.keys())}")