import torch
import torch.nn as nn

from decompand import DecompandBlackLevel
from denoise import BayerDenoise
from awb import AWB
from demosaic import Demosaic
from ccm import CCM
from ltm import LTM
from gamma import GammaCorrection
from rgb2yuv import RGBtoYUV


class ISPPipeline(nn.Module):
    """
    Полный ISP Pipeline для обработки RAW изображений
    """
    
    def __init__(self, config: dict, device: str = 'cuda', **params):
        """
        Инициализация ISP pipeline
        
        Args:
            config: словарь конфигурации камеры
            device: устройство для вычислений ('cuda' или 'cpu')
            **params: параметры для ISP блоков:
                - awb_method: метод баланса белого ('gray_world' или 'white_world')
                - awb_max_gain: максимальное усиление для AWB (по умолчанию 4.0)
                - awb_percentile: процентиль для white_world AWB (по умолчанию 99.0)
                - denoise_radius: радиус guided filter для denoise (по умолчанию 2)
                - denoise_eps: epsilon для denoise (по умолчанию 100.0)
                - ltm_a: коэффициент сжатия для ltm (по умолчанию 0.7)
                - ltm_b: сдвиг яркости для ltm (по умолчанию 0.0)
                - ltm_radius: радиус guided filter (по умолчанию 8)
                - ltm_downsample: downsample factor для LTM (по умолчанию 0.5)
                - ltm_eps: epsilon для guided filter (по умолчанию 1e-3)
                - gamma: значение гаммы (по умолчанию 2.2)
        """
        super().__init__()
        
        # Определяем device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        
        # Значения по умолчанию
        defaults = {'awb_method': 'gray_world',
                    'awb_max_gain': 4.0,
                    'awb_percentile': 99.0,
                    'denoise_radius': 2, 
                    'denoise_eps': 100.0,
                    'ltm_a': 0.7,
                    'ltm_b': 0.0,
                    'ltm_radius': 8,
                    'ltm_downsample': 0.5,
                    'ltm_eps': 1e-3,
                    'gamma': 2.2}
        
        # Объединяем с пользовательскими параметрами
        self.params = {**defaults, **params}
        
        # Создаем все модули ISP pipeline
        # 1. Decompand + Black Level
        self.decompand_blacklevel = DecompandBlackLevel(config['decompanding'])
        
        # 2. Bayer Denoise
        self.denoise = BayerDenoise(radius=self.params['denoise_radius'],
                                    eps=self.params['denoise_eps'])
        
        # 3. Auto White Balance
        self.awb = AWB(method=self.params['awb_method'],
                       max_gain=self.params['awb_max_gain'],
                       percentile=self.params['awb_percentile'])
        
        # 4. Demosaic
        self.demosaic = Demosaic()
        
        # 5. Color Correction Matrix
        self.ccm = CCM(config['ccm'])
        
        # 6. Local Tone Mapping
        self.ltm = LTM(a=self.params['ltm_a'],
                       b=self.params['ltm_b'],
                       radius=self.params['ltm_radius'],
                       eps=self.params['ltm_eps'],
                       downsample_factor=self.params['ltm_downsample'])
        
        # 7. Gamma Correction
        self.gamma = GammaCorrection(gamma=self.params['gamma'])
        
        # 8. RGB to YUV
        self.rgb2yuv = RGBtoYUV()
        
        # Перемещаем все на нужное устройство
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Обрабатывает один RAW кадр через полный ISP pipeline
        
        Args:
            x: RAW кадр в формате Bayer RGGB, shape: [H, W], dtype: uint16
        
        Returns:
            torch.Tensor: обработанный YUV кадр в формате NV12, 1D тензор uint8
        """
        x = x.to(self.device)
        
        # Pipeline stages
        x = self.decompand_blacklevel(x)      # [H, W] uint16 -> [H, W] int32
        x = self.denoise(x)                   # [H, W] int32 -> [H, W] int32
        x = self.awb(x)                       # [H, W] int32 -> [H, W] int32
        x = self.demosaic(x)                  # [H, W] int32 -> [H, W, 3] int32
        x = self.ccm(x)                       # [H, W, 3] int32 -> [H, W, 3] float32 [0,1]
        x = self.ltm(x)                       # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]
        x = self.gamma(x)                     # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]
        x = self.rgb2yuv(x)                   # [H, W, 3] float32 -> [N] uint8 (NV12)
        
        return x
    
    def get_pipeline_info(self) -> dict:
        """
        Возвращает информацию о конфигурации pipeline
        
        Returns:
            dict: параметры всех модулей
        """
        return {'device': str(self.device),
                'parameters': self.params,
                'modules': ['DecompandBlackLevel',
                            'BayerDenoise',
                            'AWB',
                            'Demosaic',
                            'CCM',
                            'LTM',
                            'GammaCorrection',
                            'RGBtoYUV']}
