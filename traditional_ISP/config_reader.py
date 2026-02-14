import torch
import toml
from typing import Dict, Any


def read_config(config_path: str, device: str = 'cuda') -> Dict[str, Any]:
    """
    Читает TOML конфигурационный файл камеры и подготавливает тензоры для GPU
    
    Args:
        config_path: Путь к TOML файлу конфигурации
        device: Устройство для размещения тензоров ('cuda' или 'cpu')
    
    Returns:
        dict: Словарь с конфигурацией камеры
    """
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Определяем устройство
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    device_obj = torch.device(device)
    
    # Конвертируем массивы в тензоры
    # Decompanding параметры
    if 'decompanding' in config:
        if 'compand_knee' in config['decompanding']:
            config['decompanding']['compand_knee'] = torch.tensor(config['decompanding']['compand_knee'], 
                                                                  dtype=torch.int32, 
                                                                  device=device_obj)
        
        if 'compand_lut' in config['decompanding']:
            config['decompanding']['compand_lut'] = torch.tensor(config['decompanding']['compand_lut'], 
                                                                 dtype=torch.int32, 
                                                                 device=device_obj)
    
    # CCM матрица
    if 'ccm' in config and 'ccm_matrix' in config['ccm']:
        config['ccm']['ccm_matrix'] = torch.tensor(config['ccm']['ccm_matrix'], 
                                                   dtype=torch.float32, 
                                                   device=device_obj)
    
    return config
