import torch
import toml


def read_config(config_path: str) -> dict:
    """
    Читает TOML конфигурационный файл камеры
    
    Args:
        config_path (str): Путь к TOML файлу конфигурации
    
    Returns:
        dict: Словарь с конфигурацией камеры
    """
    with open(config_path, 'r') as f:
        config = toml.load(f)
        
    # Преобразуем числовые параметры в тензоры
    for key, value in config.items():
        if isinstance(value, (int, float, list)):
            config[key] = torch.tensor(value, dtype=torch.float32)
    return config