import torch


def subtract_black_level(frame: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Вычитает уровень черного из декомпандированного изображения.
    
    Args:
        frame: декомпандированный кадр (24-битный)
        config: словарь с конфигурационными параметрами
        
    Returns:
        torch.Tensor: кадр с вычтенным уровнем черного
    """
    black_level = int(config['decompanding']['black_level'])
    frame = torch.clamp(frame.to(torch.int32) - black_level, min=0).to(torch.uint32)
    
    return frame