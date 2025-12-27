import torch


def decompand(frame: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Декомпандирует 12-битное изображение в 24-битное линейное.
    
    Args:
        frame: входной кадр (12-битные данные)
        config: cловарь с конфигурационными параметрами, включая таблицы компандирования
        
    Returns:
        torch.Tensor: декомпандированный кадр (24-битный)
    """
    device = frame.device
    frame = frame.to(torch.int32) 
    
    # Получаем таблицы компандирования из конфига
    compand_knee = torch.tensor(config['decompanding']['compand_knee'], dtype=torch.int32, device=device)
    compand_lut = torch.tensor(config['decompanding']['compand_lut'], dtype=torch.int32, device=device)
    
    # Создаем полную LUT таблицу для всех возможных 12-битных значений (0-4095)
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
            # Интерполяция в float, затем округление
            segment_values = torch.linspace(start_val, end_val, steps=num_points, dtype=torch.int32, device=device)
            lut_full[start_idx:end_idx] = segment_values
    
    # Последнее значение
    lut_full[compand_lut[-1].item():] = compand_knee[-1].item()
    
    # Применяем LUT
    frame_clipped = torch.clamp(frame, 0, 4095)
    
    return lut_full[frame_clipped].to(torch.uint32)