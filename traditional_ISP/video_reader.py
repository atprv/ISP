import torch


def read_video_frames_generator(video_path: str, config: dict):
    """
    Генератор для построчного чтения RAW видео покадрово.
    
    Args:
        video_path: путь к бинарному файлу видео
        config: конфигурация камеры
    
    Yields:
        torch.Tensor: кадр в формате uint16
    """
    # Параметры из конфигурации
    width = config['img']['width']
    height = config['img']['height']
    emb_lines = config['img']['emb_lines']
    
    # Вычисляем размер одного кадра в байтах (16 бит на пиксель = 2 байта)
    pixels_per_frame = width * height
    bytes_per_pixel = 2
    frame_size_bytes = pixels_per_frame * bytes_per_pixel

    with open(video_path, 'rb') as f:
        frame_number = 0
        while True:
            frame_bytes = f.read(frame_size_bytes)
            
            # Достигнут конец файла
            if len(frame_bytes) < frame_size_bytes:
                break

            # Преобразуем в 1D тензор uint16
            frame_1d = torch.tensor(list(frame_bytes), dtype=torch.uint16)
            
            # Объединяем два байта в один uint16 (little-endian)
            frame_1d = frame_1d.reshape(-1, 2)
            frame_1d = frame_1d[:,0].to(torch.int32) + (frame_1d[:,1].to(torch.int32) << 8)
            frame_1d = frame_1d.to(torch.uint16)

            # Формируем 2D кадр
            frame_2d = frame_1d.reshape(height, width)

            # Удаляем строки согласно emb_lines
            top_lines, bottom_lines = emb_lines
            if top_lines > 0:
                frame_2d = frame_2d[top_lines:, :]
            if bottom_lines > 0:
                frame_2d = frame_2d[:-bottom_lines, :]

            frame_number += 1
            yield frame_2d, frame_number