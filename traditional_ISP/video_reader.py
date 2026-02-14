import torch
import numpy as np
from typing import Tuple


class RAWVideoReader:
    """
    Чтение RAW видео файлов
    """
    
    def __init__(self, video_path: str, config: dict, device: str = 'cuda'):
        """
        Args:
            video_path: путь к бинарному RAW видео файлу
            config: конфигурация камеры (dict)
            device: устройство для размещения данных
        """
        self.video_path = video_path
        self.config = config
        
        # Определяем device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)
        
        # Параметры из конфигурации
        self.width = config['img']['width']
        self.height = config['img']['height']
        self.emb_lines = config['img']['emb_lines']
        
        # Вычисляем размеры
        self.pixels_per_frame = self.width * self.height
        self.bytes_per_pixel = 2  # 16-bit
        self.frame_size_bytes = self.pixels_per_frame * self.bytes_per_pixel
        
        # Размеры после удаления emb_lines
        top_lines, bottom_lines = self.emb_lines
        self.output_height = self.height - top_lines - bottom_lines
        self.output_width = self.width
        
        # Открываем файл
        self.file_handle = None
    
    def __enter__(self):
        self.file_handle = open(self.video_path, 'rb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle is not None:
            self.file_handle.close()
    
    def read_frame(self) -> Tuple[torch.Tensor, bool]:
        """
        Читает один кадр из видео
        
        Returns:
            tuple: (frame_tensor, success)
                - frame_tensor: [H, W] uint16 тензор на self.device
                - success: True если кадр прочитан, False если конец файла
        """
        if self.file_handle is None:
            raise RuntimeError("RAWVideoReader must be used as context manager")
        
        # Читаем байты кадра
        frame_bytes = self.file_handle.read(self.frame_size_bytes)
        
        # Проверяем конец файла
        if len(frame_bytes) < self.frame_size_bytes:
            return None, False
        
        # Конвертация bytes -> tensor
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint16)
        
        # Конвертируем в torch tensor
        frame_1d = torch.from_numpy(frame_np.copy()) 
        
        # Reshape в 2D
        frame_2d = frame_1d.reshape(self.height, self.width)
        
        # Удаляем embedded lines
        top_lines, bottom_lines = self.emb_lines
        if top_lines > 0:
            frame_2d = frame_2d[top_lines:, :]
        if bottom_lines > 0:
            frame_2d = frame_2d[:-bottom_lines, :]
        
        # Перемещаем на целевое устройство
        frame_2d = frame_2d.to(self.device)
        
        return frame_2d, True
    
    def __iter__(self):
        if self.file_handle is None:
            raise RuntimeError("RAWVideoReader must be used as context manager")
        
        # Сбрасываем позицию в начало
        self.file_handle.seek(0)
        frame_number = 0
        
        while True:
            frame, success = self.read_frame()
            if not success:
                break
            
            frame_number += 1
            yield frame, frame_number


def read_video_frames_generator(video_path: str, config: dict, device: str = 'cuda'):
    """
    Генератор для построчного чтения RAW видео покадрово
    
    Args:
        video_path: путь к бинарному файлу видео
        config: конфигурация камеры
        device: устройство для данных
    
    Yields:
        tuple: (frame_tensor, frame_number)
    """
    with RAWVideoReader(video_path, config, device) as reader:
        yield from reader
