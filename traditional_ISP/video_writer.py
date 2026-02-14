import torch
import numpy as np
from queue import Queue
from threading import Thread
from typing import Optional


class AsyncYUVWriter:
    """
    Асинхронный writer для YUV файлов
    """
    
    def __init__(self, output_path: str, queue_size: int = 10):
        """
        Args:
            output_path: путь к выходному YUV файлу
            queue_size: размер очереди кадров
        """
        self.output_path = output_path
        self.queue = Queue(maxsize=queue_size)
        self.writer_thread = None
        self.is_running = False
        self.file_handle = None
    
    def _writer_worker(self):
        """Worker thread для записи кадров"""
        with open(self.output_path, 'wb') as f:
            self.file_handle = f
            
            while self.is_running or not self.queue.empty():
                try:
                    # Получаем кадр из очереди
                    yuv_frame = self.queue.get(timeout=0.1)
                    
                    if yuv_frame is None:  # Сигнал завершения
                        break
                    
                    # Переносим на CPU если нужно
                    if yuv_frame.is_cuda:
                        yuv_frame = yuv_frame.cpu()
                    
                    # Конвертируем в numpy и пишем
                    yuv_np = yuv_frame.numpy()
                    f.write(yuv_np.tobytes())
                    
                    self.queue.task_done()
                    
                except:
                    continue
        
        self.file_handle = None
    
    def start(self):
        """Запускает writer thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.writer_thread = Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
    
    def write(self, yuv_frame: torch.Tensor):
        """
        Добавляет кадр в очередь на запись
        
        Args:
            yuv_frame: YUV кадр в формате NV12, uint8 тензор
        """
        if not self.is_running:
            raise RuntimeError("Writer not started. Call start() first.")
        
        # Добавляем в очередь
        self.queue.put(yuv_frame)
    
    def finish(self):
        """Завершает запись и ждет окончания"""
        if not self.is_running:
            return
        
        # Сигнал завершения
        self.queue.put(None)
        
        # Ждем завершения потока
        self.is_running = False
        if self.writer_thread is not None:
            self.writer_thread.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def save_yuv(yuv_frame: torch.Tensor, filename: str, append: bool = True):
    """
    Синхронная функция сохранения YUV кадра
    
    Args:
        yuv_frame: YUV тензор (uint8, 1D, размер H*W*3/2)
        filename: имя выходного файла
        append: если True, кадр добавляется к файлу
    """
    mode = 'ab' if append else 'wb'
    
    with open(filename, mode) as f:
        # Переносим на CPU если нужно
        if yuv_frame.is_cuda:
            yuv_frame = yuv_frame.cpu()
        
        yuv_np = yuv_frame.numpy()
        f.write(yuv_np.tobytes())