import torch
from pathlib import Path
from config_reader import read_config
from video_reader import read_video_frames_generator
from isp_pipeline import ISPPipeline
from rgb2yuv import save_yuv
import time



def process_video(video_path: str, config_path: str, output_path: str, max_frames: int = None, 
                  verbose: bool = True, device: str = 'cuda', **isp_params):
    """
    Обрабатывает RAW видео через ISP pipeline
    
    Args:
        video_path: путь к RAW видео файлу (.bin)
        config_path: путь к конфигурационному файлу камеры (.toml)
        output_path: путь к выходному YUV файлу
        max_frames: максимальное количество кадров для обработки (None = все)
        verbose: выводить информацию о прогрессе
        **isp_params: параметры для ISP pipeline:
            - awb_method: метод баланса белого ('gray_world' или 'white_world')
            - awb_max_gain: максимальное усиление для AWB (по умолчанию 4.0)
            - awb_percentile: процентиль для white_world AWB (по умолчанию 99.5)
            - denoise_kernel: размер ядра для denoise (по умолчанию 3)
            - ltm_a: коэффициент сжатия для ltm (по умолчанию 0.7)
            - ltm_b: сдвиг яркости для ltm (по умолчанию 0.0)
            - ltm_radius: радиус guided filter (по умолчанию 32)
            - gamma: значение гаммы (по умолчанию 2.2)
    """
    
    # Читаем конфигурацию
    if verbose:
        print(f"Loading configuration from {config_path}...")
    config = read_config(config_path)
    
    # Создаем ISP pipeline
    if verbose:
        if device == 'cuda' and torch.cuda.is_available():
            print(f"Initializing ISP Pipeline on CUDA: {torch.cuda.get_device_name(0)}...")
        else:
            print(f"Initializing ISP Pipeline on CPU...")
        if isp_params:
            print(f"ISP Parameters: {isp_params}")
    isp = ISPPipeline(config, device=device, **isp_params)
    
    # Обрабатываем видео покадрово
    if verbose:
        print(f"\nProcessing video: {video_path}")
        print(f"Output: {output_path}\n")
    
    frame_count = 0
    start_time = time.perf_counter()
    for raw_frame, frame_number in read_video_frames_generator(video_path, config):
        frame_start = time.perf_counter()
        
        # Обработка через ISP
        yuv_frame = isp.process_frame(raw_frame, verbose=verbose)
        
        # Сохранение
        append = (frame_number > 1)  # Первый кадр перезаписывает файл
        save_yuv(yuv_frame, output_path, append=append)
        
        frame_count += 1
        frame_time = time.perf_counter() - frame_start
        
        if verbose:
            print(f"Frame {frame_number} saved.")
        
        # Ограничение по количеству кадров
        if max_frames and frame_count >= max_frames:
            if verbose:
                print(f"\nReached max_frames limit ({max_frames})")
                print(f"Frame time: {frame_time:.3f} s | FPS: {1.0 / frame_time:.2f}")
            break
        
    total_time = time.perf_counter() - start_time
    avg_fps = frame_count / total_time
    
    if verbose:
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}")


def main():
    """
    Пример использования ISP pipeline
    """
    
    # Параметры
    video_path = ""       # Путь к RAW-видео
    config_path = ""      # Путь к конфигурационному файлу камеры
    output_path = ""      # Путь к выходному YUV-файлу
    
    # Проверяем доступность CUDA
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA available: {torch.cuda.get_device_name(0)}\n")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU\n")
    
    # Обработка видео
    process_video(video_path=video_path,
                  config_path=config_path,
                  output_path=output_path,
                  awb_method='white_world',
                  max_frames=50,
                  verbose=True,
                  device=device)


if __name__ == "__main__":
    main()