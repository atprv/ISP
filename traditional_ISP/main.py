import torch
import time
import argparse
from pathlib import Path

from config_reader import read_config
from video_reader import RAWVideoReader
from isp_pipeline import ISPPipeline
from video_writer import AsyncYUVWriter


def process_video(video_path: str, config_path: str, output_path: str, 
                  max_frames: int = None, verbose: bool = True, 
                  device: str = 'cuda', **isp_params):
    """
    Обрабатывает RAW видео через ISP pipeline
    
    Args:
        video_path: путь к RAW видео файлу (.bin)
        config_path: путь к конфигурационному файлу камеры (.toml)
        output_path: путь к выходному YUV файлу
        max_frames: максимальное количество кадров для обработки (None = все)
        verbose: выводить информацию о прогрессе
        device: устройство ('cuda' или 'cpu')
        **isp_params: параметры для ISP pipeline
    """
    
    # Проверяем доступность CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    # Читаем конфигурацию
    if verbose:
        print(f"Loading configuration from {config_path}...")
    config = read_config(config_path, device=device)
    
    # Создаем ISP pipeline
    if verbose:
        if device == 'cuda':
            print(f"Initializing ISP Pipeline on CUDA: {torch.cuda.get_device_name(0)}...")
        else:
            print(f"Initializing ISP Pipeline on CPU...")
        if isp_params:
            print(f"ISP Parameters: {isp_params}")
    
    isp = ISPPipeline(config, device=device, **isp_params)
    
    # Переводим в eval режим
    isp.eval()
    
    if verbose:
        info = isp.get_pipeline_info()
        print(f"\nPipeline Info:")
        print(f"  Device: {info['device']}")
        print(f"  Modules: {len(info['modules'])}")
        print(f"\nProcessing video: {video_path}")
        print(f"Output: {output_path}\n")
    
    # Счетчики
    frame_count = 0
    total_isp_time = 0.0
    total_write_time = 0.0
    
    # Обработка с асинхронным writer
    with RAWVideoReader(video_path, config, device=device) as reader, AsyncYUVWriter(output_path) as writer:
        
        with torch.no_grad():
            # Делаем warmup чтобы не портить статистику
            if verbose:
                print("Warming up CUDA kernels (compiling on first run)...")
                
            warmup_frame, _ = reader.read_frame()
            if warmup_frame is not None:
                _ = isp(warmup_frame)
                if device == 'cuda':
                    torch.cuda.synchronize() 
            
            if verbose:
                print("Warmup complete. Starting processing...\n")
            
            # Сброс reader в начало после warmup
            reader.file_handle.seek(0)

            start_time = time.perf_counter()
            
            for raw_frame, frame_number in reader:
                isp_start = time.perf_counter()
                
                # Обработка через ISP
                yuv_frame = isp(raw_frame)
                
                # Синхронизация CUDA для точных измерений
                if device == 'cuda':
                    torch.cuda.synchronize()
                    
                isp_time = time.perf_counter() - isp_start
                total_isp_time += isp_time
                
                write_start = time.perf_counter()
                
                # Переносим на CPU перед отправкой в writer
                if yuv_frame.is_cuda:
                    yuv_frame_cpu = yuv_frame.cpu()
                else:
                    yuv_frame_cpu = yuv_frame
                
                # Асинхронная запись
                writer.write(yuv_frame_cpu)
                
                write_time = time.perf_counter() - write_start
                total_write_time += write_time
                
                frame_count += 1
                
                if verbose and frame_count % 10 == 0:
                    instant_fps = 1.0 / isp_time

                    print(f"Frame {frame_number:4d} | "
                          f"ISP: {isp_time*1000:6.2f}ms | "
                          f"Write: {write_time*1000:5.2f}ms | "
                          f"Instant FPS: {instant_fps:5.1f} ")
                
                # Ограничение по количеству кадров
                if max_frames and frame_count >= max_frames:
                    if verbose:
                        print(f"\nReached max_frames limit ({max_frames})")
                    break
    
    # Финальная статистика
    total_elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
    avg_isp_time = total_isp_time / frame_count if frame_count > 0 else 0
    avg_write_time = total_write_time / frame_count if frame_count > 0 else 0
    
    if verbose:
        print(f"\n{'='*80}")
        print("Processing complete!")
        print(f"{'='*80}")
        print(f"Total frames processed:    {frame_count}")
        print(f"Total time:                {total_elapsed:.2f} s")
        print(f"")
        print(f"Average ISP time:          {avg_isp_time*1000:.2f} ms/frame")
        print(f"Average write time:        {avg_write_time*1000:.2f} ms/frame")
        print(f"Average total time:        {(avg_isp_time + avg_write_time)*1000:.2f} ms/frame")
        print(f"")
        print(f"ISP throughput:            {1.0/avg_isp_time:.2f} FPS (pure processing)")
        print(f"Overall throughput:        {avg_fps:.2f} FPS (with I/O)")
        print(f"")
        print(f"Output saved to: {output_path}")
        print(f"{'='*80}")
    
    return {'frames': frame_count,
            'total_time': total_elapsed,
            'avg_fps': avg_fps,
            'avg_isp_time': avg_isp_time,
            'avg_write_time': avg_write_time,
            'isp_fps': 1.0/avg_isp_time if avg_isp_time > 0 else 0}

def main():
    """
    CLI для ISP pipeline
    """
    parser = argparse.ArgumentParser(description='ISP Pipeline for RAW video processing')
    
    # Основные аргументы
    parser.add_argument('--video', type=str, required=True, help='Path to RAW video file')
    parser.add_argument('--config', type=str, required=True, help='Path to camera config (TOML)')
    parser.add_argument('--output', type=str, required=True, help='Path to output YUV file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process (default: all)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    # ISP параметры
    parser.add_argument('--awb-method', type=str, default='gray_world', choices=['gray_world', 'white_world'], help='AWB method (default: gray_world)')
    parser.add_argument('--awb-max-gain', type=float, default=4.0, help='Max AWB gain (default: 4.0)')
    parser.add_argument('--awb-percentile', type=float, default=99.0, help='Percentile for white_world AWB (default: 99.0)')
    parser.add_argument('--denoise-radius', type=int, default=2, help='Denoise filter radius (default: 2)')
    parser.add_argument('--denoise-eps', type=float, default=100.0, help='Denoise epsilon (default: 100.0)')
    parser.add_argument('--ltm-a', type=float, default=0.7, help='LTM compression coefficient (default: 0.7)')
    parser.add_argument('--ltm-b', type=float, default=0.0, help='LTM brightness shift (default: 0.0)')
    parser.add_argument('--ltm-radius', type=int, default=8, help='LTM guided filter radius (default: 8)')
    parser.add_argument('--ltm-downsample', type=float, default=0.5, help='Downsample factor for LTM (default: 0.5)')
    parser.add_argument('--ltm-eps', type=float, default=1e-3, help='Epsilon for guided filter in LTM (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=2.2, help='Gamma value (default: 2.2)')
    
    args = parser.parse_args()
    
    # Проверяем существование файлов
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Собираем ISP параметры
    isp_params = {# AWB параметры
                  'awb_method': args.awb_method,
                  'awb_max_gain': args.awb_max_gain,
                  'awb_percentile': args.awb_percentile,
                  
                  # Denoise параметры
                  'denoise_radius': args.denoise_radius,
                  'denoise_eps': args.denoise_eps,
                  
                  # LTM параметры
                  'ltm_a': args.ltm_a,
                  'ltm_b': args.ltm_b,
                  'ltm_radius': args.ltm_radius,
                  'ltm_downsample': args.ltm_downsample,
                  'ltm_eps': args.ltm_eps,
                  
                  # Gamma параметры
                  'gamma': args.gamma}
    
    # Обработка
    process_video(video_path=args.video,
                  config_path=args.config,
                  output_path=args.output,
                  max_frames=args.max_frames,
                  verbose=not args.quiet,
                  device=args.device,
                  **isp_params)


if __name__ == "__main__":
    main()
