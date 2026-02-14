# Traditional ISP Pipeline

Полнофункциональный Image Signal Processor (ISP), реализованный на PyTorch с поддержкой GPU ускорения.

## Содержание

- [Описание](#описание)
- [Архитектура](#архитектура)
- [Использование](#использование)
- [Конфигурация](#конфигурация)
- [Этапы обработки](#этапы-обработки)
- [Структура проекта](#структура-проекта)

## Описание

 Проект представляет собой традиционный ISP pipeline, который преобразует RAW данные с камеры (Bayer pattern) в стандартный YUV формат. Данная реализация использует PyTorch для GPU-ускорения и обеспечивает обработку видео в реальном времени.

### Ключевые особенности

- **GPU-ускорение** через PyTorch CUDA
- **Полный ISP pipeline** (8 этапов обработки)
- **Real-time обработка** благодаря оптимизациям
- **Асинхронный I/O** для минимизации задержек
- **Гибкая конфигурация** через TOML файлы
- **Модульная архитектура** - каждый этап независим
- **CLI интерфейс** с подробной статистикой

## Архитектура

### Pipeline Flow

```
RAW Video (12-bit Bayer RGGB)
         ↓
[1] DecompandBlackLevel  → 24-bit linear + black level subtraction
         ↓
[2] BayerDenoise         → Guided Filter на Bayer pattern
         ↓
[3] AWB                  → Auto White Balance (Gray/White World)
         ↓
[4] Demosaic             → Malvar-He-Cutler (Bayer → RGB)
         ↓
[5] CCM                  → Color Correction Matrix
         ↓
[6] LTM                  → Local Tone Mapping (HDR → SDR)
         ↓
[7] GammaCorrection      → Gamma encoding (linear → perceptual)
         ↓
[8] RGB2YUV              → NV12 format (4:2:0 subsampling)
         ↓
YUV Video (NV12, 8-bit)
```

### Компоненты системы

- **ISP Pipeline** - основной процессор изображений
- **Config Reader** - загрузка параметров камеры из TOML
- **Video Reader** - streaming чтение RAW данных
- **Video Writer** - асинхронная запись YUV
- **Main** - CLI и оркестрация процесса

## Использование

### Базовое использование

```bash
python main.py \
    --video path/to/raw_video.bin \
    --config path/to/camera_config.toml \
    --output path/to/output.yuv
```

### Расширенное использование с параметрами

```bash
python main.py \
    --video input.bin \
    --config camera.toml \
    --output output.yuv \
    --device cuda \
    --max-frames 100 \
    --awb-method gray_world \
    --awb-max-gain 4.0 \
    --denoise-radius 2 \
    --denoise-eps 100.0 \
    --ltm-a 0.7 \
    --ltm-radius 8 \
    --gamma 2.2
```

### Параметры командной строки

#### Основные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--video` | Путь к RAW видео файлу | *обязательный* |
| `--config` | Путь к TOML конфигурации камеры | *обязательный* |
| `--output` | Путь к выходному YUV файлу | *обязательный* |
| `--device` | Устройство (`cuda` / `cpu`) | `cuda` |
| `--max-frames` | Максимальное количество кадров | `None` (все) |
| `--quiet` | Отключить вывод статистики | `False` |

#### AWB параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--awb-method` | Метод AWB (`gray_world` / `white_world`) | `gray_world` |
| `--awb-max-gain` | Максимальное усиление | `4.0` |
| `--awb-percentile` | Процентиль для white_world | `99.0` |

#### Denoise параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--denoise-radius` | Радиус фильтра | `2` |
| `--denoise-eps` | Epsilon регуляризации | `100.0` |

#### LTM параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--ltm-a` | Коэффициент сжатия диапазона | `0.7` |
| `--ltm-b` | Сдвиг яркости | `0.0` |
| `--ltm-radius` | Радиус guided filter | `8` |
| `--ltm-downsample` | Фактор downsample | `0.5` |
| `--ltm-eps` | Epsilon для GF | `1e-3` |

#### Gamma параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--gamma` | Значение гаммы | `2.2` |

## Конфигурация

### Формат TOML конфигурации

Пример файла `camera_config.toml`:

```toml
[img]
width = 1920
height = 1080
emb_lines = [0, 0]  # [top, bottom] embedded lines

[decompanding]
black_level = 64
compand_knee = [0, 256, 512, 1024, 2048, 4095, 16777215]
compand_lut = [0, 256, 512, 1024, 2048, 4095, 4096]

[ccm]
ccm_matrix = [
    [1.5, -0.3, -0.2],
    [-0.2, 1.4, -0.2],
    [-0.1, -0.5, 1.6]
]
```

### Параметры конфигурации

#### `[img]` - Параметры изображения
- `width` - ширина кадра в пикселях
- `height` - высота кадра в пикселях
- `emb_lines` - количество embedded строк [сверху, снизу]

#### `[decompanding]` - Декомпандирование
- `black_level` - уровень черного 
- `compand_knee` - контрольные точки для LUT
- `compand_lut` - индексы в LUT для каждой контрольной точки

#### `[ccm]` - Color Correction Matrix
- `ccm_matrix` - матрица 3×3 для цветокоррекции 

## Этапы обработки

### 1. DecompandBlackLevel

**Назначение:** Линеаризация данных + вычитание уровня черного

**Детали:**
- Преобразование 12-bit companded → 24-bit linear через LUT
- Вычитание black level интегрировано в LUT
- Линейная интерполяция между контрольными точками

**Вход:** `[H, W]` uint16, диапазон [0, 4095]  
**Выход:** `[H, W]` int32, диапазон [0, 0xFFFFFF]

### 2. BayerDenoise

**Назначение:** Шумоподавление с сохранением деталей

**Алгоритм:** Fast Guided Filter
- Фильтр, сохраняющий края
- Обработка каждого канала Bayer независимо (R, Gr, Gb, B)
- Батчевая обработка для эффективности

**Вход/Выход:** `[H, W]` int32 Bayer pattern

### 3. AWB (Auto White Balance)

**Назначение:** Коррекция цветовой температуры освещения

**Методы:**
- **Gray World** - предположение о среднем сером
- **White World** - использование ярких областей (percentile)

**Принцип:** Нормализация R и B каналов относительно G

**Вход/Выход:** `[H, W]` int32 Bayer pattern

### 4. Demosaic

**Назначение:** Восстановление полного RGB из Bayer pattern

**Алгоритм:** Malvar-He-Cutler
- 5×5 свертки для каждого типа интерполяции
- Учет цветовых корреляций
- Минимизация zipper и false color артефактов

**Вход:** `[H, W]` int32 Bayer RGGB  
**Выход:** `[H, W, 3]` int32 RGB

### 5. CCM (Color Correction Matrix)

**Назначение:** Преобразование цветового пространства сенсора в sRGB

**Операция:** Матричное умножение 3×3

**Вход:** `[H, W, 3]` int32, [0, 0xFFFFFF]  
**Выход:** `[H, W, 3]` float32, [0, 1]

### 6. LTM (Local Tone Mapping)

**Назначение:** Сжатие динамического диапазона 

**Алгоритм:** Base-Detail Decomposition
1. Извлечение яркости (Rec.709)
2. Log domain преобразование
3. Guided Filter для выделения базовой компоненты
4. Сжатие базовой, сохранение деталей
5. Цветовое масштабирование

**Оптимизации:**
- Separable box filter
- Downsample/Upsample для ускорения

**Вход/Выход:** `[H, W, 3]` float32, [0, 1]

### 7. GammaCorrection

**Назначение:** Гамма-кодирование для корректного отображения

**Формула:** `output = input^(1/γ)`

**Стандарт:** γ=2.2 (sRGB approximation)

**Вход/Выход:** `[H, W, 3]` float32, [0, 1]

### 8. RGB2YUV

**Назначение:** Конвертация в YUV NV12 формат

**Преобразование:** BT.709 full range

**Subsampling:** 4:2:0 через average pooling

**Формат NV12:**
- Y plane: [H×W] полное разрешение
- UV plane: [H/2×W/2] interleaved

**Вход:** `[H, W, 3]` float32, [0, 1]  
**Выход:** `[N]` uint8, где N = 1.5×H×W

## Структура проекта

```
traditional_ISP/
├── awb.py              # Auto White Balance
├── ccm.py              # Color Correction Matrix
├── config_reader.py    # TOML config parser
├── decompand.py        # Decompanding + Black Level
├── demosaic.py         # Malvar-He-Cutler demosaicing
├── denoise.py          # Guided Filter denoising
├── gamma.py            # Gamma correction
├── isp_pipeline.py     # Main ISP pipeline orchestrator
├── ltm.py              # Local Tone Mapping
├── main.py             # CLI entry point
├── rgb2yuv.py          # RGB to YUV NV12 conversion
├── video_reader.py     # RAW video streaming reader
└── video_writer.py     # Async YUV writer
```
