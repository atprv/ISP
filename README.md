# Traditional ISP Pipeline

An Image Signal Processor (ISP) implemented in PyTorch with GPU acceleration support.

## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Processing Stages](#processing-stages)
- [Project Structure](#project-structure)

## Description

This project represents a traditional ISP pipeline that converts RAW camera data (Bayer pattern) into standard YUV format. This implementation uses PyTorch for GPU acceleration and provides real-time video processing.

### Key Features

- **GPU acceleration** via PyTorch CUDA
- **Complete ISP pipeline** (8 processing stages)
- **Real-time processing** 
- **Asynchronous I/O** to minimize latency
- **Flexible configuration** via TOML files
- **Modular architecture** - each stage is independent
- **CLI interface** with detailed statistics

## Architecture

### Pipeline Flow

```
RAW Video (12-bit Bayer RGGB)
         ↓
[1] DecompandBlackLevel  → 24-bit linear + black level subtraction
         ↓
[2] BayerDenoise         → Guided Filter on Bayer pattern
         ↓
[3] AWB                  → Auto White Balance (Gray/White World)
         ↓
[4] Demosaic             → Malvar-He-Cutler (Bayer → RGB)
         ↓
[5] CCM                  → Color Correction Matrix
         ↓
[6] LTM                  → Local Tone Mapping
         ↓
[7] GammaCorrection      → Gamma encoding
         ↓
[8] RGB2YUV              → NV12 format (4:2:0 subsampling)
         ↓
YUV Video (NV12, 8-bit)
```

### System Components

- **ISP Pipeline** - main image processor
- **Config Reader** - loads camera parameters from TOML
- **Video Reader** - streaming RAW data reading
- **Video Writer** - asynchronous YUV writing
- **Main** - CLI and process orchestration

## Usage

### Basic Usage

```bash
python main.py \
    --video path/to/raw_video.bin \
    --config path/to/camera_config.toml \
    --output path/to/output.yuv
```

### Advanced Usage with Parameters

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

### Command Line Parameters

#### Main Parameters

| Parameter | Description | Default |
|----------|----------|--------------|
| `--video` | Path to RAW video file | *required* |
| `--config` | Path to camera TOML configuration | *required* |
| `--output` | Path to output YUV file | *required* |
| `--device` | Device (`cuda` / `cpu`) | `cuda` |
| `--max-frames` | Maximum number of frames | `None` (all) |
| `--quiet` | Disable statistics output | `False` |

#### AWB Parameters

| Parameter | Description | Default |
|----------|----------|--------------|
| `--awb-method` | AWB method (`gray_world` / `white_world`) | `gray_world` |
| `--awb-max-gain` | Maximum gain | `4.0` |
| `--awb-percentile` | Percentile for white_world | `99.0` |

#### Denoise Parameters

| Parameter | Description | Default |
|----------|----------|--------------|
| `--denoise-radius` | Filter radius | `2` |
| `--denoise-eps` | Regularization epsilon | `100.0` |

#### LTM Parameters

| Parameter | Description | Default |
|----------|----------|--------------|
| `--ltm-a` | Dynamic range compression coefficient | `0.7` |
| `--ltm-b` | Brightness shift | `0.0` |
| `--ltm-radius` | Guided filter radius | `8` |
| `--ltm-downsample` | Downsample factor | `0.5` |
| `--ltm-eps` | Epsilon for GF | `1e-3` |

#### Gamma Parameters

| Parameter | Description | Default |
|----------|----------|--------------|
| `--gamma` | Gamma value | `2.2` |

## Configuration

### TOML Configuration Format

Example `camera_config.toml` file:

```toml
[img]
width = 1920
height = 1080
emb_lines = [0, 0] 

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

### Configuration Parameters

#### `[img]` - Image Parameters
- `width` - frame width in pixels
- `height` - frame height in pixels
- `emb_lines` - number of embedded lines [top, bottom]

#### `[decompanding]` - Decompanding
- `black_level` - black level
- `compand_knee` - control points for LUT
- `compand_lut` - LUT indices for each control point

#### `[ccm]` - Color Correction Matrix
- `ccm_matrix` - 3×3 matrix for color correction

## Processing Stages

### 1. DecompandBlackLevel

**Purpose:** Data linearization + black level subtraction

**Details:**
- Conversion from 12-bit companded to 24-bit linear via LUT
- Black level subtraction integrated into LUT
- Linear interpolation between control points

**Input:** `[H, W]` uint16, range [0, 4095]  
**Output:** `[H, W]` int32, range [0, 0xFFFFFF]

### 2. BayerDenoise

**Purpose:** Noise reduction while preserving details

**Algorithm:** Fast Guided Filter
- Edge-preserving filter
- Each Bayer channel processed independently (R, Gr, Gb, B)
- Batch processing for efficiency

**Input/Output:** `[H, W]` int32 Bayer pattern

### 3. AWB (Auto White Balance)

**Purpose:** Illumination color temperature correction

**Methods:**
- **Gray World** - assumes average gray
- **White World** - uses bright regions (percentile)

**Principle:** Normalize R and B channels relative to G

**Input/Output:** `[H, W]` int32 Bayer pattern

### 4. Demosaic

**Purpose:** Reconstruct full RGB from Bayer pattern

**Algorithm:** Malvar-He-Cutler
- 5×5 convolutions for each interpolation type
- Accounts for color correlations
- Minimizes zipper and false color artifacts

**Input:** `[H, W]` int32 Bayer RGGB  
**Output:** `[H, W, 3]` int32 RGB

### 5. CCM (Color Correction Matrix)

**Purpose:** Convert sensor color space to sRGB

**Operation:** 3×3 matrix multiplication

**Input:** `[H, W, 3]` int32, [0, 0xFFFFFF]  
**Output:** `[H, W, 3]` float32, [0, 1]

### 6. LTM (Local Tone Mapping)

**Purpose:** Dynamic range compression

**Algorithm:** Base-Detail Decomposition
1. Extract luminance (Rec.709)
2. Log domain transformation
3. Guided Filter to extract base component
4. Compress base, preserve details
5. Color scaling

**Optimizations:**
- Separable box filter
- Downsample/Upsample for speedup

**Input/Output:** `[H, W, 3]` float32, [0, 1]

### 7. GammaCorrection

**Purpose:** Gamma encoding for correct display

**Formula:** `output = input^(1/γ)`

**Standard:** γ=2.2 (sRGB approximation)

**Input/Output:** `[H, W, 3]` float32, [0, 1]

### 8. RGB2YUV

**Purpose:** Convert to YUV NV12 format

**Conversion:** BT.709 full range

**Subsampling:** 4:2:0 via average pooling

**NV12 Format:**
- Y plane: [H×W] full resolution
- UV plane: [H/2×W/2] interleaved

**Input:** `[H, W, 3]` float32, [0, 1]  
**Output:** `[N]` uint8, where N = 1.5×H×W

## Project Structure

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
