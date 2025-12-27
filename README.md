# Traditional ISP Pipeline

A complete Image Signal Processor (ISP) pipeline implementation in PyTorch with CUDA support for processing RAW camera sensor data.

## Overview

This project implements a traditional ISP pipeline that processes RAW Bayer pattern video streams from camera sensors into YUV video output. The pipeline processes video frame-by-frame, applying all essential stages of image signal processing with optimized algorithms and GPU acceleration.

## Pipeline Stages

1. **Decompanding** - Convert 12-bit companded data to 24-bit linear
2. **Black Level Subtraction** - Remove sensor dark current offset
3. **Bayer Denoise** - Median filtering for noise reduction
4. **Auto White Balance (AWB)** - Gray World / White World algorithms
5. **Demosaicing** - Malvar-He-Cutler algorithm (Bayer → RGB)
6. **Color Correction Matrix (CCM)** - Color space transformation
7. **Local Tone Mapping (LTM)** - Dynamic range compression with guided filter
8. **Gamma Correction** - Non-linear encoding (sRGB/Rec.709)
9. **RGB to YUV** - Convert to NV12 format

## Input Format

- **Video**: RAW binary file (.bin) with Bayer RGGB pattern
- **Config**: TOML file with camera parameters (sensor specs, decompanding tables, CCM, etc.)

## Output Format

- **YUV NV12** format (4:2:0 chroma subsampling)
- BT.709 color space, full range
- 8-bit per channel

## Key Algorithms

### Auto White Balance
- **Gray World**: Assumes average scene color is gray
- **White World**: Uses high percentile values as white reference

### Demosaicing
- **Malvar-He-Cutler**: High-quality edge-aware interpolation

### Local Tone Mapping
- **Guided Filter**: Edge-preserving smoothing
- **Integral Image**: O(1) box filtering for any radius size
- **Base-Detail Decomposition**: Preserves textures while compressing dynamic range
