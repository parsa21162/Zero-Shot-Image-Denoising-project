# ZS-HPD: Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation

PyTorch implementation of the CVPR 2026 paper **"Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation"**.

## Overview

ZS-HPD is a zero-shot image denoising method that does not require any external training data. It leverages:
- **Local Prior (Sd)**: Gradient-based down-sampler that preserves spatial locality
- **Global Prior (Sr)**: Gaussian-conditioned random sampler that captures non-local self-similarity
- **Spectral Weighted Loss (SWL)**: Frequency-domain loss that discriminatively handles low/high-frequency components

## Project Structure

```
zs_hpd/
├── network.py          # 8-layer denoising CNN
├── samplers.py         # Local (Sd) and Global (Sr) samplers
├── loss.py             # Spectral Weighted Loss (SWL)
├── utils.py            # Image I/O, noise generation, metrics
├── train.py            # Training script
├── test.py             # Inference script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training (Zero-Shot Denoising)

#### Gaussian Noise
```bash
python train.py \
    --input path/to/image.png \
    --noise_type gaussian \
    --sigma 25 \
    --output ./results/gaussian_25 \
    --iterations 1500 \
    --lr 1e-3 \
    --crop_size 256
```

#### Poisson Noise
```bash
python train.py \
    --input path/to/image.png \
    --noise_type poisson \
    --lam 50 \
    --output ./results/poisson_50 \
    --iterations 1500 \
    --lr 1e-3 \
    --crop_size 256
```

#### With Ground Truth (for evaluation)
```bash
python train.py \
    --input path/to/noisy.png \
    --input_gt path/to/clean.png \
    --noise_type gaussian \
    --sigma 25 \
    --output ./results/eval \
    --crop_size 256
```

### Inference

If you have a saved checkpoint:
```bash
python test.py \
    --input path/to/noisy.png \
    --checkpoint ./results/model.pth \
    --output ./results/inference
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 1500 | Total training iterations |
| `--lr` | 1e-3 | Initial learning rate (5e-4 for real-world noise) |
| `--M` | 1024 | Number of candidate pixels for global sampler |
| `--K` | 10 | Top-K similar candidates per pixel |
| `--sigma_G` | 10 | Gaussian std for spatial weighting in Sr |
| `--crop_size` | 256 | Center crop size for evaluation |

## Implementation Details

- **Network**: 8-layer fully convolutional network with 48 channels and 3x3 kernels
- **Optimizer**: Adam with learning rate halved at iterations 500 and 1000
- **Sd**: Aggregates Sobel gradients in 5x5 windows, sorts within 2x2 tiles
- **Sr**: Uses Y-channel patches (7x7) for similarity, Gaussian distance weighting
- **SWL**: Binary circular mask with radius ratio 0.2, weights alpha=0.5, beta=1.0

## Citation

```bibtex
@inproceedings{zhao2026zeroshot,
  title={Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation},
  author={Zhao, Xiaole and Pang, Qingsong and Zhang, Xiaobo and Xu, Xun and Gong, Xun and Yang, Yan and Li, Tianrui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Notes

- This is a zero-shot method: each image is trained independently from scratch.
- Training takes ~20-30 seconds per 256x256 image on a modern GPU.
- For best results on real-world noise, use `--lr 5e-4`.
