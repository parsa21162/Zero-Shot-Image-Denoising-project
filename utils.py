"""
Utility functions for ZS-HPD.
"""
import torch
import numpy as np
from PIL import Image
import os


def load_image(path, size=None):
    """Load image and normalize to [0, 1]."""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return img_tensor


def save_image(tensor, path):
    """Save tensor as image."""
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(path)


def add_gaussian_noise(img, sigma):
    """Add Gaussian noise."""
    noise = torch.randn_like(img) * (sigma / 255.0)
    noisy = img + noise
    return torch.clamp(noisy, 0.0, 1.0)


def add_poisson_noise(img, lam):
    """Add Poisson noise."""
    # Scale to [0, lam], apply Poisson, scale back
    img_scaled = img * lam
    noisy = torch.poisson(img_scaled) / lam
    return torch.clamp(noisy, 0.0, 1.0)


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """Calculate SSIM between two images."""
    # Simple SSIM implementation
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1*img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean().item()


def center_crop(img, crop_size=256):
    """Center crop image to specified size."""
    b, c, h, w = img.shape
    if h <= crop_size and w <= crop_size:
        return img
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[:, :, top:top+crop_size, left:left+crop_size]
