"""
Spectral Weighted Loss (SWL) for ZS-HPD.
Works in the Fourier domain with discriminative weights for LF and HF bands.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralWeightedLoss(nn.Module):
    def __init__(self, radius_ratio=0.2, alpha=0.5, beta=1.0):
        """
        Args:
            radius_ratio: normalized radius for spectrum division (default: 0.2)
            alpha: weight for low-frequency band (default: 0.5)
            beta: weight for high-frequency band (default: 1.0)
        """
        super(SpectralWeightedLoss, self).__init__()
        self.radius_ratio = radius_ratio
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='sum')

    def _create_circular_mask(self, h, w, radius, device):
        """Create binary circular mask. 1 inside circle, 0 outside."""
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h, device=device), 
                               torch.arange(w, device=device), indexing='ij')
        dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        mask = (dist <= radius).float()
        return mask

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            loss: scalar
        """
        b, c, h, w = pred.shape
        device = pred.device

        # Compute radius
        radius = self.radius_ratio * min(h, w)

        # Create masks
        mask_lf = self._create_circular_mask(h, w, radius, device)  # (H, W)
        mask_hf = 1.0 - mask_lf

        # Expand masks for batch and channel dimensions
        mask_lf = mask_lf.view(1, 1, h, w).expand(b, c, h, w)
        mask_hf = mask_hf.view(1, 1, h, w).expand(b, c, h, w)

        # Convert to Fourier domain
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # Apply masks in frequency domain
        pred_lf = pred_fft * mask_lf
        pred_hf = pred_fft * mask_hf
        target_lf = target_fft * mask_lf
        target_hf = target_fft * mask_hf

        # Convert back to spatial domain for MSE computation
        pred_lf_spatial = torch.fft.ifft2(pred_lf, dim=(-2, -1)).real
        pred_hf_spatial = torch.fft.ifft2(pred_hf, dim=(-2, -1)).real
        target_lf_spatial = torch.fft.ifft2(target_lf, dim=(-2, -1)).real
        target_hf_spatial = torch.fft.ifft2(target_hf, dim=(-2, -1)).real

        # Compute losses
        loss_lf = F.mse_loss(pred_lf_spatial, target_lf_spatial, reduction='sum') / (b * c * h * w)
        loss_hf = F.mse_loss(pred_hf_spatial, target_hf_spatial, reduction='sum') / (b * c * h * w)

        loss = self.alpha * loss_lf + self.beta * loss_hf

        return loss
