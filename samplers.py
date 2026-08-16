"""
Samplers for ZS-HPD:
- Sd: Local-Prior Guided Down-Sampler
- Sr: Global-Prior Guided Random Sampler
"""
import torch
import torch.nn.functional as F
import numpy as np


class LocalPriorDownSampler:
    """
    Local-Prior Guided Down-Sampler (Sd).
    Generates down-sampled training pairs based on gradient merging and grouping.
    """
    def __init__(self, aggregate_size=5, tile_size=2):
        self.aggregate_size = aggregate_size
        self.tile_size = tile_size

    def _compute_sobel_gradient(self, img):
        """Compute Sobel gradient magnitude map."""
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                                dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                                dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        b, c, h, w = img.shape

        grad_x = []
        grad_y = []
        for i in range(c):
            gx = F.conv2d(img[:, i:i+1], sobel_x, padding=1)
            gy = F.conv2d(img[:, i:i+1], sobel_y, padding=1)
            grad_x.append(gx)
            grad_y.append(gy)

        grad_x = torch.stack(grad_x, dim=1).mean(dim=1)
        grad_y = torch.stack(grad_y, dim=1).mean(dim=1)

        return grad_x, grad_y

    def _aggregate_gradients(self, grad_x, grad_y):
        """Aggregate gradients within local region (5x5)."""
        kernel_size = self.aggregate_size
        grad_x_agg = F.avg_pool2d(grad_x.abs(), kernel_size=kernel_size, 
                                   stride=1, padding=kernel_size//2, count_include_pad=False)
        grad_y_agg = F.avg_pool2d(grad_y.abs(), kernel_size=kernel_size, 
                                   stride=1, padding=kernel_size//2, count_include_pad=False)

        grad_mag = torch.sqrt(grad_x_agg**2 + grad_y_agg**2 + 1e-8)
        return grad_mag

    def __call__(self, img):
        """
        Args:
            img: (B, C, H, W) - noisy image
        Returns:
            xl, yl, xh, yh: each (B, C, H/2, W/2) - down-sampled samples
        """
        b, c, h, w = img.shape
        assert h % 2 == 0 and w % 2 == 0, "Image dimensions must be even."

        grad_x, grad_y = self._compute_sobel_gradient(img)
        grad_mag = self._aggregate_gradients(grad_x, grad_y)

        grad_mag_tiles = grad_mag.view(b, 1, h//2, 2, w//2, 2).permute(0, 2, 4, 3, 5, 1)
        grad_mag_tiles = grad_mag_tiles.reshape(b, h//2, w//2, 4)

        img_tiles = img.view(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 3, 5, 1)
        img_tiles = img_tiles.reshape(b, h//2, w//2, 4, c)

        _, ranks = torch.sort(grad_mag_tiles, dim=-1, descending=False)

        ranks_expanded = ranks.unsqueeze(-1).expand(-1, -1, -1, -1, c)
        sorted_pixels = torch.gather(img_tiles, dim=3, index=ranks_expanded)

        xl = sorted_pixels[:, :, :, 0, :].permute(0, 3, 1, 2)
        yl = sorted_pixels[:, :, :, 1, :].permute(0, 3, 1, 2)
        xh = sorted_pixels[:, :, :, 2, :].permute(0, 3, 1, 2)
        yh = sorted_pixels[:, :, :, 3, :].permute(0, 3, 1, 2)

        return xl, yl, xh, yh


class GlobalPriorRandomSampler:
    """
    Global-Prior Guided Random Sampler (Sr).
    Constructs pseudo samples by searching throughout the entire image 
    with a Gaussian constraint.
    """
    def __init__(self, M=1024, K=10, sigma_G=10, patch_size=7):
        self.M = M
        self.K = K
        self.sigma_G = sigma_G
        self.patch_size = patch_size
        self.pixel_bank = None

    def _rgb_to_ycbcr(self, img):
        """Convert RGB to YCbCr."""
        if img.shape[1] == 1:
            return img, img

        mat = torch.tensor([[0.299, 0.587, 0.114],
                            [-0.169, -0.331, 0.5],
                            [0.5, -0.419, -0.081]], 
                           dtype=img.dtype, device=img.device)

        img_perm = img.permute(0, 2, 3, 1)
        ycbcr = torch.matmul(img_perm, mat.T)
        ycbcr[:, :, :, 1:] += 0.5

        y = ycbcr[:, :, :, 0:1].permute(0, 3, 1, 2)
        return y, ycbcr.permute(0, 3, 1, 2)

    def _extract_patches(self, y_channel):
        """Extract l x l patches from Y channel."""
        b, c, h, w = y_channel.shape
        l = self.patch_size
        padding = l // 2

        y_padded = F.pad(y_channel, (padding, padding, padding, padding), mode='reflect')
        patches = F.unfold(y_padded, kernel_size=l, padding=0)
        patches = patches.view(b, l*l, h, w)

        return patches

    def build_pixel_bank(self, img):
        """
        Build pixel bank B_p of shape (B, H, W, C, K).
        Vectorized implementation for efficiency.
        """
        b, c, h, w = img.shape
        device = img.device

        # Extract Y channel
        y_channel, _ = self._rgb_to_ycbcr(img)

        # Extract patches from Y channel: (B, l*l, H, W)
        y_patches = self._extract_patches(y_channel)
        y_patches = y_patches.view(b, -1, h*w).permute(0, 2, 1)  # (B, H*W, l*l)

        # Create coordinate grid
        coords = torch.stack(torch.meshgrid(torch.arange(h, device=device), 
                                            torch.arange(w, device=device), indexing='ij'), dim=-1)
        coords = coords.view(-1, 2).float()  # (H*W, 2)

        pixel_bank_list = []

        for bi in range(b):
            # Compute pairwise distances: (H*W, H*W)
            dists = torch.cdist(coords, coords)  # (H*W, H*W)

            # Gaussian weights: (H*W, H*W)
            gaussian_weights = torch.exp(-dists**2 / (2 * self.sigma_G**2))

            # For each pixel, sample M candidates according to Gaussian weights
            num_pixels = h * w
            M = min(self.M, num_pixels)

            # Sample candidates: (H*W, M)
            candidate_indices = torch.multinomial(gaussian_weights, num_samples=M, replacement=False)

            # Gather candidate patches: (H*W, M, l*l)
            anchor_patches = y_patches[bi].unsqueeze(1)  # (H*W, 1, l*l)
            candidate_patches = y_patches[bi][candidate_indices]  # (H*W, M, l*l)

            # Compute L1 distances: (H*W, M)
            l1_dists = torch.abs(anchor_patches - candidate_patches).sum(dim=-1)

            # Select top-K candidates
            _, topk_indices = torch.topk(l1_dists, k=min(self.K, M), largest=False, dim=1)
            selected_indices = torch.gather(candidate_indices, dim=1, index=topk_indices)  # (H*W, K)

            # Gather pixel values: (H*W, K, C)
            img_flat = img[bi].view(c, -1).permute(1, 0)  # (H*W, C)
            bank = img_flat[selected_indices]  # (H*W, K, C)

            # Reshape to (H, W, K, C)
            bank = bank.view(h, w, self.K, c)
            pixel_bank_list.append(bank)

        self.pixel_bank = torch.stack(pixel_bank_list, dim=0)  # (B, H, W, K, C)
        return self.pixel_bank

    def __call__(self, img):
        """
        Generate pseudo samples from pixel bank.
        Returns:
            x_p, y_p: pseudo sample pairs, each (B, C, H, W)
        """
        if self.pixel_bank is None:
            self.build_pixel_bank(img)

        b, c, h, w = img.shape
        device = img.device

        x_p_list = []
        y_p_list = []

        for bi in range(b):
            rand_k_x = torch.randint(0, self.K, (h, w), device=device)
            rand_k_y = torch.randint(0, self.K, (h, w), device=device)

            # Vectorized gathering
            x_p = self.pixel_bank[bi, torch.arange(h, device=device)[:, None], 
                                   torch.arange(w, device=device)[None, :], 
                                   rand_k_x, :]  # (H, W, C)
            y_p = self.pixel_bank[bi, torch.arange(h, device=device)[:, None], 
                                   torch.arange(w, device=device)[None, :], 
                                   rand_k_y, :]  # (H, W, C)

            x_p = x_p.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            y_p = y_p.permute(2, 0, 1).unsqueeze(0)

            x_p_list.append(x_p)
            y_p_list.append(y_p)

        x_p = torch.cat(x_p_list, dim=0)
        y_p = torch.cat(y_p_list, dim=0)

        return x_p, y_p

    def reset(self):
        """Reset pixel bank for new image."""
        self.pixel_bank = None
