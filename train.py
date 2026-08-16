"""
Training script for ZS-HPD (Zero-Shot Hybrid Prior-guided Denoising).
"""
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from network import DenoisingNetwork
from samplers import LocalPriorDownSampler, GlobalPriorRandomSampler
from loss import SpectralWeightedLoss
from utils import (load_image, save_image, add_gaussian_noise, add_poisson_noise,
                   calculate_psnr, calculate_ssim, center_crop)


def train_zs_hpd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load image
    print(f"Loading image from {args.input}...")
    img_clean = load_image(args.input, size=args.resize)

    # Center crop for standardized evaluation (if specified)
    if args.crop_size > 0:
        img_clean = center_crop(img_clean, args.crop_size)

    img_clean = img_clean.to(device)
    b, c, h, w = img_clean.shape

    # Add noise
    if args.noise_type == 'gaussian':
        img_noisy = add_gaussian_noise(img_clean, args.sigma)
        print(f"Added Gaussian noise with sigma={args.sigma}")
    elif args.noise_type == 'poisson':
        img_noisy = add_poisson_noise(img_clean, args.lam)
        print(f"Added Poisson noise with lambda={args.lam}")
    else:
        raise ValueError(f"Unknown noise type: {args.noise_type}")

    # Initialize network
    model = DenoisingNetwork(in_channels=c, num_layers=8, num_features=48).to(device)

    # Initialize samplers
    sd_sampler = LocalPriorDownSampler(aggregate_size=5, tile_size=2)
    sr_sampler = GlobalPriorRandomSampler(M=args.M, K=args.K, sigma_G=args.sigma_G, patch_size=7)

    # Build pixel bank for Sr (once at the beginning)
    print("Building pixel bank for global sampler...")
    sr_sampler.build_pixel_bank(img_noisy)

    # Initialize loss and optimizer
    criterion = SpectralWeightedLoss(radius_ratio=0.2, alpha=0.5, beta=1.0)

    # Learning rate setup
    initial_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Training loop
    num_iterations = args.iterations
    print(f"Starting training for {num_iterations} iterations...")

    model.train()
    for iteration in tqdm(range(num_iterations)):
        # Learning rate scheduling: halve at 500 and 1000
        if iteration == 500 or iteration == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"LR reduced to {optimizer.param_groups[0]['lr']}")

        optimizer.zero_grad()

        total_loss = 0.0

        # ---- Local Sampler (Sd) ----
        xl, yl, xh, yh = sd_sampler(img_noisy)

        # Train on low-gradient pair (xl -> yl)
        pred_l = model(xl)
        loss_l = criterion(pred_l, yl)
        total_loss += loss_l

        # Train on high-gradient pair (xh -> yh)
        pred_h = model(xh)
        loss_h = criterion(pred_h, yh)
        total_loss += loss_h

        # ---- Global Sampler (Sr) ----
        xp, yp = sr_sampler(img_noisy)

        # Train on pseudo pair (xp -> yp)
        pred_p = model(xp)
        loss_p = criterion(pred_p, yp)
        total_loss += loss_p

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        if (iteration + 1) % 100 == 0:
            print(f"Iter [{iteration+1}/{num_iterations}], Loss: {total_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Inference on full image
    print("Running inference on full image...")
    model.eval()
    with torch.no_grad():
        img_denoised = model(img_noisy)
        img_denoised = torch.clamp(img_denoised, 0.0, 1.0)

    # Save results
    os.makedirs(args.output, exist_ok=True)

    save_image(img_noisy, os.path.join(args.output, 'noisy.png'))
    save_image(img_denoised, os.path.join(args.output, 'denoised.png'))
    if args.input_gt:
        save_image(img_clean, os.path.join(args.output, 'clean.png'))

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output, 'model.pth'))

    # Compute metrics if ground truth is available
    if args.input_gt:
        psnr_noisy = calculate_psnr(img_noisy, img_clean)
        ssim_noisy = calculate_ssim(img_noisy, img_clean)
        psnr_denoised = calculate_psnr(img_denoised, img_clean)
        ssim_denoised = calculate_ssim(img_denoised, img_clean)

        print(f"\n{'='*50}")
        print(f"Noisy  -> PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")
        print(f"Denoised -> PSNR: {psnr_denoised:.2f} dB, SSIM: {ssim_denoised:.4f}")
        print(f"{'='*50}")

        with open(os.path.join(args.output, 'metrics.txt'), 'w') as f:
            f.write(f"Noisy PSNR: {psnr_noisy:.4f}\n")
            f.write(f"Noisy SSIM: {ssim_noisy:.4f}\n")
            f.write(f"Denoised PSNR: {psnr_denoised:.4f}\n")
            f.write(f"Denoised SSIM: {ssim_denoised:.4f}\n")

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZS-HPD Training')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--input_gt', type=str, default=None, help='Path to ground truth image (optional)')
    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'poisson'])
    parser.add_argument('--sigma', type=int, default=25, help='Gaussian noise sigma')
    parser.add_argument('--lam', type=int, default=50, help='Poisson noise lambda')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--iterations', type=int, default=1500, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--M', type=int, default=1024, help='Number of candidate pixels for Sr')
    parser.add_argument('--K', type=int, default=10, help='Number of top-K candidates for Sr')
    parser.add_argument('--sigma_G', type=int, default=10, help='Gaussian std for Sr sampling')
    parser.add_argument('--crop_size', type=int, default=256, help='Center crop size (0 to disable)')
    parser.add_argument('--resize', type=int, default=None, help='Resize image to this size (None to keep original)')

    args = parser.parse_args()
    train_zs_hpd(args)
