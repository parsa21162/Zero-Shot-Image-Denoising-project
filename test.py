"""
Inference script for ZS-HPD.
"""
import os
import argparse
import torch

from network import DenoisingNetwork
from utils import load_image, save_image, calculate_psnr, calculate_ssim


def test_zs_hpd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load noisy image
    print(f"Loading noisy image from {args.input}...")
    img_noisy = load_image(args.input, size=args.resize)
    if args.crop_size > 0:
        from utils import center_crop
        img_noisy = center_crop(img_noisy, args.crop_size)
    img_noisy = img_noisy.to(device)

    b, c, h, w = img_noisy.shape

    # Initialize model
    model = DenoisingNetwork(in_channels=c, num_layers=8, num_features=48).to(device)

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("Warning: No checkpoint provided. Using randomly initialized network.")

    model.eval()

    # Inference
    print("Running inference...")
    with torch.no_grad():
        img_denoised = model(img_noisy)
        img_denoised = torch.clamp(img_denoised, 0.0, 1.0)

    # Save result
    os.makedirs(args.output, exist_ok=True)
    save_image(img_denoised, os.path.join(args.output, 'denoised.png'))
    print(f"Denoised image saved to {os.path.join(args.output, 'denoised.png')}")

    # Compute metrics if ground truth is available
    if args.input_gt:
        img_clean = load_image(args.input_gt, size=args.resize)
        if args.crop_size > 0:
            from utils import center_crop
            img_clean = center_crop(img_clean, args.crop_size)
        img_clean = img_clean.to(device)

        psnr = calculate_psnr(img_denoised, img_clean)
        ssim = calculate_ssim(img_denoised, img_clean)
        print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZS-HPD Inference')
    parser.add_argument('--input', type=str, required=True, help='Path to noisy input image')
    parser.add_argument('--input_gt', type=str, default=None, help='Path to ground truth image (optional)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--crop_size', type=int, default=0, help='Center crop size (0 to disable)')
    parser.add_argument('--resize', type=int, default=None, help='Resize image to this size')

    args = parser.parse_args()
    test_zs_hpd(args)
