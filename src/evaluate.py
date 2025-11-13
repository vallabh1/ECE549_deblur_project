"""
Evaluation script for comparing deblurring models.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import json

from interfaces import DeblurGANv2
from metrics import calculate_psnr, calculate_ssim, entropy_of_difference


def load_image_pair(blur_path, sharp_path):
    """Load blurred and sharp image pair."""
    blur_img = cv2.imread(str(blur_path))
    sharp_img = cv2.imread(str(sharp_path))

    if blur_img is None or sharp_img is None:
        raise ValueError(f"Failed to load images: {blur_path}, {sharp_path}")

    # Convert BGR to RGB
    blur_rgb = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
    sharp_rgb = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

    return blur_rgb, sharp_rgb


def save_triplet(blurred, predicted, ground_truth, output_path):
    """Save triplet visualization (blur | predicted | sharp)."""
    triplet = np.hstack([blurred, predicted, ground_truth])
    triplet_bgr = cv2.cvtColor(triplet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), triplet_bgr)


def evaluate_model(model, blur_files, sharp_files, output_dir):
    """Evaluate a single model on dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    triplets_dir = output_dir / 'triplets'
    triplets_dir.mkdir(exist_ok=True)

    results = []

    print(f"Evaluating {model.model_name}...")
    for idx, (blur_path, sharp_path) in enumerate(tqdm(zip(blur_files, sharp_files))):
        blur_rgb, sharp_rgb = load_image_pair(blur_path, sharp_path)

        # Predict
        predicted_rgb = model.predict(blur_rgb)

        # Compute metrics
        psnr = calculate_psnr(predicted_rgb, sharp_rgb)
        ssim = calculate_ssim(predicted_rgb, sharp_rgb)
        entropy_diff = entropy_of_difference(predicted_rgb, blur_rgb)

        results.append({
            'image': Path(blur_path).name,
            'psnr': float(psnr),
            'ssim': float(ssim),
            'entropy_diff': float(entropy_diff)
        })

        # Save triplet
        save_triplet(blur_rgb, predicted_rgb, sharp_rgb,
                    triplets_dir / f"{Path(blur_path).stem}.png")

    # Compute averages
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_entropy = np.mean([r['entropy_diff'] for r in results])

    summary = {
        'model': model.model_name,
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
        'avg_entropy_diff': float(avg_entropy),
        'per_image': results
    }

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{model.model_name} Results:")
    print(f"  PSNR: {avg_psnr:.2f}")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Entropy Diff: {avg_entropy:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate deblurring models')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--model', type=str, default='DeblurGANv2',
                       help='Model name (default: DeblurGANv2)')
    parser.add_argument('--blur_dir', type=str, required=True,
                       help='Directory with blurred images')
    parser.add_argument('--sharp_dir', type=str, required=True,
                       help='Directory with sharp images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    if args.model == 'DeblurGANv2':
        model = DeblurGANv2()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_model(args.weights)

    # Get image pairs
    print("Loading image pairs...")
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    blur_files = sorted([
        f for f in glob(f"{args.blur_dir}/**/*.*", recursive=True)
        if Path(f).suffix.lower() in extensions
    ])
    sharp_files = sorted([
        f for f in glob(f"{args.sharp_dir}/**/*.*", recursive=True)
        if Path(f).suffix.lower() in extensions
    ])

    if len(blur_files) == 0:
        raise ValueError(f"No images found in {args.blur_dir}")

    if len(blur_files) != len(sharp_files):
        print(f"Warning: {len(blur_files)} blur != {len(sharp_files)} sharp images")
        min_len = min(len(blur_files), len(sharp_files))
        blur_files = blur_files[:min_len]
        sharp_files = sharp_files[:min_len]

    print(f"Found {len(blur_files)} image pairs")

    # Evaluate
    evaluate_model(model, blur_files, sharp_files,
                  Path(args.output_dir) / args.model)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
