#!/usr/bin/env python3
"""
Comprehensive evaluation pipeline for deblurring methods.
Computes PSNR, SSIM, and entropy metrics across all blur variants and deblurring methods.
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from image_matcher import build_image_mapping, get_all_blur_variants, print_mapping_summary
from seed_aggregator import load_image, evaluate_all_seeds
from results_builder import ResultsBuilder
from metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of deblurring methods with multiple metrics'
    )
    parser.add_argument('--original_dir', type=str,
                       default='data/original',
                       help='Directory with original images')
    parser.add_argument('--blurred_dir', type=str,
                       default='data/blurred',
                       help='Directory with blurred images')
    parser.add_argument('--deblurgan_dir', type=str,
                       default='data/deblurgan_result_images',
                       help='Directory with DeblurGAN results')
    parser.add_argument('--deblurdiff_dir', type=str,
                       default='data/deblurdiff_result_images',
                       help='Directory with DeblurDiff results')
    parser.add_argument('--output_dir', type=str,
                       default='results',
                       help='Output directory for results')
    parser.add_argument('--output_csv', type=str,
                       default='comprehensive_metrics.csv',
                       help='Output CSV filename')
    parser.add_argument('--skip_deblurgan', action='store_true',
                       help='Skip DeblurGAN evaluation')
    parser.add_argument('--skip_deblurdiff', action='store_true',
                       help='Skip DeblurDiff evaluation')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / 'evaluation.log'
    log_messages = []

    def log(message):
        """Log message to both console and file."""
        print(message)
        log_messages.append(message)

    log("=" * 60)
    log("Comprehensive Deblurring Evaluation")
    log("=" * 60)
    log(f"Original images: {args.original_dir}")
    log(f"Blurred images: {args.blurred_dir}")
    log(f"DeblurGAN results: {args.deblurgan_dir}")
    log(f"DeblurDiff results: {args.deblurdiff_dir}")
    log(f"Output directory: {args.output_dir}")
    log("")

    start_time = time.time()

    # Build image mapping
    log("Building image mapping...")
    deblurgan_dir = None if args.skip_deblurgan else Path(args.deblurgan_dir)
    deblurdiff_dir = None if args.skip_deblurdiff else Path(args.deblurdiff_dir)

    mapping = build_image_mapping(
        Path(args.original_dir),
        Path(args.blurred_dir),
        deblurgan_dir,
        deblurdiff_dir
    )

    print_mapping_summary(mapping)

    # Get all blur variants
    variants = get_all_blur_variants(mapping)
    log(f"Processing {len(variants)} blur variants...")
    log("")

    # Initialize results builder
    results = ResultsBuilder()

    # Track errors
    errors = []

    # Process each blur variant
    for base_name, variant_key, variant_data in tqdm(variants, desc="Evaluating"):
        try:
            # Load images
            original_path = variant_data['original_path']
            blurred_path = variant_data['blurred_path']
            blur_metadata = variant_data['blur_metadata']

            blurred_img = load_image(blurred_path)
            original_img = load_image(original_path)

            # Resize original to match blurred dimensions (models output at smaller resolution)
            if original_img.shape != blurred_img.shape:
                original_img = cv2.resize(original_img,
                                         (blurred_img.shape[1], blurred_img.shape[0]),
                                         interpolation=cv2.INTER_LANCZOS4)

            # 1. Blurred vs Original (degradation)
            metrics_blur_vs_orig = compute_all_metrics(
                blurred_img, original_img, 'blurred_vs_original'
            )
            results.add_blurred_vs_original(
                base_name, blur_metadata, metrics_blur_vs_orig
            )

            # 2. DeblurGAN evaluation
            if variant_data['deblurgan_path'] and not args.skip_deblurgan:
                try:
                    deblurgan_img = load_image(variant_data['deblurgan_path'])

                    # DeblurGAN vs Original
                    metrics_gan_vs_orig = compute_all_metrics(
                        deblurgan_img, original_img, 'deblurred_vs_original'
                    )
                    results.add_deblurgan_result(
                        base_name, blur_metadata, 'deblurred_vs_original', metrics_gan_vs_orig
                    )

                    # DeblurGAN vs Blurred
                    metrics_gan_vs_blur = compute_all_metrics(
                        deblurgan_img, blurred_img, 'deblurred_vs_blurred'
                    )
                    results.add_deblurgan_result(
                        base_name, blur_metadata, 'deblurred_vs_blurred', metrics_gan_vs_blur
                    )

                except Exception as e:
                    error_msg = f"Error evaluating DeblurGAN for {variant_key}: {e}"
                    errors.append(error_msg)
                    log(f"Warning: {error_msg}")

            # 3. DeblurDiff evaluation (all seeds)
            if variant_data['deblurdiff_paths'] and not args.skip_deblurdiff:
                try:
                    seed_results = evaluate_all_seeds(
                        variant_data['deblurdiff_paths'],
                        original_img,
                        blurred_img
                    )

                    if seed_results:
                        # Add individual seed results
                        for seed, seed_metrics in seed_results['per_seed'].items():
                            # Seed vs Original
                            results.add_deblurdiff_seed_result(
                                base_name, blur_metadata, 'deblurred_vs_original',
                                seed, seed_metrics['deblurred_vs_original']
                            )
                            # Seed vs Blurred
                            results.add_deblurdiff_seed_result(
                                base_name, blur_metadata, 'deblurred_vs_blurred',
                                seed, seed_metrics['deblurred_vs_blurred']
                            )

                        # Add mean results
                        results.add_deblurdiff_mean_result(
                            base_name, blur_metadata, 'deblurred_vs_original',
                            seed_results['mean']['deblurred_vs_original'],
                            seed_results['std']['deblurred_vs_original']
                        )
                        results.add_deblurdiff_mean_result(
                            base_name, blur_metadata, 'deblurred_vs_blurred',
                            seed_results['mean']['deblurred_vs_blurred'],
                            seed_results['std']['deblurred_vs_blurred']
                        )

                        # Add best seed results
                        best_seed = seed_results['best_seed']
                        results.add_deblurdiff_best_result(
                            base_name, blur_metadata, 'deblurred_vs_original',
                            best_seed, seed_results['best_seed_metrics']['deblurred_vs_original']
                        )
                        results.add_deblurdiff_best_result(
                            base_name, blur_metadata, 'deblurred_vs_blurred',
                            best_seed, seed_results['best_seed_metrics']['deblurred_vs_blurred']
                        )

                except Exception as e:
                    error_msg = f"Error evaluating DeblurDiff for {variant_key}: {e}"
                    errors.append(error_msg)
                    log(f"Warning: {error_msg}")

        except Exception as e:
            error_msg = f"Error processing {variant_key}: {e}"
            errors.append(error_msg)
            log(f"Error: {error_msg}")
            continue

    # Save results
    log("")
    log("Saving results...")
    output_csv_path = output_dir / args.output_csv
    results.save_csv(str(output_csv_path))

    # Get and save summary statistics
    summary = results.get_summary_statistics()
    summary_path = output_dir / 'comprehensive_metrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved to: {summary_path}")

    # Print summary
    log("")
    log("=" * 60)
    log("Evaluation Summary")
    log("=" * 60)
    log(f"Total rows in CSV: {summary['total_rows']}")
    log(f"Original images: {summary['num_original_images']}")
    log(f"Blur variants: {summary['num_blur_variants']}")
    log("")
    log("Blur categories:")
    for category, count in summary['blur_categories'].items():
        log(f"  {category}: {count}")
    log("")

    # Print average metrics
    if 'average_metrics' in summary:
        log("Average Metrics by Method:")
        for method, comparisons in summary['average_metrics'].items():
            log(f"\n{method}:")
            for comp_type, metrics in comparisons.items():
                log(f"  {comp_type}:")
                log(f"    PSNR: {metrics['avg_psnr']:.2f} dB")
                log(f"    SSIM: {metrics['avg_ssim']:.4f}")
                log(f"    Entropy Diff: {metrics['avg_entropy_diff']:.4f} bits")

    # Print errors if any
    if errors:
        log("")
        log("=" * 60)
        log(f"Encountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            log(f"  - {error}")
        if len(errors) > 10:
            log(f"  ... and {len(errors) - 10} more")

    # Computation time
    elapsed_time = time.time() - start_time
    log("")
    log(f"Evaluation completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    log("=" * 60)

    # Save log
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_messages))
    print(f"\nLog saved to: {log_file}")

    # Instructions for using the results
    print("\n" + "=" * 60)
    print("How to use the results:")
    print("=" * 60)
    print(f"import pandas as pd")
    print(f"df = pd.read_csv('{output_csv_path}')")
    print("")
    print("# Example: Plot PSNR vs blur strength for linear blur, DeblurGAN")
    print("df_filtered = df[")
    print("    (df['blur_category'] == 'linear') &")
    print("    (df['deblur_method'] == 'DeblurGAN') &")
    print("    (df['comparison_type'] == 'deblurred_vs_original')")
    print("]")
    print("df_filtered.groupby('blur_strength')['psnr'].mean().plot()")
    print("=" * 60)


if __name__ == '__main__':
    main()
