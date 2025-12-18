"""
DeblurDiff seed aggregation module.
Handles multiple random seeds and computes statistics across them.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List
from metrics import compute_all_metrics


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image and convert from BGR to RGB.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array in RGB format
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def evaluate_single_seed(deblurred_img: np.ndarray,
                        original_img: np.ndarray,
                        blurred_img: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single seed's deblurred result against original and blurred images.

    Args:
        deblurred_img: Deblurred image from one seed
        original_img: Original ground truth image
        blurred_img: Blurred input image

    Returns:
        Dictionary with two comparison types:
        {
            'deblurred_vs_original': {psnr, ssim, entropy_diff},
            'deblurred_vs_blurred': {psnr, ssim, entropy_diff}
        }
    """
    results = {}

    # Comparison 1: Deblurred vs Original (quality of deblurring)
    results['deblurred_vs_original'] = compute_all_metrics(
        deblurred_img, original_img, 'deblurred_vs_original'
    )

    # Comparison 2: Deblurred vs Blurred (improvement)
    results['deblurred_vs_blurred'] = compute_all_metrics(
        deblurred_img, blurred_img, 'deblurred_vs_blurred'
    )

    return results


def evaluate_all_seeds(deblurdiff_paths: Dict[int, Path],
                       original_img: np.ndarray,
                       blurred_img: np.ndarray) -> Dict:
    """
    Evaluate all 4 seeds for a single blur variant.

    Args:
        deblurdiff_paths: Dictionary mapping seed number to image path {11: Path, 42: Path, ...}
        original_img: Original ground truth image
        blurred_img: Blurred input image

    Returns:
        {
            'per_seed': {
                11: {'deblurred_vs_original': {...}, 'deblurred_vs_blurred': {...}},
                42: {...},
                127: {...},
                231: {...}
            },
            'mean': {
                'deblurred_vs_original': {psnr, ssim, entropy_diff},
                'deblurred_vs_blurred': {psnr, ssim, entropy_diff}
            },
            'std': {
                'deblurred_vs_original': {psnr, ssim, entropy_diff},
                'deblurred_vs_blurred': {psnr, ssim, entropy_diff}
            },
            'best_seed': int,
            'best_seed_metrics': {
                'deblurred_vs_original': {psnr, ssim, entropy_diff},
                'deblurred_vs_blurred': {psnr, ssim, entropy_diff}
            }
        }
    """
    if not deblurdiff_paths:
        return None

    per_seed_results = {}

    # Evaluate each seed
    for seed, deblurred_path in sorted(deblurdiff_paths.items()):
        try:
            deblurred_img = load_image(deblurred_path)
            per_seed_results[seed] = evaluate_single_seed(
                deblurred_img, original_img, blurred_img
            )
        except Exception as e:
            print(f"Warning: Failed to evaluate seed {seed}: {e}")
            continue

    if not per_seed_results:
        return None

    # Compute statistics across seeds
    comparison_types = ['deblurred_vs_original', 'deblurred_vs_blurred']
    metric_names = ['psnr', 'ssim', 'entropy_diff']

    mean_results = {}
    std_results = {}

    for comp_type in comparison_types:
        mean_results[comp_type] = {}
        std_results[comp_type] = {}

        for metric_name in metric_names:
            # Collect metric values across all seeds
            values = [
                per_seed_results[seed][comp_type][metric_name]
                for seed in per_seed_results.keys()
            ]

            mean_results[comp_type][metric_name] = float(np.mean(values))
            std_results[comp_type][metric_name] = float(np.std(values))

    # Find best seed based on PSNR for deblurred_vs_original
    best_seed = max(
        per_seed_results.keys(),
        key=lambda s: per_seed_results[s]['deblurred_vs_original']['psnr']
    )
    best_seed_metrics = per_seed_results[best_seed]

    return {
        'per_seed': per_seed_results,
        'mean': mean_results,
        'std': std_results,
        'best_seed': best_seed,
        'best_seed_metrics': best_seed_metrics
    }


def aggregate_seed_statistics(seed_results: List[Dict]) -> Dict:
    """
    Aggregate statistics across multiple blur variants.

    Args:
        seed_results: List of seed evaluation results from evaluate_all_seeds()

    Returns:
        Overall statistics including:
        - Average mean PSNR/SSIM/entropy across all variants
        - Average std dev across all variants
        - Seed frequency (which seed was best most often)
    """
    if not seed_results:
        return {}

    # Filter out None results
    valid_results = [r for r in seed_results if r is not None]
    if not valid_results:
        return {}

    comparison_types = ['deblurred_vs_original', 'deblurred_vs_blurred']
    metric_names = ['psnr', 'ssim', 'entropy_diff']

    # Aggregate mean values
    overall_mean = {}
    overall_std = {}

    for comp_type in comparison_types:
        overall_mean[comp_type] = {}
        overall_std[comp_type] = {}

        for metric_name in metric_names:
            # Average of means
            mean_values = [r['mean'][comp_type][metric_name] for r in valid_results]
            overall_mean[comp_type][metric_name] = float(np.mean(mean_values))

            # Average of std devs
            std_values = [r['std'][comp_type][metric_name] for r in valid_results]
            overall_std[comp_type][metric_name] = float(np.mean(std_values))

    # Count best seed frequency
    seed_frequency = {}
    for result in valid_results:
        best_seed = result['best_seed']
        seed_frequency[best_seed] = seed_frequency.get(best_seed, 0) + 1

    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'seed_frequency': seed_frequency,
        'num_variants': len(valid_results)
    }
