#!/usr/bin/env python3
"""
Helper script to analyze and visualize the comprehensive metrics results.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

def load_results(csv_path='results/comprehensive_metrics.csv'):
    """Load the results CSV."""
    return pd.read_csv(csv_path)

def filter_for_analysis(df, blur_category=None, deblur_method=None,
                       comparison_type=None, seed_type='mean'):
    """
    Filter dataframe for analysis.

    Args:
        df: Input dataframe
        blur_category: 'linear', 'shake', or 'angular'
        deblur_method: 'DeblurGAN' or 'DeblurDiff'
        comparison_type: 'deblurred_vs_original', 'deblurred_vs_blurred', or 'blurred_vs_original'
        seed_type: For DeblurDiff, use 'mean', 'best', or specific seed number
    """
    filtered = df.copy()

    if blur_category:
        filtered = filtered[filtered['blur_category'] == blur_category]

    if deblur_method:
        if deblur_method == 'DeblurDiff' and seed_type:
            filtered = filtered[
                (filtered['deblur_method'] == deblur_method) &
                (filtered['seed'] == seed_type)
            ]
        else:
            filtered = filtered[filtered['deblur_method'] == deblur_method]

    if comparison_type:
        filtered = filtered[filtered['comparison_type'] == comparison_type]

    return filtered

def plot_psnr_vs_blur_strength(df, blur_category='linear', save_path=None):
    """
    Plot PSNR vs blur strength for a given blur category.
    Compares DeblurGAN and DeblurDiff performance.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Filter for deblurred vs original comparison
    for method in ['DeblurGAN', 'DeblurDiff']:
        seed_val = 'mean' if method == 'DeblurDiff' else None
        method_filter = (df['deblur_method'] == method)
        if seed_val:
            method_filter = method_filter & (df['seed'] == seed_val)

        data = df[
            (df['blur_category'] == blur_category) &
            (df['comparison_type'] == 'deblurred_vs_original') &
            method_filter
        ]

        if len(data) > 0:
            grouped = data.groupby('blur_strength')['psnr'].agg(['mean', 'std'])
            ax.errorbar(grouped.index.astype(int), grouped['mean'],
                       yerr=grouped['std'], marker='o', label=method, capsize=5)

    # Add blurred vs original as baseline
    baseline = df[
        (df['blur_category'] == blur_category) &
        (df['comparison_type'] == 'blurred_vs_original')
    ]
    if len(baseline) > 0:
        grouped_baseline = baseline.groupby('blur_strength')['psnr'].mean()
        ax.plot(grouped_baseline.index.astype(int), grouped_baseline.values,
               'k--', label='Blurred (degraded)', alpha=0.5)

    ax.set_xlabel('Blur Strength (kernel size)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'PSNR vs Blur Strength - {blur_category.capitalize()} Blur')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

def plot_all_metrics_comparison(df, blur_category='linear', save_dir='results/plots'):
    """
    Create comparison plots for all three metrics (PSNR, SSIM, Entropy).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics = ['psnr', 'ssim', 'entropy_diff']
    metric_labels = ['PSNR (dB)', 'SSIM', 'Entropy Difference (bits)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        for method in ['DeblurGAN', 'DeblurDiff']:
            seed_val = 'mean' if method == 'DeblurDiff' else None
            method_filter = (df['deblur_method'] == method)
            if seed_val:
                method_filter = method_filter & (df['seed'] == seed_val)

            data = df[
                (df['blur_category'] == blur_category) &
                (df['comparison_type'] == 'deblurred_vs_original') &
                method_filter
            ]

            if len(data) > 0:
                grouped = data.groupby('blur_strength')[metric].agg(['mean', 'std'])
                ax.errorbar(grouped.index.astype(int), grouped['mean'],
                           yerr=grouped['std'], marker='o', label=method, capsize=5)

        ax.set_xlabel('Blur Strength')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Blur Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'All Metrics - {blur_category.capitalize()} Blur', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / f'{blur_category}_all_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

def compare_deblur_methods_by_blur_type(df, save_path=None):
    """
    Compare DeblurGAN vs DeblurDiff across all blur types.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    blur_categories = ['linear', 'shake', 'angular']

    for idx, blur_cat in enumerate(blur_categories):
        ax = axes[idx]

        for method in ['DeblurGAN', 'DeblurDiff']:
            seed_val = 'mean' if method == 'DeblurDiff' else None
            method_filter = (df['deblur_method'] == method)
            if seed_val:
                method_filter = method_filter & (df['seed'] == seed_val)

            data = df[
                (df['blur_category'] == blur_cat) &
                (df['comparison_type'] == 'deblurred_vs_original') &
                method_filter
            ]

            if len(data) > 0:
                grouped = data.groupby('blur_strength')['psnr'].mean()
                ax.plot(grouped.index.astype(int), grouped.values,
                       marker='o', label=method)

        ax.set_xlabel('Blur Strength')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{blur_cat.capitalize()} Blur')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Method Comparison Across Blur Types', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

def print_summary_table(df):
    """Print summary statistics table."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for blur_cat in ['linear', 'shake', 'angular']:
        print(f"\n{blur_cat.upper()} BLUR:")
        print("-" * 80)

        for method in ['DeblurGAN', 'DeblurDiff']:
            seed_val = 'mean' if method == 'DeblurDiff' else None
            method_filter = (df['deblur_method'] == method)
            if seed_val:
                method_filter = method_filter & (df['seed'] == seed_val)

            data = df[
                (df['blur_category'] == blur_cat) &
                (df['comparison_type'] == 'deblurred_vs_original') &
                method_filter
            ]

            if len(data) > 0:
                print(f"\n{method}:")
                print(f"  Average PSNR: {data['psnr'].mean():.2f} ± {data['psnr'].std():.2f} dB")
                print(f"  Average SSIM: {data['ssim'].mean():.4f} ± {data['ssim'].std():.4f}")
                print(f"  Average Entropy Diff: {data['entropy_diff'].mean():.4f} ± {data['entropy_diff'].std():.4f} bits")

if __name__ == '__main__':
    print("Loading results...")
    df = load_results()

    print(f"Loaded {len(df)} rows")
    print(f"Unique images: {df['original_image'].nunique()}")
    print(f"Blur categories: {', '.join(df['blur_category'].unique())}")

    # Print summary table
    print_summary_table(df)

    # Create plots directory
    plots_dir = Path('results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)

    # Generate plots for each blur category
    for blur_cat in ['linear', 'shake', 'angular']:
        print(f"\nGenerating plots for {blur_cat} blur...")
        plot_all_metrics_comparison(df, blur_category=blur_cat, save_dir='results/plots')

    # Generate comparison plot
    print("\nGenerating method comparison plot...")
    compare_deblur_methods_by_blur_type(df, save_path='results/plots/method_comparison.png')

    print("\n" + "="*80)
    print("Analysis complete! Check the results/plots/ directory for visualizations.")
    print("="*80)
