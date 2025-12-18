"""
Results builder for constructing pandas DataFrame from evaluation results.
"""

import pandas as pd
from typing import List, Dict, Optional


class ResultsBuilder:
    """Build structured results for CSV export."""

    def __init__(self):
        """Initialize empty results list."""
        self.rows = []

    def add_blurred_vs_original(self,
                                original_name: str,
                                blur_metadata: Dict,
                                metrics: Dict):
        """
        Add row for blurred vs original comparison.

        Args:
            original_name: Name of original image
            blur_metadata: Dictionary with blur_category, blur_type, blur_strength, blur_direction
            metrics: Dictionary with psnr, ssim, entropy_diff
        """
        row = {
            'original_image': original_name,
            'blur_category': blur_metadata['blur_category'],
            'blur_type': blur_metadata['blur_type'],
            'blur_strength': blur_metadata['blur_strength'],
            'blur_direction': blur_metadata['blur_direction'],
            'deblur_method': None,
            'comparison_type': 'blurred_vs_original',
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'entropy_diff': metrics['entropy_diff'],
            'seed': None,
            'seed_std_psnr': None,
            'seed_std_ssim': None,
            'seed_std_entropy': None,
            'best_seed': None
        }
        self.rows.append(row)

    def add_deblurgan_result(self,
                            original_name: str,
                            blur_metadata: Dict,
                            comparison_type: str,
                            metrics: Dict):
        """
        Add row for DeblurGAN result.

        Args:
            original_name: Name of original image
            blur_metadata: Dictionary with blur information
            comparison_type: 'deblurred_vs_original' or 'deblurred_vs_blurred'
            metrics: Dictionary with psnr, ssim, entropy_diff
        """
        row = {
            'original_image': original_name,
            'blur_category': blur_metadata['blur_category'],
            'blur_type': blur_metadata['blur_type'],
            'blur_strength': blur_metadata['blur_strength'],
            'blur_direction': blur_metadata['blur_direction'],
            'deblur_method': 'DeblurGAN',
            'comparison_type': comparison_type,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'entropy_diff': metrics['entropy_diff'],
            'seed': None,
            'seed_std_psnr': None,
            'seed_std_ssim': None,
            'seed_std_entropy': None,
            'best_seed': None
        }
        self.rows.append(row)

    def add_deblurdiff_seed_result(self,
                                   original_name: str,
                                   blur_metadata: Dict,
                                   comparison_type: str,
                                   seed: int,
                                   metrics: Dict):
        """
        Add row for individual DeblurDiff seed result.

        Args:
            original_name: Name of original image
            blur_metadata: Dictionary with blur information
            comparison_type: 'deblurred_vs_original' or 'deblurred_vs_blurred'
            seed: Seed number (11, 42, 127, 231)
            metrics: Dictionary with psnr, ssim, entropy_diff
        """
        row = {
            'original_image': original_name,
            'blur_category': blur_metadata['blur_category'],
            'blur_type': blur_metadata['blur_type'],
            'blur_strength': blur_metadata['blur_strength'],
            'blur_direction': blur_metadata['blur_direction'],
            'deblur_method': 'DeblurDiff',
            'comparison_type': comparison_type,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'entropy_diff': metrics['entropy_diff'],
            'seed': seed,
            'seed_std_psnr': None,
            'seed_std_ssim': None,
            'seed_std_entropy': None,
            'best_seed': None
        }
        self.rows.append(row)

    def add_deblurdiff_mean_result(self,
                                   original_name: str,
                                   blur_metadata: Dict,
                                   comparison_type: str,
                                   mean_metrics: Dict,
                                   std_metrics: Dict):
        """
        Add row for DeblurDiff mean across seeds.

        Args:
            original_name: Name of original image
            blur_metadata: Dictionary with blur information
            comparison_type: 'deblurred_vs_original' or 'deblurred_vs_blurred'
            mean_metrics: Dictionary with mean psnr, ssim, entropy_diff
            std_metrics: Dictionary with std psnr, ssim, entropy_diff
        """
        row = {
            'original_image': original_name,
            'blur_category': blur_metadata['blur_category'],
            'blur_type': blur_metadata['blur_type'],
            'blur_strength': blur_metadata['blur_strength'],
            'blur_direction': blur_metadata['blur_direction'],
            'deblur_method': 'DeblurDiff',
            'comparison_type': comparison_type,
            'psnr': mean_metrics['psnr'],
            'ssim': mean_metrics['ssim'],
            'entropy_diff': mean_metrics['entropy_diff'],
            'seed': 'mean',
            'seed_std_psnr': std_metrics['psnr'],
            'seed_std_ssim': std_metrics['ssim'],
            'seed_std_entropy': std_metrics['entropy_diff'],
            'best_seed': None
        }
        self.rows.append(row)

    def add_deblurdiff_best_result(self,
                                   original_name: str,
                                   blur_metadata: Dict,
                                   comparison_type: str,
                                   best_seed: int,
                                   metrics: Dict):
        """
        Add row for DeblurDiff best seed result.

        Args:
            original_name: Name of original image
            blur_metadata: Dictionary with blur information
            comparison_type: 'deblurred_vs_original' or 'deblurred_vs_blurred'
            best_seed: Best seed number
            metrics: Dictionary with psnr, ssim, entropy_diff for best seed
        """
        row = {
            'original_image': original_name,
            'blur_category': blur_metadata['blur_category'],
            'blur_type': blur_metadata['blur_type'],
            'blur_strength': blur_metadata['blur_strength'],
            'blur_direction': blur_metadata['blur_direction'],
            'deblur_method': 'DeblurDiff',
            'comparison_type': comparison_type,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'entropy_diff': metrics['entropy_diff'],
            'seed': 'best',
            'seed_std_psnr': None,
            'seed_std_ssim': None,
            'seed_std_entropy': None,
            'best_seed': best_seed
        }
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected rows to pandas DataFrame.

        Returns:
            DataFrame with all evaluation results
        """
        if not self.rows:
            # Return empty DataFrame with correct schema
            column_order = [
                'original_image',
                'blur_category',
                'blur_type',
                'blur_strength',
                'blur_direction',
                'deblur_method',
                'comparison_type',
                'psnr',
                'ssim',
                'entropy_diff',
                'seed',
                'seed_std_psnr',
                'seed_std_ssim',
                'seed_std_entropy',
                'best_seed'
            ]
            return pd.DataFrame(columns=column_order)

        df = pd.DataFrame(self.rows)

        # Ensure column order
        column_order = [
            'original_image',
            'blur_category',
            'blur_type',
            'blur_strength',
            'blur_direction',
            'deblur_method',
            'comparison_type',
            'psnr',
            'ssim',
            'entropy_diff',
            'seed',
            'seed_std_psnr',
            'seed_std_ssim',
            'seed_std_entropy',
            'best_seed'
        ]

        # Reorder columns
        df = df[column_order]

        return df

    def save_csv(self, output_path: str):
        """
        Save results to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        print(f"Total rows: {len(df)}")

    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics from the results.

        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()

        summary = {
            'total_rows': len(df),
            'num_original_images': df['original_image'].nunique(),
            'num_blur_variants': len(df[df['comparison_type'] == 'blurred_vs_original']),
            'blur_categories': df['blur_category'].value_counts().to_dict(),
            'deblur_methods': df['deblur_method'].value_counts().to_dict()
        }

        # Compute average metrics by method and comparison type
        avg_metrics = {}
        for method in ['DeblurGAN', 'DeblurDiff']:
            avg_metrics[method] = {}
            method_df = df[
                (df['deblur_method'] == method) &
                (df['seed'].isin([None, 'mean']) if method == 'DeblurDiff' else df['seed'].isna())
            ]

            for comp_type in ['deblurred_vs_original', 'deblurred_vs_blurred']:
                comp_df = method_df[method_df['comparison_type'] == comp_type]
                if len(comp_df) > 0:
                    avg_metrics[method][comp_type] = {
                        'avg_psnr': float(comp_df['psnr'].mean()),
                        'avg_ssim': float(comp_df['ssim'].mean()),
                        'avg_entropy_diff': float(comp_df['entropy_diff'].mean())
                    }

        summary['average_metrics'] = avg_metrics

        return summary
