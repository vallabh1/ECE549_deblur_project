"""
Image matching and pairing system for deblurring evaluation.
Maps blurred and deblurred images back to their original sources.
"""

from pathlib import Path
from typing import Dict, Optional, List
import re


def parse_blur_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse blur filename to extract metadata.

    Examples:
        'bison_linear_5_45.png' -> {base_name: 'bison', blur_category: 'linear',
                                     blur_type: 'linear5_45', blur_strength: '5',
                                     blur_direction: '45'}
        'cubancroc_shake_20_10.png' -> {base_name: 'cubancroc', blur_category: 'shake',
                                         blur_type: 'shake20', blur_strength: '20',
                                         blur_direction: None}
        'squirrel4_ang_10_1.png' -> {base_name: 'squirrel4', blur_category: 'angular',
                                      blur_type: 'angular1', blur_strength: '10',
                                      blur_direction: None}

    Args:
        filename: Blurred image filename (with or without extension)

    Returns:
        Dictionary with keys: base_name, blur_category, blur_type,
                             blur_strength, blur_direction
    """
    # Remove extension
    name = Path(filename).stem

    # Try different blur patterns

    # Pattern 1: linear blur - name_linear_kernel_direction
    linear_match = re.match(r'^(.+?)_linear_(\d+)_(\d+)$', name)
    if linear_match:
        base_name = linear_match.group(1)
        kernel = linear_match.group(2)
        direction = linear_match.group(3)
        return {
            'base_name': base_name,
            'blur_category': 'linear',
            'blur_type': f'linear{kernel}_{direction}',
            'blur_strength': kernel,
            'blur_direction': direction
        }

    # Pattern 2: shake blur - name_shake_kernel_number
    shake_match = re.match(r'^(.+?)_shake_(\d+)_\d+$', name)
    if shake_match:
        base_name = shake_match.group(1)
        kernel = shake_match.group(2)
        return {
            'base_name': base_name,
            'blur_category': 'shake',
            'blur_type': f'shake{kernel}',
            'blur_strength': kernel,
            'blur_direction': None
        }

    # Pattern 3: angular blur - name_ang_kernel_number
    angular_match = re.match(r'^(.+?)_ang_(\d+)_(\d+)$', name)
    if angular_match:
        base_name = angular_match.group(1)
        kernel = angular_match.group(2)
        variant = angular_match.group(3)
        return {
            'base_name': base_name,
            'blur_category': 'angular',
            'blur_type': f'angular{variant}',
            'blur_strength': kernel,
            'blur_direction': None
        }

    # If no pattern matches, return None values
    return {
        'base_name': name,
        'blur_category': None,
        'blur_type': None,
        'blur_strength': None,
        'blur_direction': None
    }


def find_original_image(base_name: str, original_dir: Path) -> Optional[Path]:
    """
    Case-insensitive search for original image.
    Handles variations like .jpg/.JPG and typos like squrriel.

    Args:
        base_name: Base name of the image (e.g., 'bison', 'squirrel2')
        original_dir: Directory containing original images

    Returns:
        Path to original image if found, None otherwise
    """
    original_dir = Path(original_dir)

    # Common extensions to try
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    # Try exact match first
    for ext in extensions:
        candidate = original_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate

    # Try case-insensitive search
    for img_path in original_dir.iterdir():
        if img_path.is_file():
            if img_path.stem.lower() == base_name.lower():
                return img_path

    # Handle common typos (e.g., squrriel -> squirrel)
    if base_name.lower().startswith('squrriel'):
        # Try replacing 'squrriel' with 'squirrel'
        corrected = base_name.replace('squrriel', 'squirrel').replace('Squrriel', 'Squirrel')
        for ext in extensions:
            candidate = original_dir / f"{corrected}{ext}"
            if candidate.exists():
                return candidate

    return None


def build_image_mapping(original_dir: Path,
                       blurred_dir: Path,
                       deblurgan_dir: Optional[Path] = None,
                       deblurdiff_dir: Optional[Path] = None) -> Dict:
    """
    Build complete mapping structure for all image variants.

    Structure:
    {
        'original_name': {
            'blur_variant_key': {
                'original_path': Path,
                'blurred_path': Path,
                'blur_metadata': {blur_category, blur_type, blur_strength, blur_direction},
                'deblurgan_path': Path or None,
                'deblurdiff_paths': {seed: Path} or None
            }
        }
    }

    Args:
        original_dir: Directory with original images
        blurred_dir: Directory with blurred images (contains subdirectories)
        deblurgan_dir: Directory with DeblurGAN results (optional)
        deblurdiff_dir: Directory with DeblurDiff results (optional)

    Returns:
        Nested dictionary mapping
    """
    original_dir = Path(original_dir)
    blurred_dir = Path(blurred_dir)
    if deblurgan_dir:
        deblurgan_dir = Path(deblurgan_dir)
    if deblurdiff_dir:
        deblurdiff_dir = Path(deblurdiff_dir)

    mapping = {}

    # Iterate through all blurred images
    for blur_subdir in sorted(blurred_dir.iterdir()):
        if not blur_subdir.is_dir():
            continue

        for blurred_path in sorted(blur_subdir.glob('*.png')):
            # Parse filename
            metadata = parse_blur_filename(blurred_path.name)
            base_name = metadata['base_name']
            blur_type = metadata['blur_type']

            if not blur_type:
                print(f"Warning: Could not parse blur filename: {blurred_path.name}")
                continue

            # Find original image
            original_path = find_original_image(base_name, original_dir)
            if not original_path:
                print(f"Warning: Could not find original image for: {base_name}")
                continue

            # Initialize nested structure
            if base_name not in mapping:
                mapping[base_name] = {}

            # Create blur variant key
            variant_key = blurred_path.stem  # Use full filename as key

            # Store mapping
            mapping[base_name][variant_key] = {
                'original_path': original_path,
                'blurred_path': blurred_path,
                'blur_metadata': metadata,
                'deblurgan_path': None,
                'deblurdiff_paths': {}
            }

            # Find DeblurGAN result
            if deblurgan_dir:
                deblurgan_path = deblurgan_dir / blurred_path.name
                if deblurgan_path.exists():
                    mapping[base_name][variant_key]['deblurgan_path'] = deblurgan_path

            # Find DeblurDiff results (4 seeds)
            if deblurdiff_dir:
                for seed in [11, 42, 127, 231]:
                    seed_dir = deblurdiff_dir / f'results_seed{seed}'
                    deblurdiff_path = seed_dir / blurred_path.name
                    if deblurdiff_path.exists():
                        mapping[base_name][variant_key]['deblurdiff_paths'][seed] = deblurdiff_path

    return mapping


def get_all_blur_variants(mapping: Dict) -> List[tuple]:
    """
    Get flat list of all blur variants for iteration.

    Args:
        mapping: Output from build_image_mapping()

    Returns:
        List of tuples: (base_name, variant_key, variant_data)
    """
    variants = []
    for base_name, variants_dict in mapping.items():
        for variant_key, variant_data in variants_dict.items():
            variants.append((base_name, variant_key, variant_data))
    return variants


def print_mapping_summary(mapping: Dict):
    """Print summary statistics of the image mapping."""
    total_variants = sum(len(variants) for variants in mapping.values())
    num_originals = len(mapping)

    # Count available deblurred results
    deblurgan_count = 0
    deblurdiff_count = 0

    for base_name, variants in mapping.items():
        for variant_key, data in variants.items():
            if data['deblurgan_path']:
                deblurgan_count += 1
            if data['deblurdiff_paths']:
                deblurdiff_count += 1

    print(f"\n=== Image Mapping Summary ===")
    print(f"Original images: {num_originals}")
    print(f"Total blur variants: {total_variants}")
    print(f"DeblurGAN results found: {deblurgan_count}/{total_variants}")
    print(f"DeblurDiff results found: {deblurdiff_count}/{total_variants}")

    # Count by blur category
    blur_categories = {}
    for base_name, variants in mapping.items():
        for variant_key, data in variants.items():
            category = data['blur_metadata']['blur_category']
            blur_categories[category] = blur_categories.get(category, 0) + 1

    print(f"\nBlur categories:")
    for category, count in sorted(blur_categories.items()):
        print(f"  {category}: {count}")
    print("=" * 30 + "\n")
