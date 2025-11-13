"""
Image quality metrics for deblurring evaluation.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """Calculate PSNR between two images. Higher is better."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}")

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * np.log10(max_value / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """Calculate SSIM between two images. Higher is better."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}")

    try:
        from skimage.metrics import structural_similarity
        if len(img1.shape) == 3:
            return structural_similarity(img1, img2, data_range=max_value, channel_axis=2)
        else:
            return structural_similarity(img1, img2, data_range=max_value)
    except ImportError:
        print("Warning: skimage not available, using basic SSIM approximation")
        return _basic_ssim(img1, img2, max_value)


def _basic_ssim(img1: np.ndarray, img2: np.ndarray, max_value: float) -> float:
    """Basic SSIM fallback."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    return numerator / denominator


def entropy_of_difference(predicted: np.ndarray, initial: np.ndarray) -> float:
    """
    Compute entropy of difference between predicted and initial images.
    Higher entropy = more information recovered.
    """
    if predicted.shape != initial.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs initial {initial.shape}")

    diff = predicted.astype(np.float32) - initial.astype(np.float32)
    hist, _ = np.histogram(diff.flatten(), bins=256, range=(-255, 255))
    hist = hist + 1e-10
    probabilities = hist / hist.sum()

    return scipy_entropy(probabilities, base=2)
