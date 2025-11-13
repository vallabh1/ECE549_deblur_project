"""
Model interfaces for different deblurring architectures.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np


class DeblurModel(ABC):
    """Base class for deblurring models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def load_model(self, weights_path: str):
        """Load model with pretrained weights."""
        pass

    @abstractmethod
    def predict(self, blurred_image: np.ndarray) -> np.ndarray:
        """
        Deblur a single image.

        Args:
            blurred_image: RGB image (H, W, C) in range [0, 255]

        Returns:
            Deblurred RGB image (H, W, C) in range [0, 255]
        """
        pass


class DeblurGANv2(DeblurModel):
    """Interface for DeblurGANv2 model."""

    def __init__(self):
        super().__init__("DeblurGANv2")
        self.predictor = None

    def load_model(self, weights_path: str):
        """Load DeblurGANv2 model."""
        # Convert to absolute path before changing directories
        import os
        weights_path = os.path.abspath(weights_path)

        # Add DeblurGANv2 to path
        deblurgan_path = Path(__file__).parent.parent / "models" / "DeblurGANv2"
        sys.path.insert(0, str(deblurgan_path))

        from predict import Predictor

        # Change to DeblurGANv2 directory for config.yaml
        original_dir = os.getcwd()
        try:
            os.chdir(deblurgan_path)
            self.predictor = Predictor(weights_path=weights_path, model_name='')
        finally:
            os.chdir(original_dir)

    def predict(self, blurred_image: np.ndarray) -> np.ndarray:
        """Deblur image using DeblurGANv2."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure uint8 [0, 255]
        if blurred_image.dtype != np.uint8:
            if blurred_image.max() <= 1.0:
                blurred_image = (blurred_image * 255).astype(np.uint8)
            else:
                blurred_image = blurred_image.astype(np.uint8)

        return self.predictor(blurred_image, mask=None, ignore_mask=True)


# Add more model interfaces here as needed
# class NAFNet(DeblurModel):
#     ...
