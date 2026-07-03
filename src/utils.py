"""Utility functions for data preprocessing and validation."""

import os
from typing import Tuple, Optional
import cv2
import numpy as np


def validate_image(image_path: str) -> bool:
    """Validate that an image file can be read and is not corrupted."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        if img.size == 0:
            return False
        return True
    except Exception as e:
        print(f"Error validating image {image_path}: {e}")
        return False


def validate_dataset_structure(data_dir: str) -> Tuple[bool, str]:
    """Validate dataset directory structure."""
    if not os.path.exists(data_dir):
        return False, f"Dataset directory not found: {data_dir}"
    
    has_mask_folder = os.path.exists(os.path.join(data_dir, "mask"))
    has_no_mask_folder = os.path.exists(os.path.join(data_dir, "without_mask"))
    
    if not has_mask_folder and not has_no_mask_folder:
        return False, "Expected 'mask' and 'without_mask' subdirectories not found"
    
    return True, "Dataset structure is valid"


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to range [0, 1]."""
    return image.astype("float32") / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from range [0, 1] to [0, 255]."""
    return (image * 255).astype("uint8")


def get_image_stats(image_path: str) -> Optional[dict]:
    """Get statistics about an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "mean": img.mean(),
            "std": img.std(),
            "min": img.min(),
            "max": img.max(),
        }
    except Exception as e:
        print(f"Error getting image stats for {image_path}: {e}")
        return None
