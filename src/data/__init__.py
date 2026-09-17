"""Data handling and pipeline utilities for tile segmentation.

Exports core functions for dataset creation, patch extraction, and data augmentation.
"""

from .dataset import get_dataset, get_image, tf_get_image
from .patch_extraction import make_patches, make_patches_scaled, compute_tile_patch_number
from .augmentation import make_augmentation, no_augmentation

__all__ = [
    'get_dataset',
    'get_image',
    'tf_get_image',
    'make_patches',
    'make_patches_scaled',
    'compute_tile_patch_number',
    'make_augmentation',
    'no_augmentation',
]
