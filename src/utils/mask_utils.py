"""Mask and label conversion utilities."""

import numpy as np
from typing import List, Tuple

def mask2label(mask: np.ndarray, colors: List[Tuple[int, int, int]], sparse: bool = False) -> np.ndarray:
    """Convert an RGB mask to a label array.
    
    Args:
        mask: RGB mask array of shape (H, W, 3).
        colors: List of color tuples for each class.
        sparse: If True, returns (H, W). If False, returns one-hot (H, W, num_classes).
        
    Returns:
        Label array.
    """
    h, w, _ = mask.shape
    num_classes = len(colors)
    
    if sparse:
        label = np.zeros((h, w), dtype=np.int32)
        for idx, color in enumerate(colors):
            matches = np.all(mask == color, axis=-1)
            label[matches] = idx
        return label
    else:
        label = np.zeros((h, w, num_classes), dtype=np.float32)
        for idx, color in enumerate(colors):
            matches = np.all(mask == color, axis=-1)
            label[matches, idx] = 1.0
        return label

def label2mask(sparse_label: np.ndarray, colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Convert a sparse label array to an RGB mask.
    
    Args:
        sparse_label: Integer label array of shape (H, W).
        colors: List of color tuples corresponding to each class.
        
    Returns:
        RGB mask array of shape (H, W, 3).
    """
    h, w = sparse_label.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx, color in enumerate(colors):
        mask[sparse_label == idx] = color
        
    return mask
