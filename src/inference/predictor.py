"""Tile prediction logic."""

import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Tuple
from src.utils.mask_utils import label2mask
import cv2

def get_extract_pred_scatter(
    img: np.ndarray,
    model: tf.keras.Model,
    patch_size: int,
    patch_stride: int,
    batch_size: int,
    num_classes: int
) -> np.ndarray:
    """Predict on full image by scattering patch predictions.
    
    Args:
        img: Input image array.
        model: Trained segmentation model.
        patch_size: Size of the patches to extract.
        patch_stride: Stride for patch extraction.
        batch_size: Batch size for model prediction.
        num_classes: Number of segmentation classes.
        
    Returns:
        Probability map of shape (H, W, num_classes).
    """
    h, w, c = img.shape
    patches = []
    coords = []
    
    for i in range(0, max(1, h - patch_size + 1), patch_stride):
        for j in range(0, max(1, w - patch_size + 1), patch_stride):
            patches.append(img[i:i+patch_size, j:j+patch_size, :])
            coords.append([i, j])
            
    if not patches:
        return np.zeros((h, w, num_classes), dtype=np.float32)
        
    patches = np.array(patches)
    coords = np.array(coords)
    
    # Predict in batches
    preds = model.predict(patches, batch_size=batch_size, verbose=0)
    
    # Scatter ND to reconstruct full image
    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    counts = np.zeros((h, w, 1), dtype=np.float32)
    
    for coord, pred in zip(coords, preds):
        i, j = coord
        prob_map[i:i+patch_size, j:j+patch_size, :] += pred
        counts[i:i+patch_size, j:j+patch_size, :] += 1
        
    prob_map = prob_map / (counts + 1e-8)
    return prob_map

def get_tile_prediction(
    tile_path: Any,
    model: tf.keras.Model,
    config: Dict[str, Any],
    from_disk: bool = True
) -> np.ndarray:
    """Get prediction for a single tile.
    
    Args:
        tile_path: Path to the image tile, or image array if from_disk is False.
        model: Trained segmentation model.
        config: Configuration dictionary (patch_size, stride, etc.).
        from_disk: If True, load image from disk.
        
    Returns:
        Segmentation mask.
    """
    if from_disk:
        img = cv2.imread(tile_path)
        if img is None:
            raise ValueError(f"Could not read image from {tile_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = tile_path
        
    patch_size = config.get('patch_size', 256)
    patch_stride = config.get('patch_stride', 256)
    batch_size = config.get('batch_size', 8)
    num_classes = config.get('num_classes', 5)
    
    prob_map = get_extract_pred_scatter(
        img=img,
        model=model,
        patch_size=patch_size,
        patch_stride=patch_stride,
        batch_size=batch_size,
        num_classes=num_classes
    )
    
    label_map = np.argmax(prob_map, axis=-1)
    return label_map
