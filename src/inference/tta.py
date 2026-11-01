"""Test-time augmentation inference."""

import numpy as np
import tensorflow as tf
from typing import Any, Dict
import cv2
from src.inference.predictor import get_extract_pred_scatter

def get_tile_tta_pred(
    tile_path: Any,
    model: tf.keras.Model,
    config: Dict[str, Any],
    from_disk: bool = True
) -> np.ndarray:
    """Get prediction for a tile using Test-Time Augmentation (TTA).
    
    Performs 8-fold TTA (4 rotations + flip + 4 rotations).
    
    Args:
        tile_path: Path to the image tile, or array if from_disk is False.
        model: Trained segmentation model.
        config: Configuration dict containing patch_size, stride, batch_size, num_classes.
        from_disk: Whether to load from disk.
        
    Returns:
        Ensembled label mask.
    """
    if from_disk:
        img = cv2.imread(tile_path)
        if img is None:
            raise ValueError(f"Could not read image from {tile_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = tile_path
        
    flat = {**config.get('training', {}), **config.get('inference', {}), **config.get('dataset', {}), **config}
    patch_size = flat.get('patch_size', 256)
    patch_stride = flat.get('patch_stride', 256)
    batch_size = flat.get('batch_size', 8)
    num_classes = flat.get('num_classes', model.output_shape[-1])
    
    h, w, c = img.shape
    prob_sum = np.zeros((h, w, num_classes), dtype=np.float32)
    
    # 4 rotations, then flip and 4 rotations
    for flip in [False, True]:
        current_img = np.fliplr(img) if flip else img
        
        for k in range(4):
            rot_img = np.rot90(current_img, k=k)
            
            prob = get_extract_pred_scatter(
                img=rot_img,
                model=model,
                patch_size=patch_size,
                patch_stride=patch_stride,
                batch_size=batch_size,
                num_classes=num_classes
            )
            
            # Un-rotate
            prob = np.rot90(prob, k=-k)
            
            # Un-flip
            if flip:
                prob = np.fliplr(prob)
                
            prob_sum += prob
            
    prob_avg = prob_sum / 8.0
    return np.argmax(prob_avg, axis=-1)
