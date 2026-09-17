"""Inference modules for tile segmentation."""

from src.inference.predictor import get_extract_pred_scatter, get_tile_prediction
from src.inference.tta import get_tile_tta_pred

__all__ = [
    'get_extract_pred_scatter',
    'get_tile_prediction',
    'get_tile_tta_pred'
]
