"""Plotting and visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple, Optional, Any, Dict
from sklearn.metrics import confusion_matrix
import seaborn as sns
from src.utils.mask_utils import label2mask

def show_pair(img_list: List[np.ndarray], title: Optional[str] = None, interpolation: Optional[str] = None, **kwargs):
    """Show a pair of images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, img in enumerate(img_list[:2]):
        axes[i].imshow(img, interpolation=interpolation, **kwargs)
        axes[i].axis('off')
    if title:
        plt.suptitle(title)
    plt.show()

def show_grid(img_list: List[np.ndarray], nrow: int = 1, ncol: int = 2, title: Optional[str] = None, interpolation: Optional[str] = None, **kwargs):
    """Show a grid of images."""
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*5, nrow*5))
    axes = axes.flatten() if nrow * ncol > 1 else [axes]
    for ax, img in zip(axes, img_list):
        ax.imshow(img, interpolation=interpolation, **kwargs)
        ax.axis('off')
    if title:
        plt.suptitle(title)
    plt.show()

def plot_training_history(keras_history: Any):
    """Plot training and validation loss and accuracy."""
    history = keras_history.history
    epochs = range(1, len(history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    
    metrics = [m for m in history.keys() if m not in ['loss', 'val_loss']]
    if metrics:
        metric = metrics[0]
        val_metric = f'val_{metric}'
        
        ax2.plot(epochs, history[metric], 'b-', label=f'Training {metric}')
        if val_metric in history:
            ax2.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}')
        ax2.set_title(metric.capitalize())
        ax2.set_xlabel('Epochs')
        ax2.legend()
        
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Plot a confusion matrix using Seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if cm.dtype == float else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def compute_batch_iou(model: tf.keras.Model, patch_pairs: Tuple[np.ndarray, np.ndarray], color_names: List[str], num_classes: int) -> np.ndarray:
    """Compute IoU for a single batch."""
    imgs, masks = patch_pairs
    preds = model.predict(imgs, verbose=0)
    
    y_true = np.argmax(masks, axis=-1) if masks.shape[-1] == num_classes else masks
    y_pred = np.argmax(preds, axis=-1)
    
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=range(num_classes))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
    return iou

def compute_dataset_cm(model: tf.keras.Model, ds: tf.data.Dataset, steps: int, num_classes: int, normalize: bool = False) -> np.ndarray:
    """Compute confusion matrix for an entire dataset."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i, (imgs, masks) in enumerate(ds.take(steps)):
        preds = model.predict(imgs, verbose=0)
        y_true = np.argmax(masks, axis=-1) if masks.shape[-1] == num_classes else masks
        y_pred = np.argmax(preds, axis=-1)
        
        batch_cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=range(num_classes))
        cm += batch_cm
        
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
    return cm

def compute_dataset_iou(model: tf.keras.Model, ds: tf.data.Dataset, steps: int, color_names: List[str], num_classes: int) -> np.ndarray:
    """Compute IoU over an entire dataset."""
    cm = compute_dataset_cm(model, ds, steps, num_classes, normalize=False)
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
    return iou

def get_tile_cm(y_true: np.ndarray, y_pred: np.ndarray, color_list: List[Tuple[int, int, int]], num_classes: int, normalize: bool = False) -> np.ndarray:
    """Compute confusion matrix for a single tile."""
    # Assuming y_true and y_pred are sparse labels
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=range(num_classes))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    return cm

def get_tile_iou(y_true: np.ndarray, y_pred: np.ndarray, color_list: List[Tuple[int, int, int]], color_names: List[str]) -> np.ndarray:
    """Compute IoU for a single tile."""
    num_classes = len(color_list)
    cm = get_tile_cm(y_true, y_pred, color_list, num_classes, normalize=False)
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
    return iou
