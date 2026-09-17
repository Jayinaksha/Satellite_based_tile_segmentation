"""Custom loss functions for image segmentation.

This module provides Jaccard distance and focal losses (binary and categorical)
useful for handling class imbalance in segmentation tasks.
"""
from typing import Callable

import tensorflow as tf
from tensorflow.keras import backend as K


def jaccard_distance(num_classes: int = 6, smooth: float = 1.0) -> Callable:
    """Computes the Jaccard distance loss for segmentation.
    
    Jaccard distance is defined as: 1 - (Intersection over Union).
    Useful for addressing severe class imbalance.
    
    Args:
        num_classes: Number of classes in the segmentation task.
        smooth: Smoothing factor to prevent division by zero.
        
    Returns:
        A loss function that takes (y_true, y_pred) as arguments.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Inner loss function.
        
        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.
            
        Returns:
            Jaccard distance loss scalar tensor.
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    return loss


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25) -> Callable:
    """Computes the Binary Focal Loss.
    
    Focal loss addresses class imbalance by down-weighting well-classified
    examples and focusing on hard examples.
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).
    
    Args:
        gamma: Focusing parameter. Higher values reduce the loss for well-classified examples.
        alpha: Balancing parameter to weight the positive class.
        
    Returns:
        A loss function that takes (y_true, y_pred) as arguments.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Inner loss function.
        
        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted probabilities tensor.
            
        Returns:
            Binary focal loss scalar tensor.
        """
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow((1 - p_t), gamma)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        focal_loss = alpha_t * focal_weight * bce
        return K.mean(focal_loss, axis=-1)
    return loss


def categorical_focal_loss(num_classes: int = 6, gamma: float = 2.0, alpha: float = 0.25) -> Callable:
    """Computes the Categorical Focal Loss.
    
    Focal loss addresses class imbalance by down-weighting well-classified
    examples and focusing on hard examples.
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).
    
    Args:
        num_classes: Number of classes in the segmentation task.
        gamma: Focusing parameter.
        alpha: Balancing parameter.
        
    Returns:
        A loss function that takes (y_true, y_pred) as arguments.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Inner loss function.
        
        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted probabilities tensor.
            
        Returns:
            Categorical focal loss scalar tensor.
        """
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true * K.log(y_pred)
        
        focal_weight = K.pow(1 - y_pred, gamma)
        
        focal_loss = alpha * focal_weight * ce
        
        return K.sum(focal_loss, axis=-1)
    return loss
