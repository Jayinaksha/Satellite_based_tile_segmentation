"""Custom metrics for model evaluation."""

import tensorflow as tf

class CategoricalMeanIoU(tf.keras.metrics.MeanIoU):
    """Categorical Mean Intersection over Union."""
    
    def __init__(self, name: str = 'mean_iou', **kwargs):
        """Initialize CategoricalMeanIoU.
        
        Args:
            name: Name of the metric.
            **kwargs: Additional arguments passed to tf.keras.metrics.MeanIoU.
        """
        super().__init__(name=name, **kwargs)
        
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None):
        """Update the state of the metric.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional weights for each sample.
            
        Returns:
            Result of the update state operation.
        """
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        if tf.rank(y_true) == tf.rank(y_pred):
            y_true = tf.argmax(y_true, axis=-1)
            
        return super().update_state(y_true=y_true, y_pred=y_pred_classes, sample_weight=sample_weight)
