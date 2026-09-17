"""Trainer class for abstracting training logic."""

from typing import Any, Dict, List, Optional
import tensorflow as tf
from src.training.lr_finder import LearningRateFinder

class Trainer:
    """Handles model compilation, training and learning rate finding."""
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any]):
        """Initialize the Trainer.
        
        Args:
            model: Keras model to train.
            config: Configuration dictionary containing optimizer, loss, metrics, etc.
        """
        self.model = model
        self.config = config
        
    def compile(self):
        """Compile the model using configuration."""
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'categorical_crossentropy')
        metrics = self.config.get('metrics', ['accuracy'])
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def train(self, train_ds: tf.data.Dataset, val_ds: Optional[tf.data.Dataset] = None, epochs: int = 10):
        """Train the model.
        
        Args:
            train_ds: Training dataset.
            val_ds: Validation dataset.
            epochs: Number of epochs to train.
            
        Returns:
            Training history.
        """
        callbacks = self.config.get('callbacks', [])
        steps_per_epoch = self.config.get('steps_per_epoch', None)
        validation_steps = self.config.get('validation_steps', None)
        
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
    def find_lr(self, train_ds: tf.data.Dataset, start_lr: float = 1e-7, end_lr: float = 10, epochs: int = 1):
        """Find the optimal learning rate.
        
        Args:
            train_ds: Training dataset.
            start_lr: Starting learning rate.
            end_lr: Maximum learning rate to try.
            epochs: Number of epochs for the search.
        """
        steps_per_epoch = self.config.get('steps_per_epoch', 100)
        lr_finder = LearningRateFinder(self.model)
        lr_finder.find(
            trainData=train_ds,
            startLR=start_lr,
            endLR=end_lr,
            epochs=epochs,
            stepsPerEpoch=steps_per_epoch
        )
        lr_finder.plot_loss()
