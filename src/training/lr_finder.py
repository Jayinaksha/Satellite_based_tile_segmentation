"""Learning Rate Finder for Keras models.

Reference:
    Smith, L. N. (2015). Cyclical Learning Rates for Training Neural Networks.
    https://arxiv.org/abs/1506.01186
"""

import tempfile
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
from typing import Optional, Any


class LearningRateFinder:
    """Finds the optimal learning rate for training a model."""
    
    def __init__(self, model: Any, stopFactor: float = 10.0, beta: float = 0.98):
        """Initialize the LearningRateFinder.
        
        Args:
            model: The Keras model to train.
            stopFactor: Factor to multiply best loss to determine stop condition.
            beta: Beta parameter for moving average of loss.
        """
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        self.lrs = []
        self.losses = []
        self.lrMult = 1.0
        self.avgLoss = 0.0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        """Reset internal states."""
        self.lrs = []
        self.losses = []
        self.lrMult = 1.0
        self.avgLoss = 0.0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def on_batch_end(self, batch: int, logs: dict):
        """Callback for end of batch to update learning rate and track loss.
        
        Args:
            batch: Batch index.
            logs: Dictionary of logs.
        """
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        
        l = logs.get('loss', 0)
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)
        
        stopLoss = self.stopFactor * self.bestLoss
        
        if self.batchNum > 1 and smooth > stopLoss:
            self.model.stop_training = True
            return
            
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth
            
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(
        self,
        trainData: Any,
        startLR: float,
        endLR: float,
        epochs: int = 1,
        stepsPerEpoch: Optional[int] = None,
        verbose: int = 1
    ):
        """Run the learning rate finder.
        
        Args:
            trainData: Training dataset.
            startLR: Initial learning rate.
            endLR: Final learning rate.
            epochs: Number of epochs to train.
            stepsPerEpoch: Number of steps per epoch.
            verbose: Verbosity level.
        """
        self.reset()
        
        numBatchUpdates = epochs * (stepsPerEpoch if stepsPerEpoch else 1)
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
        
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)
        
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        self.model.fit(
            trainData,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[callback]
        )
        
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)

    def plot_loss(self, skipBegin: int = 2, skipEnd: int = 1, title: str = ''):
        """Plot the learning rate vs loss.
        
        Args:
            skipBegin: Number of initial batches to skip for plotting.
            skipEnd: Number of final batches to skip for plotting.
            title: Optional title for the plot.
        """
        lrs = self.lrs[skipBegin:-skipEnd] if skipEnd > 0 else self.lrs[skipBegin:]
        losses = self.losses[skipBegin:-skipEnd] if skipEnd > 0 else self.losses[skipBegin:]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15, subs=range(1, 10)))
        ax.set_xlabel('Learning Rate (Log Scale)')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        if title:
            plt.title(title)
        
        plt.show()
