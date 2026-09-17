"""One Cycle Learning Rate Scheduler.

Reference:
    Smith, L. N. (2017). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.
    https://arxiv.org/abs/1708.07120
"""

import math
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from typing import Any, Callable


class ParamScheduler:
    """Base class for parameter schedulers."""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
        
    def __call__(self, val: float) -> float:
        raise NotImplementedError


class LinearScheduler(ParamScheduler):
    """Linear scheduler."""
    def __call__(self, val: float) -> float:
        return self.start + (self.end - self.start) * val


class CosineScheduler(ParamScheduler):
    """Cosine scheduler."""
    def __call__(self, val: float) -> float:
        return self.end + (self.start - self.end) / 2.0 * (math.cos(math.pi * val) + 1)


class OneCycleScheduler(Callback):
    """Callback that schedules learning rate and momentum according to the 1cycle policy."""
    
    def __init__(self, lr_max: float, steps: int, mom_min: float = 0.85, mom_max: float = 0.95,
                 phase_1_pct: float = 0.3, div_factor: float = 25.0, final_div_factor: float = 1e4):
        """Initialize OneCycleScheduler.
        
        Args:
            lr_max: Maximum learning rate.
            steps: Total number of training steps.
            mom_min: Minimum momentum.
            mom_max: Maximum momentum.
            phase_1_pct: Percentage of steps in phase 1 (increasing LR).
            div_factor: Divisor for determining initial learning rate (lr_max / div_factor).
            final_div_factor: Divisor for determining final learning rate (initial_lr / final_div_factor).
        """
        super().__init__()
        self.lr_max = lr_max
        self.steps = steps
        self.mom_min = mom_min
        self.mom_max = mom_max
        self.phase_1_pct = phase_1_pct
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.phase_1_steps = int(steps * phase_1_pct)
        self.phase_2_steps = steps - self.phase_1_steps
        
        self.step_num = 0
        
        self.lr_phase_1 = CosineScheduler(lr_max / div_factor, lr_max)
        self.lr_phase_2 = CosineScheduler(lr_max, lr_max / (div_factor * final_div_factor))
        
        self.mom_phase_1 = CosineScheduler(mom_max, mom_min)
        self.mom_phase_2 = CosineScheduler(mom_min, mom_max)
        
    def on_batch_begin(self, batch: int, logs: dict = None):
        """Updates learning rate and momentum at the beginning of each batch."""
        if self.step_num < self.phase_1_steps:
            pct = self.step_num / self.phase_1_steps
            lr = self.lr_phase_1(pct)
            mom = self.mom_phase_1(pct)
        else:
            pct = (self.step_num - self.phase_1_steps) / max(1, self.phase_2_steps)
            lr = self.lr_phase_2(pct)
            mom = self.mom_phase_2(pct)
            
        K.set_value(self.model.optimizer.lr, lr)
        if hasattr(self.model.optimizer, 'momentum'):
            K.set_value(self.model.optimizer.momentum, mom)
        elif hasattr(self.model.optimizer, 'beta_1'):
            K.set_value(self.model.optimizer.beta_1, mom)
            
        self.step_num += 1
