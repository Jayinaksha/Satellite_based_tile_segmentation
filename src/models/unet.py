"""Residual U-Net with dilated bottleneck for high-resolution image segmentation."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.blocks import encoder_proc, bottleneck_block, decoder_proc


def get_unet_model(
    input_shape: Tuple[int, int, int], 
    filters: int = 32, 
    n_block: int = 4, 
    n_class: int = 6, 
    mode: str = 'parallel', 
    residual: bool = True
) -> models.Model:
    """Builds a Residual U-Net model with a dilated bottleneck.
    
    This architecture is suited for high-resolution image segmentation tasks.
    
    Args:
        input_shape: The shape of the input image (height, width, channels).
        filters: Base number of filters for the first encoder block.
        n_block: Number of blocks in the encoder and decoder.
        n_class: Number of output classes for segmentation.
        mode: The mode of the bottleneck block ('serial' or 'parallel').
        residual: Whether to use residual connections within blocks.
        
    Returns:
        A compiled tf.keras.models.Model.
    """
    if n_class == 2:
        final_activation = 'sigmoid'
    elif n_class > 2:
        final_activation = 'softmax'
        
    final_activation = 'softmax'
    
    inputs = layers.Input(shape=input_shape)    
    enc, skip = encoder_proc(inputs, filters, n_block, residual=residual)
    center = bottleneck_block(enc, num_filters=filters * (2 ** n_block), mode=mode)
    dec = decoder_proc(center, skip, filters, n_block, residual=residual)
    classify = layers.Conv2D(n_class, (1, 1), activation=final_activation)(dec)
    
    model = models.Model(inputs=inputs, outputs=classify)
    return model
