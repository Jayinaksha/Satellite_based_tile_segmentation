"""Building blocks for the U-Net model."""
from typing import Tuple, List, Union

import tensorflow as tf
from tensorflow.keras import layers


def conv_block(input_tensor: tf.Tensor, num_filters: int, dilation: int = 1, residual: bool = False) -> tf.Tensor:
    """Creates a convolutional block with two Conv2D and BatchNormalization layers.
    
    Args:
        input_tensor: Input tensor.
        num_filters: Number of filters for the convolution layers.
        dilation: Dilation rate for the convolution layers.
        residual: If True, adds a skip connection concatenating input and output.
        
    Returns:
        Output tensor of the convolutional block.
    """
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', 
                      dilation_rate=dilation, kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', 
                      dilation_rate=dilation, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    if residual:
        x = layers.Concatenate()([x, input_tensor])
        
    return x


def bottleneck_block(input_tensor: tf.Tensor, num_filters: int, mode: str = 'parallel') -> tf.Tensor:
    """Creates a bottleneck block, typically for the center of a U-Net.
    
    Args:
        input_tensor: Input tensor.
        num_filters: Number of base filters.
        mode: Mode of dilated convolutions ('serial', 'parallel', or other).
        
    Returns:
        Output tensor of the bottleneck block.
    """
    if mode == 'serial':
        dilated1 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', 
                                 dilation_rate=1, kernel_initializer='he_normal')(input_tensor)
        dilated1 = layers.BatchNormalization()(dilated1)
        dilated2 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', 
                                 dilation_rate=2, kernel_initializer='he_normal')(dilated1)
        dilated2 = layers.BatchNormalization()(dilated2)
        dilated4 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', 
                                 dilation_rate=4, kernel_initializer='he_normal')(dilated2)
        dilated4 = layers.BatchNormalization()(dilated4)
        dilated8 = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', 
                                 dilation_rate=8, kernel_initializer='he_normal')(dilated4)
        dilated8 = layers.BatchNormalization()(dilated8)
        x = layers.Concatenate()([dilated1, dilated2, dilated4, dilated8])
    elif mode == 'parallel':
        dilated1 = conv_block(input_tensor, num_filters, dilation=1)
        dilated2 = conv_block(input_tensor, num_filters, dilation=2)
        dilated4 = conv_block(input_tensor, num_filters, dilation=4)
        dilated8 = conv_block(input_tensor, num_filters, dilation=8)
        x = layers.Concatenate()([dilated1, dilated2, dilated4, dilated8])
    else:
        x = conv_block(input_tensor, num_filters, dilation=1, residual=True)
        
    return x


def encoder_block(input_tensor: tf.Tensor, num_filters: int, residual: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Creates an encoder block with a convolutional block and max pooling.
    
    Args:
        input_tensor: Input tensor.
        num_filters: Number of filters for the convolution block.
        residual: If True, uses residual connections in the conv block.
        
    Returns:
        A tuple of (pooled_output, unpooled_encoder_output).
    """
    encoder = conv_block(input_tensor, num_filters, residual=residual)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor: tf.Tensor, concat_tensor: tf.Tensor, num_filters: int, residual: bool = True) -> tf.Tensor:
    """Creates a decoder block with transposed convolution and standard convolutions.
    
    Args:
        input_tensor: Input tensor from the previous layer.
        concat_tensor: Skip connection tensor from the corresponding encoder block.
        num_filters: Number of filters for the convolution layers.
        residual: If True, adds a skip connection concatenating input and output.
        
    Returns:
        Output tensor of the decoder block.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), 
                               padding='same', kernel_initializer='he_normal')(input_tensor)
    # the function uses layers.concatenate in the original, adapting to layers.Concatenate
    x = layers.Concatenate(axis=-1)([concat_tensor, x])
    shortcut = x
    
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    if residual:
        x = layers.Concatenate()([x, shortcut])
        
    return x


def encoder_proc(x: tf.Tensor, filters: int = 32, n_block: int = 4, residual: bool = True) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Creates the entire encoder pathway.
    
    Args:
        x: Input tensor.
        filters: Base number of filters.
        n_block: Number of encoder blocks.
        residual: If True, uses residual connections in the blocks.
        
    Returns:
        A tuple containing the final encoder output and a list of skip connections.
    """
    skip = []
    for i in range(n_block):
        x, x_skip = encoder_block(x, filters * (2 ** i), residual=residual)
        skip.append(x_skip)
    return x, skip


def decoder_proc(x: tf.Tensor, skip: List[tf.Tensor], filters: int = 32, n_block: int = 4, residual: bool = True) -> tf.Tensor:
    """Creates the entire decoder pathway.
    
    Args:
        x: Input tensor (output from the bottleneck).
        skip: List of skip connection tensors from the encoder.
        filters: Base number of filters.
        n_block: Number of decoder blocks.
        residual: If True, uses residual connections in the blocks.
        
    Returns:
        Output tensor of the final decoder block.
    """
    for i in reversed(range(n_block)):
        x = decoder_block(x, skip[i], filters * (2 ** i), residual=residual)
    return x
