"""Patch extraction module for high-resolution aerial tiles.

Provides functionalities to extract uniform or multi-scale patches from
high-resolution aerial images and corresponding masks for segmentation tasks.
"""

from typing import List, Tuple
import tensorflow as tf
import imageio.v2 as imageio


def make_patches(
    aerial: tf.Tensor,
    mask: tf.Tensor,
    patch_size: int = 256,
    patch_stride: int = 256
) -> tf.Tensor:
    """Extracts patches from an aerial image and its mask.

    Args:
        aerial (tf.Tensor): The high-resolution aerial image.
        mask (tf.Tensor): The corresponding mask image.
        patch_size (int, optional): Size of the square patch. Defaults to 256.
        patch_stride (int, optional): Stride for patch extraction. Defaults to 256.

    Returns:
        tf.Tensor: A tensor containing paired image and mask patches.
    """
    with tf.device('/CPU:0'):
        SIZES = [1, patch_size, patch_size, 1]
        STRIDES = [1, patch_stride, patch_stride, 1]
        
        # Expand dims if necessary or pass directly if it's already a batch of 1
        patches = tf.image.extract_patches(
            images=[aerial, mask],
            sizes=SIZES,
            strides=STRIDES,
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        patch_nb = tf.shape(patches)[1] * tf.shape(patches)[2]
        patches = tf.reshape(patches, [2, patch_nb, patch_size, patch_size, 3])
        patches = tf.transpose(patches, perm=[1, 0, 2, 3, 4])
        
        return patches


def make_patches_scaled(
    aerial: tf.Tensor,
    mask: tf.Tensor,
    patch_size: int = 256,
    patch_stride: int = 256
) -> tf.Tensor:
    """Extracts multi-scale patches from an aerial image and its mask.

    Args:
        aerial (tf.Tensor): The high-resolution aerial image.
        mask (tf.Tensor): The corresponding mask image.
        patch_size (int, optional): Size of the square patch. Defaults to 256.
        patch_stride (int, optional): Stride for patch extraction. Defaults to 256.

    Returns:
        tf.Tensor: A tensor containing paired image and mask multi-scale patches.
    """
    with tf.device('/CPU:0'):
        SIZES = [1, patch_size, patch_size, 1]
        STRIDES = [1, patch_stride, patch_stride, 1]
        
        patches_rate1 = tf.image.extract_patches(
            images=[aerial, mask], sizes=SIZES, strides=STRIDES, rates=[1, 1, 1, 1], padding='VALID'
        )
        patches_rate2 = tf.image.extract_patches(
            images=[aerial, mask], sizes=SIZES, strides=STRIDES, rates=[1, 2, 2, 1], padding='VALID'
        )
        patches_rate3 = tf.image.extract_patches(
            images=[aerial, mask], sizes=SIZES, strides=STRIDES, rates=[1, 3, 3, 1], padding='VALID'
        )
        
        patch_number_rate1 = tf.shape(patches_rate1)[1] * tf.shape(patches_rate1)[2]
        patch_number_rate2 = tf.shape(patches_rate2)[1] * tf.shape(patches_rate2)[2]
        patch_number_rate3 = tf.shape(patches_rate3)[1] * tf.shape(patches_rate3)[2]
        
        patches_rate1 = tf.reshape(patches_rate1, [2, patch_number_rate1, patch_size, patch_size, 3])
        patches_rate2 = tf.reshape(patches_rate2, [2, patch_number_rate2, patch_size, patch_size, 3])
        patches_rate3 = tf.reshape(patches_rate3, [2, patch_number_rate3, patch_size, patch_size, 3])
        
        patches = tf.concat([patches_rate1, patches_rate2, patches_rate3], axis=1)
        patches = tf.transpose(patches, perm=[1, 0, 2, 3, 4])
        
        return patches


def compute_tile_patch_number(
    sample_image_path: str,
    patch_size: int = 256,
    patch_stride: int = 256,
    rates: List[int] = None
) -> int:
    """Computes the total number of patches that can be extracted from a tile.

    Args:
        sample_image_path (str): Path to the sample image tile.
        patch_size (int, optional): Size of the square patch. Defaults to 256.
        patch_stride (int, optional): Stride for patch extraction. Defaults to 256.
        rates (List[int], optional): Dilation rates for multi-scale extraction. Defaults to [1].

    Returns:
        int: The total number of patches.
    """
    if rates is None:
        rates = [1]
        
    with tf.device('/CPU:0'):
        aerial = tf.constant(imageio.imread(sample_image_path))
        total = 0
        SIZES = [1, patch_size, patch_size, 1]
        STRIDES = [1, patch_stride, patch_stride, 1]
        
        for r in rates:
            patches = tf.image.extract_patches(
                images=tf.expand_dims(aerial, axis=0),
                sizes=SIZES,
                strides=STRIDES,
                rates=[1, r, r, 1],
                padding='VALID'
            )
            total += patches.shape[1] * patches.shape[2]
            
        return int(total)
