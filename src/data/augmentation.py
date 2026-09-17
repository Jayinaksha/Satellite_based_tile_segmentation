"""Data augmentation module for aerial images and masks.

Provides functionalities to apply data augmentations to image-mask pairs
during training, and preprocessing utilities.
"""

from typing import List, Tuple
import tensorflow as tf
from src.utils.mask_utils import mask2label


def make_augmentation(
    pair_batch: tf.Tensor,
    color_list: List[List[int]],
    sparse: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies augmentation (flips, rotations) to an image-mask pair.

    Args:
        pair_batch (tf.Tensor): A tensor containing an image and mask pair.
        color_list (List[List[int]]): List of colors for mapping masks to labels.
        sparse (bool, optional): Whether to return sparse labels. Defaults to True.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Augmented image and corresponding label mask.
    """
    with tf.device('/CPU:0'):
        do_flip = tf.random.uniform([]) > 0.5
        do_rot90 = tf.random.uniform([], maxval=4, dtype=tf.int32)
        
        pair_batch = tf.cond(
            do_flip,
            lambda: tf.image.flip_left_right(pair_batch),
            lambda: pair_batch
        )
        pair_batch = tf.image.rot90(pair_batch, do_rot90)
        pair_batch = tf.image.convert_image_dtype(pair_batch, tf.float32)
        
        img = pair_batch[0]
        mask = pair_batch[1]
        
        mask = mask2label(mask, colors=color_list, sparse=sparse)
        return img, mask


def no_augmentation(
    pair_batch: tf.Tensor,
    color_list: List[List[int]],
    sparse: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies basic preprocessing without augmentation to an image-mask pair.

    Args:
        pair_batch (tf.Tensor): A tensor containing an image and mask pair.
        color_list (List[List[int]]): List of colors for mapping masks to labels.
        sparse (bool, optional): Whether to return sparse labels. Defaults to True.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Processed image and corresponding label mask.
    """
    with tf.device('/CPU:0'):
        pair_batch = tf.image.convert_image_dtype(pair_batch, tf.float32)
        
        img = pair_batch[0]
        mask = pair_batch[1]
        
        mask = mask2label(mask, colors=color_list, sparse=sparse)
        return img, mask
