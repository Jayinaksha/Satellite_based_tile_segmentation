"""Dataset pipeline module for high-resolution aerial tiles.

Builds a tf.data.Dataset pipeline that loads images and masks, extracts patches,
and applies data augmentation.
"""

from typing import Dict, Any, Tuple, Optional
import os
import tensorflow as tf
import imageio.v2 as imageio

from src.data.patch_extraction import make_patches, make_patches_scaled
from src.data.augmentation import make_augmentation, no_augmentation


def get_image(srcpath: tf.Tensor, refpath: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads an image and corresponding mask from disk.

    Args:
        srcpath (tf.Tensor): The path to the aerial image tile.
        refpath (tf.Tensor): The path to the mask image tile.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the image and mask tensors.
    """
    color_list_local = [[0, 0, 0], [255, 255, 255]]
    
    srcpath_str = srcpath.numpy().decode('utf-8')
    refpath_str = refpath.numpy().decode('utf-8')
    
    src = imageio.imread(srcpath_str)
    ref = imageio.imread(refpath_str)
    
    if len(ref.shape) == 2:
        ref = ref // 255
        ref = tf.cast(ref, tf.int32)
        ref = tf.gather(color_list_local, ref)
        ref = tf.cast(ref, tf.uint8)
        
    return src, ref


def tf_get_image(srcpath: tf.Tensor, refpath: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow wrapper for `get_image` using py_function.

    Args:
        srcpath (tf.Tensor): A string tensor with the image file path.
        refpath (tf.Tensor): A string tensor with the mask file path.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: The loaded image and mask tensors.
    """
    src, ref = tf.py_function(func=get_image, inp=[srcpath, refpath], Tout=[tf.uint8, tf.uint8])
    return src, ref


def get_dataset(
    img_dir: str,
    mask_dir: str,
    batch_size: int,
    config: Dict[str, Any],
    shuffle: bool = True,
    repeat: bool = False,
    augment: bool = True,
    scaled: bool = False,
    cachefile: Optional[str] = None
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset for training or evaluation.

    Args:
        img_dir (str): Directory containing the image tiles.
        mask_dir (str): Directory containing the mask tiles.
        batch_size (int): Batch size for the dataset.
        config (Dict[str, Any]): Configuration dictionary containing parameters like 
            patch_size, patch_stride, color_list, etc.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        repeat (bool, optional): Whether to repeat the dataset indefinitely. Defaults to False.
        augment (bool, optional): Whether to apply data augmentation. Defaults to True.
        scaled (bool, optional): Whether to extract multi-scale patches. Defaults to False.
        cachefile (Optional[str], optional): Path to cache file. Defaults to None.

    Returns:
        tf.data.Dataset: A TensorFlow dataset yielding batches of (image, mask) pairs.
    """
    # Fetch file paths
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if not f.startswith('.')]
    img_paths = sorted(img_paths)
    
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if not f.startswith('.')]
    mask_paths = sorted(mask_paths)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(img_paths))

    if repeat:
        dataset = dataset.repeat()

    # Load images
    dataset = dataset.map(tf_get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Extract patches
    patch_size = config.get('patch_size', 256)
    patch_stride = config.get('patch_stride', 256)
    
    if scaled:
        dataset = dataset.map(
            lambda x, y: make_patches_scaled(x, y, patch_size, patch_stride),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda x, y: make_patches(x, y, patch_size, patch_stride),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Unbatch the patches so that each patch is a separate element
    dataset = dataset.unbatch()
    
    if shuffle:
        # Shuffle patches
        dataset = dataset.shuffle(buffer_size=1000)

    # Apply augmentation and mask to label mapping
    color_list = config.get('color_list', [[0, 0, 0], [255, 255, 255]])
    sparse = config.get('sparse', True)
    
    if augment:
        dataset = dataset.map(
            lambda x: make_augmentation(x, color_list, sparse),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda x: no_augmentation(x, color_list, sparse),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)

    # Cache if specified
    if cachefile is not None:
        dataset = dataset.cache(cachefile)
        
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
