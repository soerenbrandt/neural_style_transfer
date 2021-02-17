""" Utilities for neural style transfer.

Neural style transfer generates new images from a content and a style image.
Here are the data processing functions to convert images for use.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error


def preprocess_image(image_path, image_size):
    """Preprocess the image for neural style transfer.

    Args:
        image_path (strr): absolute or relative file path for the image
        image_size (tuple(int,int)): target height and width of the image

    Returns:
        image (tf.tensor): image converted for tf.tensor for neural style transfer
    """
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)

    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)

    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return tf.convert_to_tensor(img)


def deprocess_image(img, image_size):
    """Deprocess tf.tensor back into an image

    Args:
        img (tf.tensor): tensor of generated image
        image_size (tuple(int,int)): image height and width for neural style
            transfer

    Returns:
        [type]: [description]
    """
    # Util function to convert a tensor into a valid image
    img = img.reshape((image_size[0], image_size[1], 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img
