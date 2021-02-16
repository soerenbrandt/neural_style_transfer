import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19  # pylint: disable=import-error


def preprocess_image(image_path, image_size):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)

    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)

    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return tf.convert_to_tensor(img)


def deprocess_image(x, image_size):
    # Util function to convert a tensor into a valid image
    x = x.reshape((image_size[0], image_size[1], 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
