""" Compute the variation cost for the neural style transfer.

The total variation cost imposes local spatial continuity between the pixels of
the generated image, giving it visual coherence. This is often helpful to
produce better results but is not absolutely needed.
"""

import tensorflow as tf


def compute_variation_cost(image, height, width):
    """Compute the variation cost to keep the image cohesive.

    Args:
        image (tf.tensor): matrix of the generated image, shape (1, height, width, 3)
        height (int): height of the image
        width (int): width of the image

    Returns:
        cost (float)): variation cost of the image.
    """
    a = tf.square(image[:, :height - 1, :width - 1, :] -
                  image[:, 1:, :width - 1, :])
    b = tf.square(image[:, :height - 1, :width - 1, :] -
                  image[:, :height - 1, 1:, :])

    return tf.reduce_sum(tf.pow(a + b, 1.25))
