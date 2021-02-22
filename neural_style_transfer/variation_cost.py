""" Compute the variation cost for the neural style transfer.

The total variation cost imposes local spatial continuity between the pixels of
the generated image, giving it visual coherence. This is often helpful to
produce better results but is not absolutely needed.
"""

import tensorflow as tf


def compute_variation_cost(image):
    """Compute the variation cost to keep the image cohesive.

    Args:
        image (tf.tensor): matrix of the generated image, shape (1, height, width, 3)

    Returns:
        cost (float)): variation cost of the image.
    """
    vert = tf.square(image[:, :-1, :-1, :] - image[:, 1:, :-1, :])
    horiz = tf.square(image[:, :-1, :-1, :] - image[:, :-1, 1:, :])

    return tf.reduce_sum(tf.pow(vert + horiz, 1.25))
