import tensorflow as tf


def compute_variation_cost(image, height, width):
    # TODO add description
    a = tf.square(image[:, :height - 1, :width - 1, :] -
                  image[:, 1:, :width - 1, :])
    b = tf.square(image[:, :height - 1, :width - 1, :] -
                  image[:, :height - 1, 1:, :])

    return tf.reduce_sum(tf.pow(a + b, 1.25))
