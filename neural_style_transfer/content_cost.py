""" Compute the content cost for the neural style transfer.

The cost between the content image C and the generated image G is calculated
based on the layer activations activations_c and activations_g. The cost is
computed as:

    J_content = 1/(4*n_H*n_W*n_C) * sum((a_C - a_G)^2)
"""

import tensorflow as tf


def compute_content_cost(activation_c, activation_g):
    """
    Computes the content cost

    Args:
        activation_c (tf.tensor): hidden layer activations representing content
            of the content image C. tensor of dimension (n_H, n_W, n_C).
        activation_g (tf.tensor): hidden layer activations representing content
            of the generated image G. tensor of dimension (n_H, n_W, n_C).

    Returns:
        J_content (float): content cost of the image G given C given the
            equation:
                J_content = 1/(4*n_H*n_W*n_C) * sum((activation_c - activation_g)^2)
    """

    # Retrieve dimensions from activation_c
    n_H, n_W, n_C = activation_c.get_shape().as_list()

    # Reshape activation_c and activation_c
    activation_c_flattened = tf.reshape(activation_c, shape=[-1, n_C])
    activation_g_flattened = tf.reshape(activation_g, shape=[-1, n_C])

    # compute the cost with tensorflow
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(
        tf.square(tf.subtract(activation_c_flattened, activation_g_flattened)))

    return J_content
