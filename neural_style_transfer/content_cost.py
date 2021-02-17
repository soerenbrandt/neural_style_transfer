""" Compute the content cost for the neural style transfer.

The cost between the content image C and the generated image G is calculated
based on the layer activations activations_c and activations_g. The cost is
computed as:

    content_cost = 1/(4*n_h*n_w*n_c) * sum((a_C - a_G)^2)
"""

import tensorflow as tf


def compute_content_cost(activation_c, activation_g):
    """Computes the content cost.

    Args:
        activation_c (tf.tensor): hidden layer activations representing content
            of the content image C. tensor of dimension (n_h, n_w, n_c).
        activation_g (tf.tensor): hidden layer activations representing content
            of the generated image G. tensor of dimension (n_h, n_w, n_c).

    Returns:
        content_cost (float): content cost of the image G given C given the
            equation:
                content_cost = 1/(4*n_h*n_w*n_c) * sum((activation_c - activation_g)^2)
    """

    # Retrieve dimensions from activation_c
    n_h, n_w, n_c = activation_c.get_shape().as_list()

    # Reshape activation_c and activation_c
    activation_c_flattened = tf.reshape(activation_c, shape=[-1, n_c])
    activation_g_flattened = tf.reshape(activation_g, shape=[-1, n_c])

    # compute the cost with tensorflow
    content_cost = 1 / (4 * n_h * n_w * n_c) * tf.reduce_sum(
        tf.square(tf.subtract(activation_c_flattened, activation_g_flattened)))

    return content_cost
