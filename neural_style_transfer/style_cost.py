""" Compute the style cost for the neural style transfer.

The cost between the style image S and the generated image G is calculated
based on the layer activations activations_s and activations_g. The cost is
computed as:

    style_cost = 1/(2*n_h*n_w*n_c)**2 * sum(sum((activation_s - activation_g)^2))

In addition, the style cost is best averaged over several layers using:

    `compute_average_style_cost(model_features, style_layers)`
"""

import tensorflow as tf


def compute_style_cost(activation_s, activation_g):
    """Compute the style cost.

    Args:
        activation_s (tf.tensor): hidden layer activations representing style
            of the style image S. tensor of dimension (n_h, n_w, n_c).
        activation_g (tf.tensor): hidden layer activations representing style
            of the generated image G. tensor of dimension (n_h, n_w, n_c).

    Returns:
        style_cost (float): content cost of the image G given C given the
            equation:
                style_cost = 1/(2*n_h*n_w*n_c)**2 * sum(sum((activation_s - activation_g)^2))
    """

    # Retrieve dimensions from a_G
    n_h, n_w, n_c = activation_s.get_shape().as_list()

    # Reshape the images to have them of shape (n_c, n_h*n_w)
    activation_s = tf.reshape(tf.transpose(activation_s), shape=[n_c, -1])
    activation_g = tf.reshape(tf.transpose(activation_g), shape=[n_c, -1])

    # Computing gram_matrices for both images S and G
    gram_matrix_s = gram_matrix(activation_s)
    gram_matrix_g = gram_matrix(activation_g)

    # Computing the loss
    style_cost = 1 / (2 * n_c * n_h * n_w)**2 * tf.reduce_sum(
        tf.square(tf.subtract(gram_matrix_s, gram_matrix_g)))

    return style_cost


def compute_average_style_cost(model_features, style_layers):
    """
    Computes the overall style cost from several chosen layers

    Args:
        model_features: Features for each layer of the model given the input
            images, e.g., layer_features = feature_extractor(input_tensor)
        style_layers (list): A list containing tupeles with the names of the
            layers we would like to extract style from and a weight coefficient
            for each layer, e.g.:

                [
                    ('conv1', 0.5),
                    ('conv2', 0.5)
                ]

    Returns:
        avg_style_cost (float): Average style cost (see equation above) given
            the equation:
                avg_style_cost = 1/(2*n_h*n_w*n_c)**2 * sum(sum((activation_s - activation_g)^2))
    """
    # initialize the overall style cost
    avg_style_cost = 0

    for layer_name, coeff in style_layers:
        # Select the output tensor of the currently selected layer
        layer_features = model_features[layer_name]

        # Set activation_s and activation_g to be the hidden layer activation
        # from the layer we have selected
        activation_s = layer_features[1, :, :, :]
        activation_g = layer_features[2, :, :, :]

        # Compute style_cost for the current layer
        style_cost = compute_style_cost(activation_s, activation_g)

        # Add coeff * style_cost of this layer to overall style cost
        avg_style_cost += coeff * style_cost

    return avg_style_cost


def gram_matrix(matrix):
    """
    Args:
        matrix (tf.tensor): matrix of image activations, shape (n_c, n_h*n_w)

    Returns:
        gram (tf.tensor): Gram matrix of matrix, of shape (n_c, n_c)
    """
    gram = tf.matmul(matrix, matrix, transpose_b=True)
    return gram
