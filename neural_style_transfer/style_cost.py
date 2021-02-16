""" """

import tensorflow as tf


def compute_style_cost(activation_s, activation_g):
    """
    Args:
        activation_s (tf.tensor): hidden layer activations representing style
            of the style image S. tensor of dimension (n_H, n_W, n_C).
        activation_g (tf.tensor): hidden layer activations representing style
            of the generated image G. tensor of dimension (n_H, n_W, n_C).

    Returns:
        J_style (float): content cost of the image G given C given the
            equation:
                J_content = 1/(2*n_H*n_W*n_C)**2 * sum(sum((activation_s - activation_g)^2))
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    n_H, n_W, n_C = activation_s.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    activation_s = tf.reshape(tf.transpose(activation_s), shape=[n_C, -1])
    activation_g = tf.reshape(tf.transpose(activation_g), shape=[n_C, -1])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(activation_s)
    GG = gram_matrix(activation_g)

    # Computing the loss
    J_style_layer = 1 / (2 * n_C * n_H * n_W)**2 * tf.reduce_sum(
        tf.square(tf.subtract(GS, GG)))

    return J_style_layer


def compute_average_style_cost(model_features, style_layers):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model_features: Features for each layer of the model given the input images.
        e.g. layer_features = feature_extractor(input_tensor)
    style_layers (list): A list containing tupeles with the names of the layers
        we would like to extract style from and a weight coefficient for each
        layer, e.g.:

            [
                ('conv1', 0.5),
                ('conv2', 0.5)
            ]

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in style_layers:
        # Select the output tensor of the currently selected layer
        layer_features = model_features[layer_name]

        # Set activation_s and activation_g to be the hidden layer activation
        # from the layer we have selected
        activation_s = layer_features[1, :, :, :]
        activation_g = layer_features[2, :, :, :]

        # Compute style_cost for the current layer
        J_style_layer = compute_style_cost(activation_s, activation_g)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    gram -- Gram matrix of A, of shape (n_C, n_C)
    """
    gram = tf.matmul(A, A, transpose_b=True)
    return gram
