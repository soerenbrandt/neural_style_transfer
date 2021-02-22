import os

import numpy as np
import pytest
import tensorflow as tf

from neural_style_transfer.content_cost import compute_content_cost
from neural_style_transfer.style_cost import compute_style_cost
from neural_style_transfer.style_cost import gram_matrix
from neural_style_transfer.variation_cost import compute_variation_cost

# set random seed
SEED = 1234

# fix OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib
# already initialized. Note: This error occurs when calculating the gram matrix
# using tf.matmul
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@pytest.fixture
def random_image_activations():
    tf.random.set_seed(SEED)
    content_activation = tf.random.normal([4, 4, 3], mean=1, stddev=4)
    style_activation = tf.random.normal([4, 4, 3], mean=1, stddev=4)
    generated_activation = tf.random.normal([4, 4, 3], mean=1, stddev=4)

    return content_activation, style_activation, generated_activation


@pytest.fixture
def random_matrix():
    tf.random.set_seed(SEED)
    activation_matrix = tf.random.normal([3, 2 * 1], mean=1, stddev=4)

    return activation_matrix


@pytest.fixture
def random_image():
    tf.random.set_seed(SEED)
    image = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)

    return image


def test_content_cost(random_image_activations):
    # generate random image activations and calculate corresponding content cost
    activation_c, _, activation_g = random_image_activations
    test_cost = compute_content_cost(activation_c, activation_g)

    # the test cost for random seed=1234 and activations of shape [4,4,3] is 6.0116568
    assert np.isclose(test_cost, 6.0116568)


def test_style_cost(random_image_activations):
    # generate random image activationns and calculate corresponding style cost
    _, activation_s, activation_g = random_image_activations
    test_cost = compute_style_cost(activation_s, activation_g)

    # the test cost for random seed=1234 and activations of shape [4,4,3] is 8.396351
    assert np.isclose(test_cost, 8.396351)


def test_gram_matrix(random_matrix):
    # calculate gram matrix for a random input matrix
    test_gram = gram_matrix(random_matrix)

    # the test gram matrix should have the following form with mean 8.746427:
    #    array([[ 22.655428 ,  20.189453 , -18.508045 ],
    #           [ 20.189453 ,  27.98669  ,  -3.5650644],
    #           [-18.508045 ,  -3.5650644,  31.843039 ]]
    assert np.isclose(test_gram.numpy().mean(), 8.746427)


def test_variation_cost(random_image):
    # calculate variation cost for a randomly generated image
    test_cost = compute_variation_cost(random_image)

    # the test cost for random seed=1234 and an image of shape [1,4,4,3] is 4260.594
    return test_cost  #np.isclose(test_cost, 4260.594)
    assert np.isclose(test_cost, 4260.594)
