""" """  # TODO add description

import argparse
from pathlib import Path
# fix OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19  # pylint: disable = import-error

from neural_style_transfer.content_cost import compute_content_cost
from neural_style_transfer.style_cost import compute_average_style_cost
from neural_style_transfer.variation_cost import compute_variation_cost
from neural_style_transfer.utils import preprocess_image, deprocess_image

# define terminal inputs
parser = argparse.ArgumentParser(description='Neural Style Transfer.')
parser.add_argument('images',
                    metavar='images',
                    type=str,
                    nargs='*',
                    help='A set of two image file paths. '
                    'First image is the content image. '
                    'Second image is the style image')
parser.add_argument('--out_dir',
                    dest='out_dir',
                    type=str,
                    nargs=None,
                    default='output_images',
                    help='Output directory')
parser.add_argument('--demo',
                    dest='demo',
                    action='store_const',
                    const=True,
                    default=False,
                    help='Run a set of demo images. Ignores image paths.')

DEMO_CONTENT_IMAGE = Path('example_images/amsterdam.jpg')
DEMO_STYLE_IMAGE = Path('example_images/munch_the_scream.jpg')

OUT_IMAGE_HEIGHT = 400

ALPHA = 1e-6  # 0.025
BETA = 1e-6  # 1
GAMMA = 2.5e-8  # 1e-4
ITERATIONS = 140
GENERATE_IMAGE_EVERY = 5

# List of layers to use for the style loss.
STYLE_LAYER_NAMES = [('block1_conv1', 0.2), ('block2_conv1', 0.2),
                     ('block3_conv1', 0.2), ('block4_conv1', 0.2),
                     ('block5_conv1', 0.2)]
# The layer to use for the content loss.
CONTENT_LAYER_NAME = "block5_conv2"


def run_style_transfer(content_image_path, style_image_path, out_image_prefix):
    # TODO description
    # Dimensions of the generated picture.
    width, height = keras.preprocessing.image.load_img(content_image_path).size
    image_size = (OUT_IMAGE_HEIGHT, int(width * OUT_IMAGE_HEIGHT / height))

    content_image = preprocess_image(content_image_path, image_size)
    style_image = preprocess_image(style_image_path, image_size)
    generated_image = tf.Variable(content_image)

    ## Load the pre-trained VGG model
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    ## Define SGD optimizer
    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0,
                                                    decay_steps=100,
                                                    decay_rate=0.96))

    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0,
                                                    decay_steps=100,
                                                    decay_rate=0.96))

    for i in range(1, ITERATIONS + 1):
        loss, grads = compute_loss_and_grads(feature_extractor, generated_image,
                                             content_image, style_image,
                                             image_size)
        optimizer.apply_gradients([(grads, generated_image)])
        if i % GENERATE_IMAGE_EVERY == 0:
            print("\rIteration %d: loss=%.0f" % (i, loss))
            img = deprocess_image(generated_image.numpy(), image_size)
            fname = out_image_prefix.parent.joinpath(
                f'{out_image_prefix.name}_at_iteration_{i}.png').absolute()
            keras.preprocessing.image.save_img(fname, img)
        else:
            print("\rIteration %d: loss=%.0f" % (i, loss), end='')


def compute_loss(feature_extractor, generated_image, content_image, style_image,
                 image_size):
    input_tensor = tf.concat([content_image, style_image, generated_image],
                             axis=0)
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Compute content cost
    layer_features = features[CONTENT_LAYER_NAME]
    base_image_features = layer_features[0, :, :, :]
    generated_image_features = layer_features[2, :, :, :]
    content_cost = compute_content_cost(base_image_features,
                                        generated_image_features)

    # Compute style cost
    style_cost = compute_average_style_cost(features, STYLE_LAYER_NAMES)

    # Compute variational cost
    variational_cost = compute_variation_cost(generated_image, image_size[0],
                                              image_size[1])

    # Return computed loss
    loss = ALPHA * content_cost + BETA * style_cost + GAMMA * variational_cost
    return loss


@tf.function
def compute_loss_and_grads(feature_extractor, generated_image, content_image,
                           style_image, image_size):
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor, generated_image, content_image,
                            style_image, image_size)
        grads = tape.gradient(loss, generated_image)
    return loss, grads


if __name__ == '__main__':
    args = parser.parse_args()

    # create output directory if not exists
    print(args.out_dir)
    print(type(args.out_dir))
    out_dir = Path(args.out_dir)
    if not (out_dir.exists() and out_dir.is_dir()):
        out_dir.mkdir(parents=True, exist_ok=False)

    # define image paths for input and output images
    image_prefix = out_dir.joinpath('generated_image')
    if args.demo:
        run_style_transfer(DEMO_CONTENT_IMAGE, DEMO_STYLE_IMAGE, image_prefix)
    else:
        assert len(args.images) == 2, "Neural Style transfer requires 2 images."
        run_style_transfer(Path(args.images[0]), Path(args.images[1]),
                           image_prefix)
