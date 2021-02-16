## Neural Style transfer

Neural Style transfer is a cool application of convolutional neural networks that combines two images into one.
Specifically, the style of one image is transfered to the content of the second image.

![Figure showing the effect of Neural Style transfer on a picture of Amsterdam in the styles of Munch, Monet, Picasso, and a mosaique.](demo.png)

Neural Style Transfer is based on [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576).
Two images are iteratively merged by transfering the style of one to the content of the other.
The process is based on convolutional neural networks and uses a content cost and a style cost to transfer the visual style.

This implementationn of neural style transfer in Tensorflow/Keras is based on the deeplearning.ai
Coursera course on Convolutional Neural Networks and the Keras code example.

I used a pre-trained VGG19 neural network ([Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)).

Example images to try style transfer and example generated images cam be found in the respective folders.
I particularly like the mosaique which gives striking results after only around 40 iterations.

### Dependencies

1. [Tensorflow2](https://www.tensorflow.org)
2. [numpy](https://numpy.org)

### Usage

The code takes two images (a content and a style image) and generates a new image.
The script can be executed from the command line using:

```
python3 style_transfer.py <content_image_path> <style_image_path>
```

The generated image will be optimized for 140 iterations, generating an image every 5 steps.
The output folder can be specified using the `--out_dir` flag.
An example can be run using `python3 style_transfer.py --demo`.
For a complete description of commands, use

```
python3 style_transfer.py -h
```

To adjust the hyperparameters or number of iterations, feel free to edit `style_transfer.py` itself.

### References

[1] deeplearning.ai Coursera specialization: Convolutional Neural Networks -
[ArtTrans](https://github.com/JudasDie/deeplearning.ai/tree/master/Convolutional%20Neural%20Networks/week4/ArtTrans) \
[2] [Keras Neural Style transfer example](https://keras.io/examples/generative/neural_style_transfer/)
