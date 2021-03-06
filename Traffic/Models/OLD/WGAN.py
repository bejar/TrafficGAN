"""
.. module:: WGAN

WGAN
*************

:Description: WGAN

    WGAN code extracted from Keras examples

An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its
gradient norm moves away from 1. This is included because the Earth Mover (EM) distance
used in WGANs is only easy to calculate for 1-Lipschitz functions (i.e. functions where
the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values
[-0.01, 0.01]. However, this drastically reduced network capacity. Penalizing the
gradient norm is more natural, but this requires second-order gradients. These are not
supported for some tensorflow ops (particularly MaxPool and AveragePool) in the current
release (1.0.x), but they are supported in the current nightly builds
(1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for
downsampling. If you wish to use pooling operations in your discriminator, please ensure
you update Tensorflow to 1.1.0-rc1 or higher. I haven't tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or
remove the calls to generate_images.


:Version: 

:Created on: 09/07/2019 13:19 

"""

import os
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers.merge import _Merge
from tensorflow.keras.layers import Convolution2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial
from tqdm import tqdm
import sys
from time import strftime

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! '
          'Please install it (e.g. with pip install pillow)')
    exit()

from Traffic.Util.Losses import wasserstein_loss, gradient_penalty_loss
from Traffic.Util.Misc import tile_images
from Traffic.Config import Config


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def __init__(self, batch_size, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.BATCH_SIZE = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class WGAN:
    BATCH_SIZE = 64
    # The training ratio is the number of discriminator updates
    # per generator update. The paper uses 5.
    TRAINING_RATIO = 5
    GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
    generator_noise_dimensions = 100
    image_dim = None
    output_dir = None
    experiment = None
    num_filters = None
    imggen = None
    nsamples = None
    dense = None
    ckernel = None
    exp = None
    model = 'WGAN'
    dropout = None
    resize = None

    def __init__(self, image_dim=None, tr_ratio=5, gen_noise_dim=100, num_filters=(128, 64), dkernel=3, gkernel=3,
                 nsamples=4, dropout=0.25, resize=2, exp=None):
        """
        Parameter initialization
        :param batch:
        :param tr_ratio:
        :param gr_penalty:
        """
        config = Config()
        self.output_dir = config.output_dir
        self.TRAINING_RATIO = tr_ratio  # Traning/generator ratio
        self.GRADIENT_PENALTY_WEIGHT = 10
        self.generator_noise_dimensions = gen_noise_dim  # Dimension of the noise
        self.num_filters = num_filters  # Number of filters in the kernels

        self.nsamples = nsamples  # Number of samples generated

        self.experiment = f"{strftime('%Y%m%d%H%M%S')}"
        self.exp = exp
        self.image_dim = image_dim
        self.dkernel = dkernel  # Size of the discriminator kernels
        self.gkernel = dkernel  # Size of the generator kernels
        self.dropout = dropout
        self.dense = 512
        self.resize = resize

    def make_generator(self):
        """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
        and outputs images of size correspondinh to the input images sizes."""

        xdim, ydim, chann = self.image_dim

        # reduce dimensionality n steps
        rdim = 2**self.resize
        if (xdim % rdim !=0) or (ydim %rdim!=0):
            raise NameError('invalid upscaling')

        xdim = xdim // rdim
        ydim = ydim // rdim

        model = Sequential()
        model.add(Dense(self.dense, input_dim=self.generator_noise_dimensions))
        model.add(LeakyReLU())
        model.add(Dense(self.num_filters[0] * xdim * ydim))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((xdim, ydim, self.num_filters[0]), input_shape=(self.num_filters[0] * xdim * ydim,)))
        bn_axis = -1

        for i in range(self.resize):
            model.add(Conv2DTranspose(self.num_filters[0], (self.gkernel, self.gkernel), strides=2, padding='same'))
            model.add(BatchNormalization(axis=bn_axis))
            model.add(LeakyReLU())
            model.add(Convolution2D(self.num_filters[1], (self.gkernel, self.gkernel), padding='same'))
            model.add(BatchNormalization(axis=bn_axis))
            model.add(LeakyReLU())

        # model.add(Conv2DTranspose(self.num_filters[1], (self.gkernel, self.gkernel), strides=2, padding='same'))
        # model.add(BatchNormalization(axis=bn_axis))
        # model.add(LeakyReLU())

        # Because we normalized training inputs to lie in the range [-1, 1],
        # the tanh function should be used for the output of the generator to ensure
        # its output also lies in this range.
        model.add(Convolution2D(chann, (self.gkernel, self.gkernel), padding='same', activation='tanh'))
        return model

    def make_discriminator(self):
        """Creates a discriminator model that takes an image as input and outputs a single
        value, representing whether the input is real or generated. Unlike normal GANs, the
        output is not sigmoid and does not represent a probability! Instead, the output
        should be as large and negative as possible for generated inputs and as large and
        positive as possible for real inputs.
        Note that the improved WGAN paper suggests that BatchNormalization should not be
        used in the discriminator."""
        model = Sequential()

        model.add(Convolution2D(self.num_filters[1], (self.dkernel, self.dkernel), padding='same',
                                input_shape=self.image_dim))
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout))
        model.add(Convolution2D(self.num_filters[0], (self.dkernel, self.dkernel), kernel_initializer='he_normal',
                                strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout))
        model.add(Convolution2D(self.num_filters[0], (self.dkernel, self.dkernel), kernel_initializer='he_normal',
                                padding='same',
                                strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(self.dense, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(Dense(self.dense//2, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1, kernel_initializer='he_normal'))
        return model

    def generate_images(self, generator_model, epoch, gloss, dloss):
        """Feeds random seeds into the generator and tiles and saves the output to a PNG
        file."""
        test_image_stack = generator_model.predict(
            np.random.rand(self.nsamples * self.nsamples, self.generator_noise_dimensions))
        test_image_stack = (test_image_stack * 127.5) + 127.5
        test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
        tiled_output = tile_images(test_image_stack, self.nsamples)
        tiled_output = Image.fromarray(tiled_output, mode='RGB')
        outfile = os.path.join(self.output_dir,
                               f'{self.model}-{self.exp}'
                               f'_EP{epoch:03d}'
                               f'_GTR{self.TRAINING_RATIO}'
                               f'_B{self.BATCH_SIZE}'
                               f'_ND{self.generator_noise_dimensions}'
                               f'_K{self.gkernel}-{self.dkernel}'
                               f'_F{self.num_filters[0]}-{self.num_filters[1]}'
                               f'_D{self.dense}'
                               f'_DR{self.dropout}'
                               f'_U{self.resize}'
                               f'_LG{gloss:3.4f}'
                               f'_LD{dloss:3.4f}'
                               f'.png')
        tiled_output.save(outfile)

    def train(self, X_train, epochs, batch_size=128, sample_interval=50, verbose=False):
        self.BATCH_SIZE = batch_size
        self.imggen = sample_interval

        # self.image_dim=X_train.shape[1:]

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        # Now we initialize the generator and discriminator.
        generator = self.make_generator()
        discriminator = self.make_discriminator()

        # The generator_model is used when we want to train the generator layers.
        # As such, we ensure that the discriminator layers are not trainable.
        # Note that once we compile this model, updating .trainable will have no effect within
        # it. As such, it won't cause problems if we later set discriminator.trainable = True
        # for the discriminator_model, as long as we compile the generator_model first.
        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False
        generator_input = Input(shape=(self.generator_noise_dimensions,))
        generator_layers = generator(generator_input)
        discriminator_layers_for_generator = discriminator(generator_layers)
        generator_model = Model(inputs=[generator_input],
                                outputs=[discriminator_layers_for_generator])
        # We use the Adam paramaters from Gulrajani et al.
        generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=wasserstein_loss)

        # Now that the generator_model is compiled, we can make the discriminator
        # layers trainable.
        for layer in discriminator.layers:
            layer.trainable = True
        for layer in generator.layers:
            layer.trainable = False
        discriminator.trainable = True
        generator.trainable = False

        # The discriminator_model is more complex. It takes both real image samples and random
        # noise seeds as input. The noise seed is run through the generator model to get
        # generated images. Both real and generated images are then run through the
        # discriminator. Although we could concatenate the real and generated images into a
        # single tensor, we don't (see model compilation for why).
        real_samples = Input(shape=X_train.shape[1:])
        generator_input_for_discriminator = Input(shape=(self.generator_noise_dimensions,))
        generated_samples_for_discriminator = generator(generator_input_for_discriminator)
        discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = discriminator(real_samples)

        # We also need to generate weighted-averages of real and generated samples,
        # to use for the gradient norm penalty.
        averaged_samples = RandomWeightedAverage(self.BATCH_SIZE)([real_samples,
                                                                   generated_samples_for_discriminator])
        # We then run these samples through the discriminator as well. Note that we never
        # really use the discriminator output for these samples - we're only running them to
        # get the gradient norm for the gradient penalty loss.
        averaged_samples_out = discriminator(averaged_samples)

        # The gradient penalty loss function requires the input averaged samples to get
        # gradients. However, Keras loss functions can only have two arguments, y_true and
        # y_pred. We get around this by making a partial() of the function with the averaged
        # samples here.
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT)

        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Keras requires that inputs and outputs have the same number of samples. This is why
        # we didn't concatenate the real samples and generated samples before passing them to
        # the discriminator: If we had, it would create an output with 2 * BATCH_SIZE samples,
        # while the output of the "averaged" samples for gradient penalty
        # would have only BATCH_SIZE samples.

        # If we don't concatenate the real and generated samples, however, we get three
        # outputs: One of the generated samples, one of the real samples, and one of the
        # averaged samples, all of size BATCH_SIZE. This works neatly!
        discriminator_model = Model(inputs=[real_samples,
                                            generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])

        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
        # the real and generated samples, and the gradient penalty loss for the averaged samples
        discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                    loss=[wasserstein_loss,
                                          wasserstein_loss,
                                          partial_gp_loss])

        # We make three label vectors for training. positive_y is the label vector for real
        # samples, with value 1. negative_y is the label vector for generated samples, with
        # value -1. The dummy_y vector is passed to the gradient_penalty loss function and
        # is not used.
        positive_y = np.ones((self.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((self.BATCH_SIZE, 1), dtype=np.float32)

        if verbose:
            for epoch in tqdm(range(epochs), file=sys.stdout, desc='Epochs'):
                np.random.shuffle(X_train)
                discriminator_loss = []
                generator_loss = []
                minibatches_size = self.BATCH_SIZE * self.TRAINING_RATIO
                for i in tqdm(range(int(X_train.shape[0] // (self.BATCH_SIZE * self.TRAINING_RATIO))), file=sys.stdout,
                              desc='Gen'):
                    discriminator_minibatches = X_train[i * minibatches_size:
                                                        (i + 1) * minibatches_size]
                    for j in tqdm(range(self.TRAINING_RATIO), file=sys.stdout, desc='Disc'):
                        image_batch = discriminator_minibatches[j * self.BATCH_SIZE:
                                                                (j + 1) * self.BATCH_SIZE]
                        noise = np.random.rand(self.BATCH_SIZE, self.generator_noise_dimensions).astype(np.float32)
                        discriminator_loss.append(discriminator_model.train_on_batch(
                            [image_batch, noise],
                            [positive_y, negative_y, dummy_y]))
                    generator_loss.append(generator_model.train_on_batch(np.random.rand(self.BATCH_SIZE,
                                                                                        self.generator_noise_dimensions),
                                                                         positive_y))

                # Still needs some code to display losses from the generator and discriminator,
                # progress bars, etc.
                if epoch % self.imggen == 0:
                    print(generator_loss[-1])
                    self.generate_images(generator, epoch, generator_loss[-1], np.mean(discriminator_loss[-1]))
            self.generate_images(generator, epoch, generator_loss[-1], np.mean(discriminator_loss[-1]))
        else:
            for epoch in range(epochs):
                np.random.shuffle(X_train)
                print("Epoch: ", epoch)
                print("Number of batches: ", int(X_train.shape[0] // self.BATCH_SIZE))
                discriminator_loss = []
                generator_loss = []
                minibatches_size = self.BATCH_SIZE * self.TRAINING_RATIO
                for i in range(int(X_train.shape[0] // (self.BATCH_SIZE * self.TRAINING_RATIO))):
                    discriminator_minibatches = X_train[i * minibatches_size:
                                                        (i + 1) * minibatches_size]
                    for j in range(self.TRAINING_RATIO):
                        image_batch = discriminator_minibatches[j * self.BATCH_SIZE:
                                                                (j + 1) * self.BATCH_SIZE]
                        noise = np.random.rand(self.BATCH_SIZE, self.generator_noise_dimensions).astype(np.float32)
                        discriminator_loss.append(discriminator_model.train_on_batch(
                            [image_batch, noise],
                            [positive_y, negative_y, dummy_y]))
                    generator_loss.append(generator_model.train_on_batch(np.random.rand(self.BATCH_SIZE,
                                                                                        self.generator_noise_dimensions),
                                                                         positive_y))
                # Still needs some code to display losses from the generator and discriminator,
                # progress bars, etc.
                if epoch % self.imggen == 0:
                    self.generate_images(generator, epoch, generator_loss[-1], np.mean(discriminator_loss[-1]))
                for gl in generator_loss:
                    print(gl)
            self.generate_images(generator, epoch, generator_loss[-1], np.mean(discriminator_loss[-1]))
