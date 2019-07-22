"""
.. module:: WGAN2

WGAN2
*************

:Description: WGAN2

    Borrowed from Keras-GAN code

:Authors: bejar
    

:Version: 

:Created on: 19/07/2019 12:43 

"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import os
import numpy as np
import sys
from time import strftime

from Traffic.Config import Config
from Traffic.Util.Misc import tile_images


class WGAN2():
    image_dim = None
    output_dir = None
    experiment = None
    num_filters = None
    nsamples = None
    dkernel = None
    gkernel = None
    exp = None
    model = 'WGAN2'
    dropout = None

    def __init__(self, image_dim=None, tr_ratio=5,  gen_noise_dim=100, num_filters=(128, 64),
                 dkernel=3, gkernel=3, nsamples=4, dropout=0.25, exp=None):
        config = Config()
        self.output_dir = config.output_dir

        self.num_filters = num_filters  # Number of filters in the kernels

        self.nsamples = nsamples  # Number of samples generated
        self.experiment = f"{strftime('%Y%m%d%H%M%S')}"
        self.exp = exp

        self.dkernel = dkernel  # Size of the discriminator kernels
        self.gkernel = dkernel  # Size of the generator kernels
        self.image_dim = image_dim
        # xdim, ydim, chann = self.image_dim
        self.dropout = dropout


        self.latent_dim = gen_noise_dim

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = tr_ratio
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        xdim, ydim, chann = self.image_dim
        # reduce dimensionality two steps
        xdim = xdim // 4
        ydim = ydim // 4

        model = Sequential()

        model.add(Dense(self.num_filters[0] * xdim * ydim, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((xdim, ydim, self.num_filters[0])))
        model.add(UpSampling2D())
        model.add(Conv2D(self.num_filters[0], kernel_size=self.gkernel, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.num_filters[1], kernel_size=self.gkernel, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(chann, kernel_size=self.gkernel, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=self.dkernel, strides=2, input_shape=self.image_dim, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(32, kernel_size=self.dkernel, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(64, kernel_size=self.dkernel, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(128, kernel_size=self.dkernel, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.image_dim)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size=128, sample_interval=50, verbose=False):

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        self.batch_size = batch_size

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(f"{epoch} [D loss: {1 - d_loss[0]}] [G loss: {1 - g_loss[0]}]")

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, 1 - d_loss[0], 1 - g_loss[0])

    def sample_images(self, epoch, dloss, gloss):
        noise = np.random.normal(0, 1, (self.nsamples * self.nsamples, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        tiled_output = tile_images(gen_imgs, self.nsamples)
        tiled_output = Image.fromarray(tiled_output, mode='RGB')
        outfile = os.path.join(self.output_dir,
                               f'{self.model}-{self.exp}'
                               f'_EP{epoch:03d}'
                               f'_GTR{self.n_critic}'
                               f'_B{self.batch_size}'
                               f'_ND{self.generator_noise_dimensions}'
                               f'_K{self.gkernel}-{self.dkernel}'
                               f'_F{self.num_filters[0]}-{self.num_filters[1]}'
                               f'_D{self.dropout}'
                               f'_LG{gloss:3.4f}'
                               f'_LD{dloss:3.4f}'
                               f'.png')
        tiled_output.save(outfile)


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)
