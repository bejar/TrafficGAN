"""
Title: WGAN-GP overriding `Model.train_step`
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/05/9
Last modified: 2020/05/9
Description: Implementation of Wasserstein GAN with Gradient Penalty.
"""

from time import strftime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from Traffic.Config import Config

IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 256

# Size of the noise vector
noise_dim = 64


def conv_block(
        x, filters, activation, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True,
        use_bn=False, use_dropout=False, drop_value=0.5):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def upsample_block(
        x, filters, activation, kernel_size=(3, 3), strides=(1, 1), up_size=(2, 2), padding="same", use_bn=False,
        use_bias=True, use_dropout=False, drop_value=0.3):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


# g_model = get_generator_model()
# g_model.summary()


class WGANGP(keras.Model):
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

    def __init__(
            self,
            discriminator_extra_steps=3,
            gp_weight=10.0, image_dim=None, tr_ratio=5, gen_noise_dim=100, num_filters=(128, 64), dkernel=3, gkernel=3,
            nsamples=4, dropout=0.25, resize=2, exp=None
    ):
        super(WGANGP, self).__init__()
        self.discriminator = self.make_discriminator()
        self.generator = self.make_generator()
        self.latent_dim = gen_noise_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        config = Config()
        self.output_dir = config.output_dir
        self.TRAINING_RATIO = tr_ratio  # Traning/generator ratio
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

        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.d_loss_fn = self.discriminator_loss
        self.g_loss_fn = self.generator_loss

    @staticmethod
    def make_discriminator():
        img_input = layers.Input(shape=IMG_SHAPE)
        # Zero pad the input to make the input images size to (32, 32, 1).
        x = layers.ZeroPadding2D((2, 2))(img_input)
        x = conv_block(
            x, 64, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True, activation=layers.LeakyReLU(0.2),
            use_dropout=False, drop_value=0.3)
        x = conv_block(
            x, 128, kernel_size=(5, 5), strides=(2, 2), use_bn=False, activation=layers.LeakyReLU(0.2), use_bias=True,
            use_dropout=True, drop_value=0.3)
        x = conv_block(
            x, 256, kernel_size=(5, 5), strides=(2, 2), use_bn=False, activation=layers.LeakyReLU(0.2), use_bias=True,
            use_dropout=True, drop_value=0.3)
        x = conv_block(
            x, 512, kernel_size=(5, 5), strides=(2, 2), use_bn=False, activation=layers.LeakyReLU(0.2), use_bias=True,
            use_dropout=False, drop_value=0.3)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)

        d_model = keras.models.Model(img_input, x, name="discriminator")
        return d_model

    @staticmethod
    def make_generator():
        noise = layers.Input(shape=(noise_dim,))
        x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Reshape((4, 4, 256))(x)
        x = upsample_block(
            x, 128, layers.LeakyReLU(0.2), strides=(1, 1), use_bias=False, use_bn=True, padding="same",
            use_dropout=False)
        x = upsample_block(
            x, 64, layers.LeakyReLU(0.2), strides=(1, 1), use_bias=False, use_bn=True, padding="same",
            use_dropout=False)
        x = upsample_block(
            x, 1, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
        )
        x = layers.Cropping2D((2, 2))(x)

        g_model = keras.models.Model(noise, x, name="generator")
        return g_model

    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


"""
## Create a Keras callback that periodically saves generated images
"""


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
