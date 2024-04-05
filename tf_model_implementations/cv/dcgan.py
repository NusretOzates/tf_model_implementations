# https://arxiv.org/pdf/1511.06434.pdf
import keras
import tensorflow as tf
import numpy as np
#import opendatasets as od
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    ReLU,
    Rescaling,
    Reshape,
)
from matplotlib import pyplot as plt
from tqdm import tqdm

# Mixed precision training
keras.mixed_precision.set_global_policy("mixed_float16")

batch_size = 32
image_size = (512, 512)
latent_dim = 100
use_bias = False

# To scale real images between -1 and 1 because tanh activation is used in the generator
preprocess_tanh = Rescaling(1.0 / 127.5, -1)

conv_config = {
    "padding": "same",
    "strides": 2,
    "kernel_size": 5,
    "use_bias": use_bias,
}


def generative_model(latent_dimension: int) -> keras.Model:
    inp = Input(shape=(latent_dimension,), batch_size=batch_size)
    x = Dense(16 * 16 * 512, use_bias=use_bias)(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((16, 16, 512))(x)

    step1 = Conv2DTranspose(256, **conv_config)(x)
    step1 = BatchNormalization()(step1)
    step1 = ReLU()(step1)

    step2 = Conv2DTranspose(128, **conv_config)(step1)
    step2 = BatchNormalization()(step2)
    step2 = ReLU()(step2)

    step3 = Conv2DTranspose(64, **conv_config)(step2)
    step3 = BatchNormalization()(step3)
    step3 = ReLU()(step3)

    step4 = Conv2DTranspose(64, **conv_config)(step3)
    step4 = BatchNormalization()(step4)
    step4 = ReLU()(step4)

    output = Conv2DTranspose(3, **conv_config)(step4)
    output = Activation("tanh", dtype=tf.float32)(output)
    model = tf.keras.Model([inp], [output])

    return model


def discriminative_model() -> keras.Model:
    model = keras.Sequential(
        [
            Conv2D(64, input_shape=(image_size[0], image_size[1], 3), **conv_config),
            LeakyReLU(alpha=0.2),
            Conv2D(128, **conv_config),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(256, **conv_config),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(512, **conv_config),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(1024, **conv_config),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(1, kernel_size=16, strides=1, padding="valid"),
            Flatten(),
            Activation("sigmoid",dtype=tf.float32),
        ]
    )

    return model


generator = generative_model(latent_dim)
discriminator = discriminative_model()
discriminator.summary()

ce_loss = keras.losses.BinaryCrossentropy(from_logits=False)


# Recommended learning rate and beta_1 values from the DCGAN paper
optimizer_gen = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5,beta_2=0.999)
optimizer_disc = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5,beta_2=0.999)

optimizer_disc = keras.mixed_precision.LossScaleOptimizer(optimizer_disc)
optimizer_gen = keras.mixed_precision.LossScaleOptimizer(optimizer_gen)

generator_mean = keras.metrics.Mean()
discriminator_mean = keras.metrics.Mean()


@tf.function(jit_compile=True)
def train(images):
    image_count = images.shape[0]


    # Train the discriminator with real images
    with tf.GradientTape() as disc_tape_real:
        rand_noise = tf.random.uniform(shape=(image_count * 1, latent_dim), minval=-1, maxval=1)
        fake_images = generator(rand_noise, training=True)

        # Get discriminator predictions for real images
        predictions_real = discriminator(images, training=True)

        # Get discriminator predictions for fake images
        predictions_fake = discriminator(fake_images, training=True)

        # Calculate losses
        disc_loss_real = ce_loss(tf.ones((image_count, 1)), predictions_real)

        # Calculate losses
        disc_loss_fake = ce_loss(tf.zeros((image_count, 1)), predictions_fake)

        disc_loss_real = optimizer_disc.get_scaled_loss(disc_loss_real)
        disc_loss_fake = optimizer_disc.get_scaled_loss(disc_loss_fake)

        loss = disc_loss_real + disc_loss_fake

    discriminator_mean.update_state(loss)

    # Get the gradients w.r.t the discriminator loss
    gradients_disc = disc_tape_real.gradient(loss, discriminator.trainable_variables)
    gradients_disc = optimizer_disc.get_unscaled_gradients(gradients_disc)

    optimizer_disc.apply_gradients(
        zip(gradients_disc, discriminator.trainable_variables)
    )

    # Train the generator 3 times for every discriminator training
    for _ in range(1):
        # Train the generator
        with tf.GradientTape() as gen_tape:
            rand_noise = tf.random.uniform(shape=(image_count * 1, latent_dim), minval=-1, maxval=1)
            # Generate fake images using the generator
            fake_images = generator(rand_noise, training=True)
            fake_predictions = discriminator(fake_images, training=True)
            labels = tf.ones_like(fake_predictions)
            gen_loss = ce_loss(labels, fake_predictions)
            gen_loss = optimizer_gen.get_scaled_loss(gen_loss)

        generator_mean.update_state(gen_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_generator = optimizer_gen.get_unscaled_gradients(gradients_of_generator)
        # Apply the gradients to the optimizer
        optimizer_gen.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )


def train_epoch(epochs: int, dataset: tf.data.Dataset):
    samples = np.random.uniform(-1, 1, size=(5, latent_dim))
    for epoch in range(epochs):
        for batch_index, batch in tqdm(enumerate(dataset)):
            train(batch)

        print(
            f"Epoch: {epoch + 1} Generator Loss: {generator_mean.result():.4f} Discriminator Loss: {discriminator_mean.result():.4f}"
        )
        if epoch % 5 == 0:
            generated_images = generator.predict(samples)
            generated_images = (generated_images + 1) / 2
            # Create a figure and an array of subplots
            fig, axes = plt.subplots(1, len(generated_images))
            fig.set_size_inches(10, 10)

            for i, image in enumerate(generated_images):
                axes[i].imshow(image)  # image.squeeze(), cmap='gray
                axes[i].axis("off")

            # Save the figure to a file
            plt.savefig(f"gan_images/generated_images_{epoch}.png")
            plt.show()


# dataset_url = "https://www.kaggle.com/datasets/kostastokis/simpsons-faces"  # 'https://www.kaggle.com/datasets/bryanb/abstract-art-gallery' #'https://www.kaggle.com/datasets/robgonsalves/ganfolk'  #  https://www.kaggle.com/splcher/animefacedataset
# od.download(dataset_url)
#
# data_dir = "simpsons-faces/cropped"
#
# simpsons_dataset = (
#     tf.keras.utils.image_dataset_from_directory(
#         "/mnt/c/Users/mnusr/Downloads/imagen/image_res_enchance/images",
#         label_mode=None,
#         batch_size=batch_size,
#         image_size=image_size,
#         shuffle=True,
#     )
#     .map(preprocess_tanh)
#     #.cache()
#     .prefetch(tf.data.AUTOTUNE)
# )
#
# train_epoch(100, simpsons_dataset)
