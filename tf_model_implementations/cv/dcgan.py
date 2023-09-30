# !pip3 install opendatasets --upgrade --quiet
# https://arxiv.org/pdf/1511.06434.pdf
import keras
import numpy as np
import opendatasets as od
import tensorflow as tf
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

batch_size = 128
image_size = (64, 64)
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
    x = Dense(4 * 4 * 1024, use_bias=use_bias)(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((4, 4, 1024))(x)

    step1 = Conv2DTranspose(512, **conv_config)(x)
    step1 = BatchNormalization()(step1)
    step1 = ReLU()(step1)

    step2 = Conv2DTranspose(256, **conv_config)(step1)
    step2 = BatchNormalization()(step2)
    step2 = ReLU()(step2)

    step3 = Conv2DTranspose(128, **conv_config)(step2)
    step3 = BatchNormalization()(step3)
    step3 = ReLU()(step3)

    output = Conv2DTranspose(3, activation="tanh", **conv_config)(step3)

    model = tf.keras.Model([inp], [output])

    return model


def discriminative_model() -> keras.Model:
    model = tf.keras.Sequential(
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
            Conv2D(1, kernel_size=4, strides=1, padding="valid"),
            Flatten(),
            Activation("sigmoid"),
        ]
    )

    return model


generator = generative_model(latent_dim)
discriminator = discriminative_model()

ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Recommended learning rate and beta_1 values from the DCGAN paper
optimizer_gen = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

generator_mean = tf.keras.metrics.Mean()
discriminator_mean = tf.keras.metrics.Mean()


@tf.function(jit_compile=True)
def train(images):
    image_count = images.shape[0]
    rand_noise = tf.random.normal(shape=(image_count, latent_dim))

    # Train the discriminator with real images
    with tf.GradientTape() as disc_tape_real:
        # Get discriminator predictions for real images
        predictions_real = discriminator(images, training=True)
        # Calculate losses
        disc_loss_real = ce_loss(tf.zeros((image_count, 1)), predictions_real)

    # Train the discriminator with fake images
    with tf.GradientTape() as disc_tape_fake:
        # Generate fake images using the generator
        fake_images = generator(rand_noise, training=True)

        # Get discriminator predictions for fake images
        predictions_fake = discriminator(fake_images, training=True)

        # Calculate losses
        disc_loss_fake = ce_loss(tf.ones((image_count, 1)), predictions_fake)

    discriminator_mean.update_state(disc_loss_real + disc_loss_fake)

    # Calculate gradients
    gradients_of_discriminator_fake = disc_tape_fake.gradient(
        disc_loss_fake, discriminator.trainable_variables
    )
    gradients_of_discriminator_real = disc_tape_real.gradient(
        disc_loss_real, discriminator.trainable_variables
    )
    total = [
        a + b
        for a, b in zip(
            gradients_of_discriminator_fake, gradients_of_discriminator_real
        )
    ]

    optimizer_disc.apply_gradients(zip(total, discriminator.trainable_variables))

    # Train the generator
    with tf.GradientTape() as gen_tape:
        rand_noise = tf.random.normal(shape=(image_count * 1, latent_dim))
        # Generate fake images using the generator
        fake_images = generator(rand_noise, training=True)
        fake_predictions = discriminator(fake_images, training=True)
        labels = tf.zeros_like(fake_predictions)
        gen_loss = ce_loss(labels, fake_predictions)

    generator_mean.update_state(gen_loss)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Apply the gradients to the optimizer
    optimizer_gen.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )


def train_epoch(epochs: int, dataset: tf.data.Dataset):
    samples = np.random.normal(size=(5, latent_dim))
    for epoch in range(epochs):
        for batch_index, batch in enumerate(dataset):
            train(batch)

        print(
            f"Epoch: {epoch + 1} Generator Loss: {generator_mean.result():.4f} Discriminator Loss: {discriminator_mean.result():.4f}"
        )
        if epoch % 15 == 0:
            generated_images = generator.predict(samples)
            generated_images = (generated_images + 1) / 2
            # Create a figure and an array of subplots
            fig, axes = plt.subplots(1, len(generated_images))
            fig.set_size_inches(10, 10)

            for i, image in enumerate(generated_images):
                axes[i].imshow(image)  # image.squeeze(), cmap='gray
                axes[i].axis("off")

            # Save the figure to a file
            plt.savefig(f"generated_images_{epoch}.png")
            plt.show()


dataset_url = "https://www.kaggle.com/datasets/kostastokis/simpsons-faces"  # 'https://www.kaggle.com/datasets/bryanb/abstract-art-gallery' #'https://www.kaggle.com/datasets/robgonsalves/ganfolk'  #  https://www.kaggle.com/splcher/animefacedataset
od.download(dataset_url)

data_dir = "simpsons-faces/cropped"

simpsons_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode=None,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
    )
    .map(preprocess_tanh)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

train_epoch(1200, simpsons_dataset)
