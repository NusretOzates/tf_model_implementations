from keras.layers import Concatenate, Conv2D, Conv2DTranspose, MaxPool2D
from tensorflow import keras
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

class EncoderBlock(keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")
        self.conv_2 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")
        self.max_pool = MaxPool2D()

    def call(self, inputs, *args, **kwargs):
        residual = self.conv_1(inputs)
        residual = self.conv_2(residual)
        result = self.max_pool(residual)

        return residual, result


class DecoderBlock(keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")
        self.conv_2 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")
        self.conv_t = Conv2DTranspose(filters, 2, strides=2, padding="same")
        self.concat = Concatenate(axis=3)

    def call(self, inputs, *args, **kwargs):
        residual = inputs["residual"]
        previous_layer = inputs["previous"]


        result = self.conv_t(previous_layer)

        # Make sure that the residual and the previous layer have the same shape
        residual = tf.image.resize(residual, tf.shape(result)[1:3])

        result = self.concat([result, residual])
        result = self.conv_1(result)
        result = self.conv_2(result)

        return result


class Unet(keras.models.Model):
    def __init__(self, depth: int, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = []
        self.decoders = []
        i = 64
        while depth > 0:
            self.encoders.append(EncoderBlock(i))
            self.decoders.insert(0, DecoderBlock(i))
            i *= 2
            depth -= 1

        # Bottleneck
        self.conv_1 = Conv2D(i, kernel_size=3, padding="same", activation="relu")
        self.conv_2 = Conv2D(i, kernel_size=3, padding="same", activation="relu")

        activation = "sigmoid" if num_classes == 1 else "softmax"
        self.final = Conv2D(num_classes, 3, padding="same")
        self.activation = keras.layers.Activation(activation,dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        result = inputs
        encoder_results = []
        for encoder in self.encoders:
            residual, result = encoder(result)
            encoder_results.append(residual)

        # Apply the bottleneck
        result = self.conv_1(result)
        result = self.conv_2(result)

        for encoder_result, decoder in zip(reversed(encoder_results), self.decoders):
            decoder_inputs = {"residual": encoder_result, "previous": result}
            result = decoder(decoder_inputs)

        result = self.final(result)

        # Make sure that the result is in same shape as the input
        result = tf.image.resize(result, tf.shape(inputs)[1:3])

        result = self.activation(result)

        return result


model = Unet(4, 1)
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=False
)


# Download a segmentation dataset
import tensorflow_datasets as tfds

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
train = dataset["train"]
test = dataset["test"]

# Prepare the dataset
def prepare_data(data):
    def prepare_sample(sample):
        image = tf.cast(sample["image"], tf.float32) / 255.0
        mask = sample["segmentation_mask"]
        mask -= 1
        return image, mask

    return data.map(prepare_sample).prefetch(tf.data.AUTOTUNE)


train = prepare_data(train).batch(1)
test = prepare_data(test).batch(1)

# Train the model
model.fit(train, epochs=5, validation_data=test)
