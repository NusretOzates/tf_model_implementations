from keras.layers import Concatenate, Conv2D, Conv2DTranspose, MaxPool2D
from tensorflow import keras


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
        self.final = Conv2D(num_classes, 3, padding="same", activation=activation)

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

        return result
