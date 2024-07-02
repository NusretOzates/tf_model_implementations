import keras
from keras.src.layers import Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, GroupNormalization, Activation, Layer, Input

keras.mixed_precision.set_global_policy("mixed_float16")


class EncoderBlock(Layer):

    def __init__(self, filters: int, group_norm_depth: int, activation_func: str, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.normalization_1 = GroupNormalization(group_norm_depth)
        self.activation_1 = Activation(activation_func)
        self.conv_2 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.normalization_2 = GroupNormalization(group_norm_depth)
        self.activation_2 = Activation(activation_func)
        self.max_pool = MaxPooling2D()

        self.filters = filters
        self.group_norm_depth = group_norm_depth
        self.activation_func = activation_func

    def call(self, inputs):
        result = self.conv_1(inputs)
        result = self.normalization_1(result)
        result = self.activation_1(result)
        result = self.conv_2(result)
        result = self.normalization_2(result)
        residual = self.activation_2(result)

        result = self.max_pool(result)

        return residual, result

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'group_norm_depth': self.group_norm_depth,
            'activation_func': self.activation_func
        })
        return config

class DecoderBlock(Layer):

    def __init__(self, filters: int, group_norm_depth: int, activation_func: str, **kwargs):
        super().__init__(**kwargs)
        self.conv_t = Conv2DTranspose(filters, 3, strides=2, padding='same')
        self.normalization_0 = GroupNormalization(group_norm_depth)
        self.activation_0 = Activation(activation_func)
        self.conv_1 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.normalization_1 = GroupNormalization(group_norm_depth)
        self.activation_1 = Activation(activation_func)
        self.conv_2 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.normalization_2 = GroupNormalization(group_norm_depth)
        self.activation_2 = Activation(activation_func)

        self.concat = Concatenate(axis=3)

        self.filters = filters
        self.group_norm_depth = group_norm_depth
        self.activation_func = activation_func

    def call(self, inputs, *args, **kwargs):
        residual = inputs['residual']
        previous_layer = inputs['previous']

        result = self.conv_t(previous_layer)
        result = self.normalization_0(result)
        result = self.activation_0(result)
        result = self.concat([result, residual])

        result = self.conv_1(result)
        result = self.normalization_1(result)
        result = self.activation_1(result)
        result = self.conv_2(result)
        result = self.normalization_2(result)
        result = self.activation_2(result)

        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'group_norm_depth': self.group_norm_depth,
            'activation_func': self.activation_func
        })
        return config


def unet(depth: int, num_classes: int, group_norm_depth: int = 32, activation_func: str = 'relu'):
    i = 64
    inputs = Input(shape=(480, 640, 3))
    result = inputs
    encoder_results = []

    for d in range(depth):
        residual, result = EncoderBlock(i, name=f'encoder_{d}', group_norm_depth=group_norm_depth, activation_func=activation_func)(result)
        encoder_results.append(residual)
        i *= 2

    # Apply the bottleneck
    result = Conv2D(i, kernel_size=(3, 3), padding='same')(result)
    result = GroupNormalization(group_norm_depth)(result)
    result = Activation(activation_func)(result)
    result = Conv2D(i, kernel_size=(3, 3), padding='same')(result)
    result = GroupNormalization(group_norm_depth)(result)
    result = Activation(activation_func)(result)

    # We will give the results in reversed order.
    encoder_results.reverse()

    # Decoder and encoder must have the same depth
    i //= 2

    for d in range(depth):
        decoder_inputs = {
            'residual': encoder_results[d],
            'previous': result
        }

        result = DecoderBlock(i, name=f'decoder_{d}', group_norm_depth=group_norm_depth, activation_func=activation_func )(decoder_inputs)

        i //= 2

    result = Conv2D(num_classes, 1, padding='same', )(result)
    result = Activation('sigmoid', dtype='float32')(result)

    return keras.models.Model(inputs=inputs, outputs=result)


# model = unet(4, 1)
# model.compile(
#     optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
# )
#
# # Download a segmentation dataset
# import tensorflow_datasets as tfds
# import tensorflow as tf
#
# dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
# train = dataset["train"]
# test = dataset["test"]
#
#
# # Prepare the dataset
# def prepare_data(data):
#     def prepare_sample(sample):
#         image = tf.cast(sample["image"], tf.float32) / 255.0
#         mask = sample["segmentation_mask"]
#         mask -= 1
#         # Resize to (480, 640)
#         image = tf.image.resize(image, (480, 640))
#         mask = tf.image.resize(mask, (480, 640))
#         return image, mask
#
#     return data.map(prepare_sample).prefetch(tf.data.AUTOTUNE)
#
#
# train = prepare_data(train).batch(1)
# test = prepare_data(test).batch(1)
#
# # Train the model
# model.fit(train, epochs=5, validation_data=test)
