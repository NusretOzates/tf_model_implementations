"""

ResNet V1 model. It uses 1x1 3x3 1x1 convolutions for the shortcut connections.
The model is based on the paper: https://arxiv.org/pdf/1512.03385.pdf

It consists of stacked residual blocks.
Each residual block has two 3x3 convolutions with batch normalization and ReLU.
"""

import tensorflow as tf
from keras import layers
from tensorflow import keras


def batch_normalized_conv2d(
        x: tf.Tensor,
        filter_size: int,
        kernel_size: int,
        stride_size: int,
        activation: str,
        padding: str = "valid",
):
    x = layers.Conv2D(filter_size, kernel_size, stride_size, padding)(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation(activation)(x)
    return x


def residual_block(
        x: tf.Tensor,
        filter_size: int,
        stride_size: int,
        activation: str,
        is_skip_connection: bool = False,
):
    skip_connection = x
    if is_skip_connection:
        skip_connection = layers.Conv2D(filter_size * 4, 1, stride_size)(x)
        skip_connection = layers.BatchNormalization(epsilon=1.001e-5)(skip_connection)

    x = batch_normalized_conv2d(
        x, filter_size, kernel_size=1, stride_size=stride_size, activation=activation
    )
    x = batch_normalized_conv2d(
        x,
        filter_size,
        kernel_size=3,
        stride_size=1,
        padding="same",
        activation=activation,
    )

    x = layers.Conv2D(filter_size * 4, 1, 1)(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Add()([skip_connection, x])
    x = layers.Activation(activation)(x)
    return x


def residual_stack(
        x: tf.Tensor, filter_size: int, block_count: int, stride_size: int, activation: str
):
    x = residual_block(
        x=x,
        filter_size=filter_size,
        stride_size=stride_size,
        activation=activation,
        is_skip_connection=True,
    )

    for i in range(2, block_count + 1):
        x = residual_block(x, filter_size, 1, activation)

    return x


def resnet50(x: tf.Tensor, activation: str):
    x = residual_stack(x, 64, 3, 1, activation)
    x = residual_stack(x, 128, 4, 2, activation)
    x = residual_stack(x, 256, 6, 2, activation)
    x = residual_stack(x, 512, 3, 2, activation)

    return x


# TODO: Make it more generic and add more options such as resnet50, resnet101, etc.
def ResNet(rescale: bool, input_shape, batch_count, activations: str = "relu", pooling='max'):
    inputs = layers.Input(input_shape, batch_count)
    if rescale:
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)

    else:
        x = inputs

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = layers.Conv2D(64, 7, 2)(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation(activations)(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = resnet50(x, activations)
    if not pooling or pooling == 'max':
        outputs = layers.GlobalMaxPooling2D()(x)
    elif pooling == 'avg':
        outputs = layers.GlobalAveragePooling2D()(x)
    else:
        outputs = layers.GlobalMaxPooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# import numpy as np
# from tensorflow import keras
# from keras import layers
#
# # Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)
#
# # Load the data and split it between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")
#
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# batch_size = 128
# epochs = 15
# model = keras.Sequential(
#     [
#         ResNet(
#             rescale=False,
#             input_shape=input_shape,
#             batch_count=batch_size,
#             activations="relu",
#         ),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )
#
# model.summary()
#
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
