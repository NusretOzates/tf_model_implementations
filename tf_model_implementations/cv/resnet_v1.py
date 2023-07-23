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
def ResNet(rescale: bool, input_shape, batch_count, activations: str = "relu"):
    inputs = layers.Input(input_shape, batch_count)
    if rescale:
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)

    else:
        x = inputs

    x = layers.Conv2D(64, 7, 2, padding="same")(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation(activations)(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = resnet50(x, activations)
    outputs = layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# TODO Clean up this code and move it to the test folder
BATCH_SIZE = 8
IMAGE_SIZE = 512
base_model = ResNet(True, (IMAGE_SIZE, IMAGE_SIZE, 3), BATCH_SIZE, "elu")
model = keras.models.Sequential([base_model, layers.Dense(1, activation="sigmoid")])

model.compile(
    tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.F1Score(average="weighted"), "accuracy"],
    run_eagerly=False,
)


train = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\mnusr\\Documents\\hcc_data\\hcc_2048\\hcc_vs_inc\\train",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    label_mode="binary",
    class_names=["hcc", "inc"],
    shuffle=True,
    seed=42,
).prefetch(tf.data.AUTOTUNE)

validation = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\mnusr\\Documents\\hcc_data\\hcc_2048\\hcc_vs_inc\\val",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    label_mode="binary",
    class_names=["hcc", "inc"],
    shuffle=False,
)

test = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\mnusr\\Documents\\hcc_data\\hcc_2048\\hcc_vs_inc\\test",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    label_mode="binary",
    class_names=["hcc", "inc"],
    shuffle=False,
)

model.fit(train, epochs=5, validation_data=validation, class_weight={0: 0.60, 1: 0.40})

model.evaluate(test)
