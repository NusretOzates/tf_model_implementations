import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
# tf.get_logger().setLevel(logging.ERROR)

# Allow memory growth for the GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from keras_cv.layers import (
    RandomHue,
    RandomSaturation,
    RandomSharpness,
    RandomContrast,
    RandomFlip,
    Solarization,
    JitteredResize,
    RandomColorJitter,
    RandomAugmentationPipeline
)
from keras_core.ops import log_softmax, softmax, sum, mean
from keras import Sequential, Model, layers

from tf_model_implementations.cv.resnet_v1 import ResNet

# Activate mixed precision
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

BATCH_SIZE = 2

class DinoAugmenter(Model):
    def __init__(
            self,
            global_crop_size: int = 1024,
            local_crop_size: int = 512,
            local_augmentation_count=6,
    ):
        super().__init__()

        self.global_augmentation_model_1 = Sequential(
            [
                JitteredResize((global_crop_size, global_crop_size), (0.4, 1.0)),
                RandomFlip(),
                RandomAugmentationPipeline([
                    RandomColorJitter(value_range=(-1, 1), brightness_factor=0.4, contrast_factor=0.4,
                                      saturation_factor=0.4, hue_factor=0.1)
                ], rate=0.8, augmentations_per_image=1, auto_vectorize=True)
            ]
        )

        self.global_augmentation_model_2 = Sequential(
            [
                JitteredResize((global_crop_size, global_crop_size), (0.4, 1.0)),
                RandomFlip(),
                RandomAugmentationPipeline([
                    RandomColorJitter(value_range=(-1, 1), brightness_factor=0.4, contrast_factor=0.4,
                                      saturation_factor=0.4, hue_factor=0.1)
                ], rate=0.8, augmentations_per_image=1, auto_vectorize=True),
                Solarization((-1, 1), 0.2)
            ]
        )

        self.local_augmentation_model = Sequential(
            [
                JitteredResize((local_crop_size, local_crop_size), (0.05, 0.4), interpolation="bicubic"),
                RandomFlip(),
                RandomAugmentationPipeline([
                    RandomColorJitter(value_range=(-1, 1), brightness_factor=0.4, contrast_factor=0.4,
                                      saturation_factor=0.4, hue_factor=0.1)
                ], rate=0.8, augmentations_per_image=1, auto_vectorize=True)
            ]
        )

        self.local_augmentation_count = local_augmentation_count

    def call(self, inputs, training=None, mask=None):
        first_globals = self.global_augmentation_model_1(inputs)
        second_globals = self.global_augmentation_model_2(inputs)

        locals = []
        for _ in range(self.local_augmentation_count):
            local = self.local_augmentation_model(inputs)
            locals.append(local)

        return first_globals, second_globals, locals


def dino_loss(
        student_first_globals,
        student_second_globals,
        local_projections,
        teacher_first_globals,
        teacher_second_globals,
        center,
        student_temperature,
        teacher_temperature,
):
    student_first_globals = student_first_globals / student_temperature
    student_second_globals = student_second_globals / student_temperature

    local_projections = [
        local / student_temperature
        for local in local_projections
    ]

    teacher_first_globals = softmax(
        (teacher_first_globals - center) / teacher_temperature, axis=-1
    )
    teacher_second_globals = softmax(
        (teacher_second_globals - center) / teacher_temperature, axis=-1
    )

    loss = 0
    loss += mean(
        sum(
            -teacher_first_globals * log_softmax(student_second_globals, axis=-1), axis=-1
        )
    )

    loss += mean(
        sum(
            -teacher_second_globals * log_softmax(student_first_globals, axis=-1), axis=-1
        )
    )

    for local in local_projections:
        loss += mean(
            sum(-teacher_second_globals * log_softmax(local, axis=-1), axis=-1)
        )

        loss += mean(
            sum(-teacher_first_globals * log_softmax(local, axis=-1), axis=-1)
        )

    return loss


class Dino(Model):
    def __init__(
            self,
            augmenter: DinoAugmenter,
            student_model: Model,
            teacher_model: Model,
            network_momentum: float = 0.9,
            center_momentum: float = 0.9,
            student_temperature: float = 0.1,
            teacher_temperature: float = 0.04,
    ):
        super().__init__()

        self.out_dim = 65536
        self.augmenter = augmenter
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.student_projection = Sequential(
            [layers.Dense(2048, activation="gelu"), layers.Dense(2048, activation="gelu"),
             layers.Dense(256, activation=None), layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
             layers.Dense(self.out_dim, activation=None, use_bias=False, kernel_constraint="unit_norm", dtype=tf.float32)
             ]
        )

        self.teacher_projection = Sequential(
            [layers.Dense(2048, activation="gelu"), layers.Dense(2048, activation="gelu"),
             layers.Dense(256, activation=None), layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
             layers.Dense(self.out_dim, activation=None, use_bias=False, kernel_constraint="unit_norm", dtype=tf.float32)
             ]
        )

        self.student_encoder = Sequential(
            [
                self.student_model,
                self.student_projection,
            ]
        )

        self.teacher_encoder = Sequential(
            [
                self.teacher_model,
                self.teacher_projection,
            ]
        )

        self.center = tf.Variable(tf.zeros((1, self.out_dim)), trainable=False)
        self.network_momentum = network_momentum
        self.center_momentum = center_momentum
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.gradient_accumulator = [
            tf.Variable(
                tf.zeros_like(variable), trainable=False, synchronization=tf.VariableSynchronization.ON_READ
            ) for variable in self.student_encoder.trainable_variables
        ]

        self.accumulation_step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.accumulation_steps = tf.constant(128, dtype=tf.int32, name="accumulation_steps")
        self.teacher_outputs = tf.Variable(tf.zeros((BATCH_SIZE, self.out_dim)), trainable=False)
        self.first_call = True
    def train_step(self, data):

        if self.first_call:
            self.first_call = False
            for variable in self.gradient_accumulator:
                variable.assign(tf.zeros_like(variable))
            self.teacher_outputs.assign(tf.zeros_like(self.teacher_outputs))

        self.accumulation_step_counter.assign_add(1)

        epoch = self.optimizer.iterations // 31626

        #self.student_projection.layers[-1].trainable = epoch > 0

        first_globals, second_globals, local_projections = self.augmenter(data)

        teacher_first_globals = self.teacher_encoder(first_globals, training=True)
        teacher_second_globals = self.teacher_encoder(second_globals, training=True)

        self.teacher_outputs.assign_add(teacher_first_globals + teacher_second_globals)
        with tf.GradientTape() as tape:
            student_first_globals = self.student_encoder(first_globals, training=True)

            student_second_globals = self.student_encoder(second_globals, training=True)

            local_projections_part = [
                self.student_encoder(local, training=True)
                for local in local_projections
            ]

            loss = dino_loss(
                student_first_globals,
                student_second_globals,
                local_projections_part,
                teacher_first_globals,
                teacher_second_globals,
                self.center,
                self.student_temperature,
                self.teacher_temperature,
            )

            loss /= tf.cast(self.accumulation_steps, loss.dtype)

            scaled_loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(scaled_loss, self.student_encoder.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradients = self.optimizer.get_unscaled_gradients(gradients)

        for j in range(len(self.gradient_accumulator)):
            self.gradient_accumulator[j].assign_add(gradients[j])


        tf.cond(
            tf.equal(self.accumulation_step_counter, self.accumulation_steps),
            self.apply_accumulated_gradients,
            lambda: None
        )

        return {"loss": loss}


    def apply_accumulated_gradients(self):

        self.optimizer.apply_gradients(
            zip(self.gradient_accumulator, self.student_encoder.trainable_variables)
        )
        self.accumulation_step_counter.assign(0)

        for variable in self.gradient_accumulator:
            variable.assign(tf.zeros_like(variable), read_value=False)



        # Update the teacher network
        for teacher, student in zip(
                self.teacher_encoder.trainable_variables,
                self.student_encoder.trainable_variables,
        ):
            teacher.assign(
                self.network_momentum * teacher + (1 - self.network_momentum) * student
            )

        # Update the center
        self.center.assign(
            self.center_momentum * self.center + (1 - self.center_momentum) * sum(self.teacher_outputs / BATCH_SIZE, axis=0)
        )

        self.teacher_outputs.assign(tf.zeros_like(self.teacher_outputs))

        tf.print("Gradients applied")

    # def test_step(self, data):
    #     first_globals, second_globals, local_projections = self.augmenter(data)
    #
    #     teacher_first_globals = self.projector(self.teacher_encoder(first_globals))
    #     teacher_second_globals = self.projector(self.teacher_encoder(second_globals))
    #
    #     student_first_globals = self.projector(self.student_encoder(first_globals))
    #     student_second_globals = self.projector(self.student_encoder(second_globals))
    #
    #     local_projections = [
    #         self.projector(self.student_encoder(local)) for local in local_projections
    #     ]
    #
    #     loss = dino_loss(
    #         student_first_globals,
    #         student_second_globals,
    #         local_projections,
    #         teacher_first_globals,
    #         teacher_second_globals,
    #         self.center,
    #         self.student_temperature,
    #         self.teacher_temperature,
    #     )
    #
    #     return {"loss": loss}


def create_dino_model(
        student_encoder: Model,
        teacher_encoder: Model,
        global_crop_size: int = 112,
        local_crop_size: int = 38,
        local_augmentation_count=4,
        network_momentum: float = 0.9995,
        center_momentum: float = 0.9,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
):
    augmenter = DinoAugmenter(
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        local_augmentation_count=local_augmentation_count,
    )

    return Dino(
        augmenter,
        student_encoder,
        teacher_encoder,
        network_momentum=network_momentum,
        center_momentum=center_momentum,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )


# All images under this path will be used for training
dataset_path = "/mnt/c/Users/mnusr/PycharmProjects/hcc_tests/big_image_example/processed_2"

# All images under this path will be used for testing
test_dataset_path = "/mnt/c/Users/mnusr/PycharmProjects/hcc_tests/datasets/hcc"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    label_mode=None,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(1024, 1024),
    shuffle=True,
    seed=42,
    interpolation="bicubic",
).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dataset_path,
    label_mode='binary',
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(1024, 1024),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training",
    interpolation="bicubic",
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dataset_path,
    label_mode='binary',
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(1024, 1024),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation",
    interpolation="bicubic",
).prefetch(tf.data.AUTOTUNE)

teacher = ResNet(
    rescale=True,
    input_shape=(None, None, 3),
    batch_count=BATCH_SIZE,
    activations="gelu",
)

student = ResNet(
    rescale=True,
    input_shape=(None, None, 3),
    batch_count=BATCH_SIZE,
    activations="gelu",
)

model = create_dino_model(
    student,
    teacher,
    global_crop_size=512,
    local_crop_size=256,
    local_augmentation_count=2
)

schedule = tf.keras.optimizers.schedules.CosineDecay(0.0005, 264 * 4, alpha=1e-6, warmup_target=0.0005,
                                                     warmup_steps=264 * 1)

optimizer = tf.keras.optimizers.Adam(schedule)

scaled_optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=scaled_optimizer, run_eagerly=False, jit_compile=False)

model.fit(
    train_dataset,
    epochs=2,
)

del model, teacher

student.trainable = False

model_st = Sequential(
    [
        student,
        layers.Dense(1),
        layers.Activation("sigmoid", dtype=tf.float32)
    ])

optimizer = tf.keras.optimizers.Adam(0.00001)

scaled_optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model_st.compile(optimizer=scaled_optimizer, run_eagerly=False, jit_compile=False,
                 metrics=["accuracy"],
                 loss="binary_crossentropy")

model_st.fit(
    validation_dataset,
    validation_data=test_dataset,
    epochs=10
)


# from tensorflow_datasets import load

#
# dataset = load("cats_vs_dogs")
# train = dataset["train"]
#
# train = train.map(lambda x: (tf.image.resize(x["image"], (224, 224)), x['label'])).batch(64).prefetch(tf.data.AUTOTUNE)
#
# test = train.take(100)
# train = train.skip(100)

# Train only the student with supervised learning

# student = ResNet(
#     rescale=True,
#     input_shape=(None, None, 3),
#     batch_count=8,
#     activations="gelu",
# )

# model = Sequential(
#     [
#         student,
#         layers.Dense(2, activation="softmax"),
#     ]
#
# )
#
# model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), run_eagerly=False, jit_compile=False, metrics=["accuracy"],loss="sparse_categorical_crossentropy")
#
# model.fit(
#     train,
#     validation_data=test,
#     epochs=5,
# )


# Train the student with DINO

# dataset = load("cats_vs_dogs")
# train: tf.data.Dataset = dataset["train"]
#
# train = train.map(lambda x: tf.image.resize(x["image"], (224, 224)), num_parallel_calls=tf.data.AUTOTUNE).batch(
#     BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
# test = train.take(100)
# train = train.skip(100)
#
# teacher = ResNet(
#     rescale=True,
#     input_shape=(None, None, 3),
#     batch_count=BATCH_SIZE,
#     activations="gelu",
# )
#
# student = ResNet(
#     rescale=True,
#     input_shape=(None, None, 3),
#     batch_count=BATCH_SIZE,
#     activations="gelu",
# )
#
# model = create_dino_model(
#     student,
#     teacher,
#     local_augmentation_count=2
# )
#
# schedule = tf.keras.optimizers.schedules.CosineDecay(0.00005, 264 * 4, alpha=1e-6, warmup_target=0.00005,
#                                                      warmup_steps=264 * 1)
#
# model.compile(optimizer=tf.keras.optimizers.Adam(schedule), run_eagerly=False, jit_compile=False)
#
# model.fit(
#     train,
#     epochs=20,
# )

# # Train the student with DINO and supervised learning
# dataset = load("cats_vs_dogs")
# train = dataset["train"]
#
# train = train.map(lambda x: (tf.image.resize(x["image"], (224, 224)), x['label'])).batch(16).prefetch(tf.data.AUTOTUNE)
#
# test = train.take(100)
# train = train.skip(100)
#
# student.trainable = False
#
# model = Sequential(
#     [
#         student,
#         layers.Dense(2, activation="softmax"),
#     ]
#
# )
#
# model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), run_eagerly=False, jit_compile=False, metrics=["accuracy"],
#               loss="sparse_categorical_crossentropy")
#
# model.fit(
#     train,
#     validation_data=test,
#     epochs=5,
# )
