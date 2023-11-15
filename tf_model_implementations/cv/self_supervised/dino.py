from keras.layers import (
    Normalization,
    RandomCrop,
    RandomFlip,
    RandomContrast,
    RandomBrightness
)

from keras_cv.layers import (
RandomHue,
RandomSaturation,
RandomSharpness,
Solarization,
JitteredResize,
RandomColorJitter,
RandomAugmentationPipeline
)


from keras import Sequential, Model,layers
import tensorflow as tf
from tf_model_implementations.cv.resnet_v1 import ResNet

class DinoAugmenter(Model):
    def __init__(
            self,
            contrast: float = 0.4,
            brightness: float = 0.4,
            global_crop_size: int = 1024,
            local_crop_size: int = 512,
            local_augmentation_count=6,
    ):
        super().__init__()


        self.global_augmentation_model_1 = Sequential(
            [
                JitteredResize((global_crop_size,global_crop_size), (0.4, 1.0)),
                RandomFlip(),
                RandomAugmentationPipeline([
                    RandomColorJitter(value_range=(-1,1),brightness_factor=0.4, contrast_factor=0.4, saturation_factor=0.4, hue_factor=0.1)
                ], rate=0.8,augmentations_per_image=1,auto_vectorize=True)
            ]
        )


        self.global_augmentation_model_2 = Sequential(
            [
                JitteredResize((global_crop_size,global_crop_size), (0.4, 1.0)),
                RandomFlip(),
                RandomAugmentationPipeline([
                    RandomColorJitter(value_range=(-1,1),brightness_factor=0.4, contrast_factor=0.4, saturation_factor=0.4, hue_factor=0.1)
                ], rate=0.8,augmentations_per_image=1,auto_vectorize=True),
                Solarization((-1,1), 0.2)
            ]
        )


        self.local_augmentation_model = Sequential(
            [
                JitteredResize((local_crop_size,local_crop_size), (0.05, 0.4),interpolation="bicubic"),
                RandomFlip(),
                RandomAugmentationPipeline([
                    RandomColorJitter(value_range=(-1,1),brightness_factor=0.4, contrast_factor=0.4, saturation_factor=0.4, hue_factor=0.1)
                ], rate=0.8,augmentations_per_image=1,auto_vectorize=True)
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

    teacher_first_globals = tf.nn.softmax(
        (teacher_first_globals - center) / teacher_temperature, axis=-1
    )
    teacher_second_globals = tf.nn.softmax(
        (teacher_second_globals - center) / teacher_temperature, axis=-1
    )

    loss = 0
    loss += tf.reduce_mean(
        tf.reduce_sum(
            -teacher_first_globals * tf.nn.log_softmax(student_second_globals, axis=-1), axis=-1
        )
    )

    loss += tf.reduce_mean(
        tf.reduce_sum(
            -teacher_second_globals * tf.nn.log_softmax(student_first_globals, axis=-1), axis=-1
        )
    )

    for local in local_projections:
        loss += tf.reduce_mean(
            tf.reduce_sum(-teacher_second_globals * tf.nn.log_softmax(local, axis=-1), axis=-1)
        )

        loss += tf.reduce_mean(
            tf.reduce_sum(-teacher_first_globals * tf.nn.log_softmax(local,axis=-1), axis=-1)
        )

    return loss


class Dino(Model):
    def __init__(
            self,
            augmenter: DinoAugmenter,
            student_encoder: Model,
            teacher_encoder: Model,
            network_momentum: float = 0.9,
            center_momentum: float = 0.9,
            student_temperature: float = 0.1,
            teacher_temperature: float = 0.04,
    ):
        super().__init__()

        self.augmenter = augmenter
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.center = tf.Variable(tf.zeros((1,2048*8)), trainable=False)
        self.network_momentum = network_momentum
        self.center_momentum = center_momentum
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature


    def train_step(self, data):
        first_globals, second_globals, local_projections = self.augmenter(data)

        teacher_first_globals = self.teacher_encoder(first_globals, training=True)
        teacher_second_globals = self.teacher_encoder(second_globals, training=True)

        with tf.GradientTape() as tape:
            student_first_globals = self.student_encoder(first_globals, training=True)

            student_second_globals = self.student_encoder(second_globals, training=True)

            local_projections = [
                    self.student_encoder(local, training=True)
                for local in local_projections
            ]

            loss = dino_loss(
                student_first_globals,
                student_second_globals,
                local_projections,
                teacher_first_globals,
                teacher_second_globals,
                self.center,
                self.student_temperature,
                self.teacher_temperature,
            )

        gradients = tape.gradient(loss, self.student_encoder.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student_encoder.trainable_variables)
        )

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
            self.center_momentum * self.center + (1 - self.center_momentum) * tf.reduce_mean(tf.concat([teacher_first_globals, teacher_second_globals],axis=0), axis=0)
        )
        return {"loss": loss}

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
        contrast: float = 0.4,
        brightness: float = 0.4,
        global_crop_size: int = 112,
        local_crop_size: int = 38,
        local_augmentation_count=4,
        network_momentum: float = 0.9,
        center_momentum: float = 0.9,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
):
    augmenter = DinoAugmenter(
        contrast=contrast,
        brightness=brightness,
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


from tensorflow_datasets import load

dataset = load("cats_vs_dogs")
train = dataset["train"]

train = train.map(lambda x: (tf.image.resize(x["image"], (224, 224)),x['label'])).batch(16).prefetch(tf.data.AUTOTUNE)

test = train.take(100)
train = train.skip(100)



# Train only the student with supervised learning

student = ResNet(
    rescale=True,
    input_shape=(None, None, 3),
    batch_count=8,
    activations="gelu",
)

projector_st = Sequential(
    [layers.Dense(2048, activation="gelu"), layers.Dense(2048, activation="gelu"),
     layers.Dense(256, activation=None), layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
     layers.Dense(2048*8,activation=None,use_bias=False,kernel_constraint="unit_norm")
     ]
)

student = Sequential(
    [
        student,
        projector_st,
    ]
)




model = Sequential(
    [
        student,
        layers.Dense(2, activation="softmax"),
    ]

)

model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), run_eagerly=False, jit_compile=False, metrics=["accuracy"],loss="sparse_categorical_crossentropy")

model.fit(
    train,
    validation_data=test,
    epochs=5,
)



# Train the student with DINO

# dataset = load("cats_vs_dogs")
# train = dataset["train"]
#
# train = train.map(lambda x: tf.image.resize(x["image"], (224, 224))).batch(16).prefetch(tf.data.AUTOTUNE)
#
# test = train.take(100)
# train = train.skip(100)
#
# teacher = ResNet(
#     rescale=True,
#     input_shape=(None, None, 3),
#     batch_count=16,
#     activations="gelu",
# )
#
# projector = Sequential(
#     [layers.Dense(2048, activation="gelu"), layers.Dense(2048, activation="gelu"),
#      layers.Dense(256, activation=None), layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
#      layers.Dense(2048*8,activation=None,use_bias=False,kernel_constraint="unit_norm")
#      ]
# )
#
# teacher = Sequential(
#     [
#         teacher,
#         projector,
#     ]
# )
#
#
# student = ResNet(
#     rescale=True,
#     input_shape=(None, None, 3),
#     batch_count=16,
#     activations="gelu",
# )
#
# projector_st = Sequential(
#     [layers.Dense(2048, activation="gelu"), layers.Dense(2048, activation="gelu"),
#      layers.Dense(256, activation=None), layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
#      layers.Dense(2048*8,activation=None,use_bias=False,kernel_constraint="unit_norm")
#      ]
# )
#
# student = Sequential(
#     [
#         student,
#         projector_st,
#     ]
# )
#
#
# model = create_dino_model(
#     student,
#     teacher,
# )
#
# model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), run_eagerly=False, jit_compile=False)
#
# model.fit(
#     train,
#     epochs=20,
# )
#
#
# # Train the student with DINO and supervised learning
# dataset = load("cats_vs_dogs")
# train = dataset["train"]
#
# train = train.map(lambda x: (tf.image.resize(x["image"], (224, 224)),x['label'])).batch(8).prefetch(tf.data.AUTOTUNE)
#
# test = train.take(100)
# train = train.skip(100)
#
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

