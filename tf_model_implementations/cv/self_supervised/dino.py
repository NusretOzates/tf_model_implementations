from keras.layers import (
    Normalization,
    RandomCrop,
    RandomFlip,
    RandomContrast,
    RandomBrightness,
)
from keras import Sequential, Model
import tensorflow as tf


class DinoAugmenter(Model):
    def __init__(
        self,
        contrast: float = 0.4,
        brightness: float = 0.4,
        global_crop_size: int = 1024,
        local_crop_size: int = 512,
        local_augmentation_count=6,
        mean: list[int] = None,
        variance: list[int] = None,
    ):
        super().__init__()

        if not mean:
            mean = [0.485, 0.456, 0.406]
        if not variance:
            variance = [0.229, 0.224, 0.225]

        self.local_augmentation_model = Sequential(
            [
                RandomCrop(local_crop_size, local_crop_size),
                RandomFlip(),
                RandomContrast(contrast),
                RandomBrightness(brightness),
                Normalization(mean=mean, variance=variance),
            ]
        )

        self.global_augmentation_model = Sequential(
            [
                RandomCrop(global_crop_size, global_crop_size),
                RandomFlip(),
                RandomContrast(contrast),
                RandomBrightness(brightness),
                Normalization(mean=mean, variance=variance),
            ]
        )

        self.local_augmentation_count = local_augmentation_count

    def call(self, inputs, training=None, mask=None):

        first_globals = self.global_augmentation_model(inputs)
        second_globals = self.global_augmentation_model(inputs)

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

    student_first_globals = tf.nn.softmax(
        student_first_globals / student_temperature, axis=1
    )
    student_second_globals = tf.nn.softmax(
        student_second_globals / student_temperature, axis=1
    )
    local_projections = [
        tf.nn.softmax(local / student_temperature, axis=1)
        for local in local_projections
    ]

    teacher_first_globals = tf.nn.softmax(
        (teacher_first_globals - center) / teacher_temperature, axis=1
    )
    teacher_second_globals = tf.nn.softmax(
        (teacher_second_globals - center) / teacher_temperature, axis=1
    )

    loss = 0
    loss += tf.reduce_mean(
        tf.reduce_sum(
            -teacher_first_globals * tf.math.log(student_first_globals), axis=1
        )
    )
    loss += tf.reduce_mean(
        tf.reduce_sum(
            -teacher_second_globals * tf.math.log(student_second_globals), axis=1
        )
    )

    for local in local_projections:
        loss += tf.reduce_mean(
            tf.reduce_sum(teacher_second_globals * tf.math.log(local), axis=1)
        )

    return loss


class Dino(Model):
    def __init__(
        self,
        augmenter: DinoAugmenter,
        student_encoder: Model,
        teacher_encoder: Model,
        projector: Model,
        network_momentum: float = 0.9,
        center_momentum: float = 0.9,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
    ):
        super().__init__()

        self.augmenter = augmenter
        self.projector = projector
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.center = tf.zeros((1, 512))
        self.network_momentum = network_momentum
        self.center_momentum = center_momentum
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature

        """
        
        # gs, gt: student and teacher networks
        # C: center (K)
        # tps, tpt: student and teacher temperatures
        # l, m: network and center momentum rates
        
        gt.params = l*gt.params + (1-l)*gs.params
        C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)
        
        """

    def train_step(self, data):

        first_globals, second_globals, local_projections = self.augmenter(data)

        teacher_first_globals = self.projector(
            self.teacher_encoder(first_globals, training=True), training=True
        )
        teacher_second_globals = self.projector(
            self.teacher_encoder(second_globals, training=True), training=True
        )

        with tf.GradientTape() as tape:
            student_first_globals = self.projector(
                self.student_encoder(first_globals, training=True), training=True
            )
            student_second_globals = self.projector(
                self.student_encoder(second_globals, training=True), training=True
            )

            local_projections = [
                self.projector(
                    self.student_encoder(local, training=True), training=True
                )
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
            self.center_momentum * self.center
            + (1 - self.center_momentum)
            * tf.concat([teacher_first_globals, teacher_second_globals], axis=0).mean(
                axis=0
            )
        )

        return {"loss": loss}

    def test_step(self, data):

        first_globals, second_globals, local_projections = self.augmenter(data)

        teacher_first_globals = self.projector(self.teacher_encoder(first_globals))
        teacher_second_globals = self.projector(self.teacher_encoder(second_globals))

        student_first_globals = self.projector(self.student_encoder(first_globals))
        student_second_globals = self.projector(self.student_encoder(second_globals))

        local_projections = [
            self.projector(self.student_encoder(local)) for local in local_projections
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

        return {"loss": loss}


def create_dino_model(
    student_encoder: Model,
    teacher_encoder: Model,
    projector: Model,
    contrast: float = 0.4,
    brightness: float = 0.4,
    global_crop_size: int = 256,
    local_crop_size: int = 128,
    local_augmentation_count=6,
    mean: list[int] = None,
    variance: list[int] = None,
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
        mean=mean,
        variance=variance,
    )

    return Dino(
        augmenter,
        student_encoder,
        teacher_encoder,
        projector,
        network_momentum=network_momentum,
        center_momentum=center_momentum,
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )


from tf_model_implementations.cv.resnet_v1 import ResNet
from keras import layers

teacher = ResNet(
    rescale=True,
    input_shape=(None, None, 3),
    batch_count=1,
    activations="relu",
)

student = ResNet(
    rescale=True,
    input_shape=(None, None, 3),
    batch_count=1,
    activations="relu",
)

projector = Sequential(
    [layers.Dense(512, activation="relu"), layers.Dense(512, activation="relu")]
)

model = create_dino_model(
    student,
    teacher,
    projector,
)

model.compile(optimizer="adam", run_eagerly=True)

from tensorflow_datasets import load

dataset = load("cats_vs_dogs")
train = dataset["train"]

train = train.map(lambda x: x["image"]).batch(1).prefetch(tf.data.AUTOTUNE)

model.fit(
    train,
    epochs=10,
)
