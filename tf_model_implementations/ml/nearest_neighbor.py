import tensorflow as tf

from .base import BaseModel


class NearestNeighbor(BaseModel):
    def __init__(self, feature_extractor: tf.keras.Model):
        super().__init__(feature_extractor)
        self.labels = None
        self.features = None

    def train(self, dataset: tf.data.Dataset):
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y)
        self.features = self.feature_extractor.predict(x)
        self.labels = y

    def predict(self, dataset: tf.data.Dataset):
        feature_vector = self.feature_extractor.predict(dataset)

        # Get the distance between the feature vector and the training features
        distances = tf.reduce_sum(tf.square(feature_vector - self.features), axis=1)

        # Get the index of the minimum distance
        min_index = tf.argmin(distances, axis=1)

        # Get the label of the minimum distance
        labels = tf.gather(self.labels, min_index)

        return labels


class KNearestNeighbor(NearestNeighbor):
    def __init__(self, feature_extractor: tf.keras.Model, k: int):
        super().__init__(feature_extractor)
        self.k = k

    def predict(self, dataset: tf.data.Dataset):
        feature_vector = self.feature_extractor.predict(dataset)

        # Get the distance between the feature vector and the training features
        distances = tf.reduce_sum(tf.square(feature_vector - self.features), axis=1)

        # Get the index of the minimum distance
        min_index = tf.argsort(distances, axis=1)[:, : self.k]

        # Get the label of the minimum distance
        labels = tf.gather(self.labels, min_index)

        # Count the number of occurrences of each label
        labels = tf.reshape(labels, (-1, self.k))
        labels, _, counts = tf.unique_with_counts(tf.reshape(labels, (-1,)))

        # Get the most common label
        labels = tf.reshape(labels, (-1, self.k))
        labels = tf.gather(labels, tf.argmax(counts))

        return labels
