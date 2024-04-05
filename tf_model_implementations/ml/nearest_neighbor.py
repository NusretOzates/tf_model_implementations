import tensorflow as tf
import keras
from keras import ops
import numpy as np


class NearestNeighbor:
    def __init__(self, feature_extractor: keras.Model):
        self.labels = None
        self.features = None
        self.feature_extractor = feature_extractor

    def train(self, dataset: tf.data.Dataset):
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y).unbatch()
        self.features = self.feature_extractor.predict(x).squeeze()
        # Finding a better way to do this would be nice
        self.labels = ops.array(list(y.as_numpy_iterator()))

    def predict(self, dataset: tf.data.Dataset):

        result = ops.empty((0,))

        for x,y in dataset:

            feature_vector = self.feature_extractor.predict(x).squeeze()

            # Get the distance between the feature vector and the training features
            feature_vector_sum = ops.sum(ops.square(feature_vector), axis=1)
            feature_vector_sum = ops.expand_dims(feature_vector_sum, axis=-1)

            features_sum = ops.sum(ops.square(self.features), axis=1)
            features_sum = ops.expand_dims(features_sum, axis=0)

            dot_product = ops.dot(feature_vector, self.features.transpose())
            distances = feature_vector_sum + features_sum - 2 * dot_product

            # Get the index of the minimum distance
            min_index = ops.argmin(distances, axis=1)

            # Get the label of the minimum distance, currently no better way to do this in keras
            labels = tf.gather(self.labels, min_index)

            result = ops.append(result, labels)

        return result


class KNearestNeighbor(NearestNeighbor):
    def __init__(self, feature_extractor: tf.keras.Model, k: int):
        super().__init__(feature_extractor)
        self.k = k

    def predict(self, dataset: tf.data.Dataset):

        result = ops.empty((0,))

        for x,y in dataset:

            feature_vector = self.feature_extractor.predict(x).squeeze()

            # Get the distance between the feature vector and the training features
            feature_vector_sum = ops.sum(ops.square(feature_vector), axis=1)
            feature_vector_sum = ops.expand_dims(feature_vector_sum, axis=-1)

            features_sum = ops.sum(ops.square(self.features), axis=1)
            features_sum = ops.expand_dims(features_sum, axis=0)

            dot_product = ops.dot(feature_vector, self.features.transpose())
            distances = feature_vector_sum + features_sum - 2 * dot_product

            # Get the index of the minimum distance
            min_index = ops.argsort(distances, axis=1)[:, : self.k]

            # Get the label of the minimum distance, currently no better way to do this in keras
            labels = tf.gather(self.labels, min_index)


            # Count the number of occurrences of each label. TODO: Find a way to do this in keras.
            labels = ops.reshape(labels, (-1, self.k))
            selected_labels = []
            for i in range(labels.shape[0]):
                unique, counts = np.unique(labels[i], return_counts=True)
                selected_labels.append(unique[np.argmax(counts)])

            result = ops.append(result, selected_labels)

        return result
