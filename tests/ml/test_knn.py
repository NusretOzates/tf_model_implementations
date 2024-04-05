"""
Testing the KNearestNeighbor and NearestNeighbor classes,
should create mocks for the VGG16 model and the cifar10 dataset
"""

from keras import ops
from keras.applications import VGG16
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score

from tf_model_implementations.ml.nearest_neighbor import NearestNeighbor, tf, KNearestNeighbor


def test_cifar10_nn():
    dataset = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = dataset
    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    test = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = VGG16(include_top=False, input_shape=(32, 32, 3))

    knn = NearestNeighbor(model)

    knn.train(train)

    predictions = knn.predict(test)

    labels = test.map(lambda x, y: y).unbatch()

    labels = list(labels.as_numpy_iterator())
    predictions = list(ops.convert_to_numpy(predictions))

    accuracy = accuracy_score(labels, predictions)

    assert accuracy > 0.1


def test_cifar10_knn():
    dataset = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = dataset
    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    test = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = VGG16(include_top=False, input_shape=(32, 32, 3))

    knn = KNearestNeighbor(model,3)

    knn.train(train)

    predictions = knn.predict(test)

    labels = test.map(lambda x, y: y).unbatch()

    labels = list(labels.as_numpy_iterator())
    predictions = list(ops.convert_to_numpy(predictions))

    accuracy = accuracy_score(labels, predictions)

    assert accuracy > 0.1
