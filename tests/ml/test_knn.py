from keras.applications import VGG16
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score

from tf_model_implementations.ml.nearest_neighbor import NearestNeighbor, tf


def test_cifar10():
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

    labels = test.map(lambda x, y: y)

    labels = list(labels.as_numpy_iterator())
    predictions = list(predictions.as_numpy_iterator())

    accuracy = accuracy_score(labels, predictions)

    assert accuracy > 0.1
