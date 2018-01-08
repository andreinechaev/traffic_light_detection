from lenet import *

class MobileNet:

    def __init__(self, ch=1, n_classes=0, epochs=0, rate=0.001, batch=32, name='lenet'):
        self._channels = ch
        self._n_classes = n_classes
        self._epochs = epochs
        self._rate = rate
        self._batch = batch
        self._name = name


    def logits(self, x):
        tf.reshape(x, (-1, 224, 224, self._channels))

        # Layer 1:Input = 224x224x3. Output = 112x112x32.
        conv1, conv1_W, conv1_b = cnn_layer(
            features=x,
            shape=(3, 3, self._channels, 32),
            strides=[1, 1, 1, 1],
            ksize=[1, 2, 2, 1]
        )

        fc0 = cnn_flatten_layer(
            features=conv1,
            shape=(3, 3, 32, 32),
            strides=[1, 1, 1, 1],
            ksize=[1, 2, 2, 1]
        )

        # Layer 1:Input = 224x224x3. Output = 112x112x32.
        fc1 = cnn_connected(
            fc=fc0,
            shape=(64, 32),
            zeros=120
        )

        fc2 = cnn_connected(
            fc=fc0,
            shape=(64, 32),
            zeros=120
        )




    # def train(self, images, labels, valid_images, valid_labels, x, y):

