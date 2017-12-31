from tensorflow.contrib.layers import flatten
import tensorflow as tf

mu = 0
sigma = 0.1


def cnn_layer(features, shape, strides, ksize, padding='VALID'):
    # Layer 1: Convolutional.
    conv_W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
    conv_b = tf.Variable(tf.zeros(6))
    conv = tf.nn.conv2d(features, conv_W, strides=strides, padding=padding) + conv_b

    # Activation.
    conv = tf.nn.relu(conv)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv = tf.nn.max_pool(conv, ksize=ksize, strides=ksize, padding=padding)

    return conv, conv_W, conv_b


def cnn_flatten_layer(features, shape, strides, ksize, padding='VALID'):
    conv_W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
    conv_b = tf.Variable(tf.zeros(16))
    conv = tf.nn.conv2d(features, conv_W, strides=strides, padding=padding) + conv_b

    # Activation.
    conv = tf.nn.relu(conv)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv = tf.nn.max_pool(conv, ksize=ksize, strides=ksize, padding=padding)

    # Flatten. Input = 5x5x16. Output = 400.
    fc = flatten(conv)
    return fc


def cnn_connected(fc, shape, zeros, activation=True):
    fc_W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
    fc_b = tf.Variable(tf.zeros(zeros))
    fc = tf.matmul(fc, fc_W) + fc_b

    if activation:
        # Activation.
        return tf.nn.relu(fc)
    else:
        return fc


class LeNet:

    def __init__(self, ch=1, n_classes=0, epochs=0, rate=0.001, batch=32, name='lenet'):
        self._channels = ch
        self._n_classes = n_classes
        self._epochs = epochs
        self._rate = rate
        self._batch = batch
        self._name = name

    def get_logits(self, x):

        tf.reshape(x, (-1, 32, 32, self._channels))

        # Layer 1:Input = 32x32x3. Output = 28x28x6.
        conv1, conv1_W, conv1_b = cnn_layer(
            features=x,
            shape=(5, 5, self._channels, 6),
            strides=[1, 1, 1, 1],
            ksize=[1, 2, 2, 1]
        )

        # Layer 2: Convolutional. Output = 10x10x16.
        fc0 = cnn_flatten_layer(
            features=conv1,
            shape=(5, 5, 6, 16),
            strides=[1, 1, 1, 1],
            ksize=[1, 2, 2, 1]
        )

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1 = cnn_connected(
            fc=fc0,
            shape=(400, 120),
            zeros=120
        )

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2 = cnn_connected(
            fc=fc1,
            shape=(120, 84),
            zeros=84
        )

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        logits = cnn_connected(
            fc=fc2,
            shape=(84, self._n_classes),
            zeros=self._n_classes,
            activation=False
        )

        return logits

    def train(self, images, labels, valid_images, valid_labels, x, y):
        from sklearn.utils import shuffle

        one_hot_y = tf.one_hot(y, self._n_classes)

        logits = self.get_logits(x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._rate)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        tf.argmax(logits, 1, name='predictions')
        tf.nn.softmax(logits, name='Softmax')

        acc_points = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(images)
            for i in range(self._epochs):
                images, labels = shuffle(images, labels)
                for offset in range(0, num_examples, self._batch):
                    end = offset + self._batch
                    batch_x, batch_y = images[offset:end], labels[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

                validation_accuracy = self.evaluate(valid_images, valid_labels, accuracy_operation, x, y)
                acc_points.append(validation_accuracy)
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            self._name = self._name + '.tf'
            saver.save(sess, './models/' + self._name)
            print('Model saved with name {}'.format(self._name))

        return logits, training_operation

    def evaluate(self, x_data, y_data, accuracy_operation, x, y):
        num_examples = len(x_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self._batch):
            batch_x, batch_y = x_data[offset:offset + self._batch], y_data[offset:offset + self._batch]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))

        return total_accuracy / num_examples
