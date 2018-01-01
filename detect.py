import cv2
import os

TF_TRUE = 1
TF_FALSE = 0

rate = 0.0001
EPOCHS = 256
BATCH_SIZE = 128

images = []
labels = []


def normalize(img, beta=255):
    cv2.normalize(img, img, alpha=0, beta=beta, norm_type=cv2.NORM_MINMAX)


def image_from_file(path, shape=(32, 32)):
    img = cv2.imread(path)
    img = cv2.resize(img, shape)
    normalize(img)
    return img


def read_data(path):
    for root, subs, files in os.walk(path):
        if subs:
            for sub in subs:
                read_data(sub)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                fp = os.path.join(root, file)
                img = image_from_file(fp)
                images.append(img)
                if 'non_tf' in root:
                    labels.append(TF_FALSE)
                else:
                    labels.append(TF_TRUE)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from lenet import LeNet
    import tensorflow as tf

    n_classes = 2

    read_data('dataset')

    images, valid_images, labels, valid_labels = train_test_split(images, labels, test_size=0.20, random_state=0)
    print('Loaded {} images & {} labels'.format(len(images), len(labels)))
    print('Loaded {} validation images & {} labels'.format(len(valid_images), len(valid_labels)))

    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='x_value')
    y = tf.placeholder(tf.int32, None)

    nn = LeNet(3, n_classes, EPOCHS, rate, BATCH_SIZE, 'detect/tf_lenet')
    logits, _ = nn.train(images, labels, valid_images, valid_labels, x, y)
    print('Done training')

    # im = cv2.imread('/Users/anechaev/Developer/Python/TrafficLightClassification/dataset/udacity-sdc/red-green/0.jpg')
    # im = cv2.resize(im, (32, 32))
    #
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph('models/detect/tf_lenet.tf.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint('models/detect'))
    #
    #     graph = tf.get_default_graph()
    #     x = graph.get_tensor_by_name('x_value:0')
    #     predictions = graph.get_tensor_by_name('predictions:0')
    #     probabilities = graph.get_tensor_by_name('Softmax:0')
    #
    #     pred_nums, sf = sess.run([predictions, probabilities], feed_dict={x: [im]})
    #     print('Result {} nums; sf = {}'.format(pred_nums, sf))
