import tensorflow as tf
import numpy as np

from network.simpleNet import simpleNet


def _test(test_images, test_labels):

    graph = tf.Graph()

    with graph.as_default():
        model = simpleNet(epochs=30, batch_size=100)
        x_pl, y_pl = model.placeholders()
        output = model.network(x_pl)

        # load saved model
        saver = tf.train.Saver()
        sess = tf.Session(graph=graph)
        saver.restore(sess, "./checkpoints/model.ckpt")

        iter_per_epoch = len(test_images)/model.batch_size
        i = 0
        sum_correct = 0

        while(i<iter_per_epoch):
            feed_dict = {x_pl: test_images[i*model.batch_size: (i+1)*model.batch_size]}
            predict_y = sess.run(output, feed_dict=feed_dict)

            test_label_batch = test_labels[i*model.batch_size: (i+1)*model.batch_size]
            predict_y = np.argmax(predict_y, axis=1)
            correct = predict_y==test_label_batch
            correct = correct.astype(int)
            correct = np.sum(correct)
            sum_correct += correct
        
            i+=1

        print("acc = ", sum_correct/float(len(test_labels)))


def run(test_images, test_labels):
    _test(test_images, test_labels)
