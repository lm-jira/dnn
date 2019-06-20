import tensorflow as tf
from tensorflow import keras

import numpy as np
import sklearn.metrics as metrics

from network.simpleNet import simpleNet

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# preprocess images
train_images = train_images / 255.0
test_images = test_images / 255.0

graph = tf.Graph()

with graph.as_default():
    model = simpleNet(epochs=10, batch_size=100)

    x_pl, y_pl = model.placeholders()
    output = model.network(x_pl)
    loss = model.loss(y_pl, output)

    opt = model.optimizer()
    train_op = model.train_op(loss, opt)

    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())

    i=0
    while(i<model.epochs):
        print("epochs = ", i)
        indexes = np.random.permutation(len(train_labels))
        iter_per_epoch = len(train_images)/model.batch_size
        j=0

        while(j<iter_per_epoch):
            x_batch = train_images[indexes[j*model.batch_size: (j+1)*(model.batch_size)]]
            y_batch = train_labels[indexes[j*model.batch_size: (j+1)*(model.batch_size)]]

            feed_dict = {x_pl: x_batch,
                         y_pl: y_batch
                         }

            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)

            j+=1
        i+=1

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


