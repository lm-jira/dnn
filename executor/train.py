import tensorflow as tf
import numpy as np

from network.simpleNet import simpleNet


def _train(train_images, train_labels):

    graph = tf.Graph()

    with graph.as_default():
        model = simpleNet(epochs=10, batch_size=100)

        x_pl, y_pl = model.placeholders()
        output = model.network(x_pl)
        loss = model.loss(y_pl, output)

        opt = model.optimizer()
        train_op = model.train_op(loss, opt)

        # saver object for saving checkpoints
        saver = tf.train.Saver()

        sess = tf.Session(graph=graph)
        sess.run(tf.global_variables_initializer())

        i=1
        while(i<=model.epochs):
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

            if(i%10==0):
                # save checkpoints every 10 epochs
                save_path = saver.save(sess, "./checkpoints/model.ckpt", global_step = i)
                print("Model saved in path {}".format(save_path))
            i+=1

        save_path = saver.save(sess, "./checkpoints/model.ckpt")


def run(train_images, train_labels):
    _train(train_images, train_labels)
