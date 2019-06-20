import tensorflow as tf

class simpleNet:
    def __init__(self, epochs=20, batch_size=100, image_size=28):
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size

        self.x_pl = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size], name="x_pl")
        self.y_pl = tf.placeholder(tf.int32, shape=[None], name="y_pl")

    def network(self, x_pl):
        # building model
        x = tf.reshape(x_pl, shape=(-1, 28*28))
        x = tf.layers.dense(x, 128)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, 128)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, 10)
        output = tf.nn.softmax(x)
        return output

    def loss(self, y_pl, y_pred):
        loss = tf.keras.metrics.sparse_categorical_crossentropy(y_true=y_pl, y_pred=y_pred)
        return loss

    def train_op(self, loss, opt):
        train_op = opt.minimize(loss)
        return train_op

    def optimizer(self):
        opt = tf.train.AdamOptimizer()
        return opt

    def placeholders(self):
        return self.x_pl, self.y_pl
