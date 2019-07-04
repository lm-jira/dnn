import tensorflow as tf

from network.simpleNet import simpleNet

def _export():
    restore_path = './checkpoints/model.ckpt'

    graph = tf.Graph()

    with graph.as_default():
        model = simpleNet(epochs=10, batch_size=100)

        x_pl, y_pl = model.placeholders()
        output = model.network(x_pl)
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()

    sess = tf.Session(graph=graph)
    sess.run(init_op)
    saver.restore(sess, restore_path)

    output_node_names = ["softmax"]
    output_dir = "./export"
    minimal_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names)

    tf.train.write_graph(minimal_graph_def, output_dir, "graph_def.pb", as_text=False)

def run():
    _export()
