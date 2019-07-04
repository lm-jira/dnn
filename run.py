import click
import importlib

from tensorflow import keras

@click.command()
@click.option('--opt', type=click.Choice(['train', 'test', 'export', 'predict']), default='train', 
                  help='train or test the model')
@click.option('--predict_img', type=click.Choice(['train', 'test', 'export', 'predict']), default='train', 
                  help='train or test the model')
def run(opt, predict_img):
    fashion_mnist = keras.datasets.fashion_mnist
    mod_name = "executor.{}".format(opt)
    mod = importlib.import_module(mod_name)

    if opt == "train":
        (train_images, train_labels), (_, _) = fashion_mnist.load_data()
        # preprocess images
        train_images = train_images / 255.0

        mod.run(train_images, train_labels)

    elif opt == "test":
        (_, _), (test_images, test_labels) = fashion_mnist.load_data()
        # preprocess images
        test_images = test_images / 255.0

        mod.run(test_images, test_labels)

    elif opt == "export":
        mod.run()

    elif opt == 'predict':
        mod.run(predict_img)


if __name__ == "__main__":
    run()

    


