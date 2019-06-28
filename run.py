import click
import importlib

from tensorflow import keras

@click.command()
@click.option('--opt', type=click.Choice(['train', 'test']), default='train', 
                  help='train or test the model')
def run(opt):
    fashion_mnist = keras.datasets.fashion_mnist

    if opt == "train":
        (train_images, train_labels), (_, _) = fashion_mnist.load_data()
        # preprocess images
        train_images = train_images / 255.0

        mod_name = "executor.train"
        mod = importlib.import_module(mod_name)
        mod.run(train_images, train_labels)

    elif opt == "test":
        (_, _), (test_images, test_labels) = fashion_mnist.load_data()
        # preprocess images
        test_images = test_images / 255.0

        mod_name = "executor.test"
        mod = importlib.import_module(mod_name)
        mod.run(test_images, test_labels)


if __name__ == "__main__":
    run()

    


