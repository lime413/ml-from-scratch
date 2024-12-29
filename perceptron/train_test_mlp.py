import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from mlp import multilayer_perceptron


def load_data(data_path):
    df = pd.read_csv(data_path)
    y_char = df[df.columns[1]].to_numpy()
    x = df[df.columns[[2, 3, 4, 5, 7, 8, 9, 12, 15, 22, 24, 25, 28, 29]]].to_numpy()

    y = np.zeros(len(y_char))
    y[y_char == 'M'] = 1

    n_train = 0.8
    x_train = x[:int(n_train*len(x))]
    x_val = x[int(n_train*len(x)):]
    y_train = y[:int(n_train*len(y))]
    y_val = y[int(n_train*len(y)):]

    return x_train, y_train, x_val, y_val


def train_mlp(args):
    print('Training MLP...')
    x_train, y_train, x_val, y_val = load_data(args.input)

    # rescale data between 0 - 1
    x_train = x_train/x_train.max()
    x_val = x_val/x_val.max()

    model = multilayer_perceptron(layers=np.hstack(
        ([x_train.shape[1]+1], args.layers, [2])))
    model.train(x_train, y_train, x_val, y_val, args.batch_size,
                args.epochs, args.learning_rate)

    _, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(np.abs(model.train_loss), label='Train loss')
    ax[0].plot(np.abs(model.val_loss), label='Val loss')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].grid()

    ax[1].plot(model.train_acc, label='Train acc')
    ax[1].plot(model.val_acc, label='Val acc')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid()


def test_mlp(args):
    print('Testing MLP...')
    _, _, x_val, y_val = load_data(args.input)
    x_val = x_val/x_val.max()
    with open('perceptron/layers.pickle', 'rb') as file:
        layers = pickle.load(file)
    model = multilayer_perceptron(layers)
    acc, entropy = model.evaluate(x_val, y_val)
    print('Accuracy: ', round(acc, 5))
    print('Cross-entropy: ', round(entropy, 5))


if __name__ == '__main__':
    np.random.seed(2024)

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='perceptron\data.csv',
                        help='filepath for csv-file with data')
    parser.add_argument(
        '-layers', type=int, default=[24, 24, 24], nargs='+', help='number of neurons in hidden layers')
    parser.add_argument('-batch-size', type=int, default=4)
    parser.add_argument('-learning-rate', type=float, default=0.1)
    parser.add_argument('-epochs', type=int, default=50)
    args = parser.parse_args()

    train_mlp(args)
    test_mlp(args)
    plt.show()
