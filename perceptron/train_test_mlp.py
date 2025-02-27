import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

from mlp import multilayer_perceptron


def filter_nan(arr):
    if np.isnan(arr).any():
        return False
    return True


def load_data(data_path, columns):
    df = pd.read_csv(data_path)
    y_char = df[df.columns[1]].to_numpy()
    x = df[df.columns[columns]].to_numpy()

    y = np.zeros(len(y_char))
    y[y_char == 'M'] = 1

    n_train = 0.8
    x_train = x[:int(n_train*len(x))]
    x_val = x[int(n_train*len(x)):]
    y_train = y[:int(n_train*len(y))]
    y_val = y[int(n_train*len(y)):]

    return x_train, y_train, x_val, y_val


def train_mlp(args, columns):
    print('Training MLP...')
    x_train, y_train, x_val, y_val = load_data(args.input, columns)

    # rescale data between 0 - 1
    x_train = x_train/x_train.max()
    x_val = x_val/x_val.max()

    model = multilayer_perceptron(layers=np.hstack(
        ([x_train.shape[1]+1], args.layers, [2])))
    model.train(x_train, y_train, x_val, y_val, args.batch_size,
                args.epochs, args.learning_rate)

    return model


def test_mlp(args, pupu):
    print('Testing MLP...')
    _, _, x_val, y_val = load_data(args.input, pupu)
    x_val = x_val/x_val.max()
    with open('perceptron/layers.pickle', 'rb') as file:
        layers = pickle.load(file)
    model = multilayer_perceptron(layers)
    acc, entropy = model.evaluate(x_val, y_val)
    print('Accuracy: ', round(acc, 5))
    print('Cross-entropy: ', round(entropy, 5))
    return acc


def find_columns(args):
    columns = np.arange(0, 32)
    columns = columns[columns != 1]
    train_mlp(args, columns)
    max_acc = test_mlp(args, columns)
    
    for i in range(0, 32):
        if i == 1:
            continue
        columns_cur = columns[columns != i]
        model = train_mlp(args, columns_cur)
        acc = test_mlp(args, columns_cur)
        if acc > max_acc:
            columns = columns_cur
            max_acc = acc
    print(columns)
    print(max_acc)
    return columns



if __name__ == '__main__':
    np.random.seed(2024)

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='perceptron/data.csv',
                        help='filepath for csv-file with data')
    parser.add_argument(
        '-layers', type=int, default=[24, 24, 24], nargs='+', help='number of neurons in hidden layers')
    parser.add_argument('-batch-size', type=int, default=4)
    parser.add_argument('-learning-rate', type=float, default=0.1)
    parser.add_argument('-epochs', type=int, default=50)
    args = parser.parse_args()

    columns = find_columns(args)
    os.system('cls')
    model = train_mlp(args, columns)
    test_mlp(args, columns)

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

    plt.show()