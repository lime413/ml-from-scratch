import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from mlp import multilayer_perceptron


def load_data():
    data_path = 'perceptron/data.csv'
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


def train_mlp():
    x_train, y_train, x_val, y_val = load_data()

    # rescale data between 0 - 1
    x_train = x_train/x_train.max()
    x_val = x_val/x_val.max()

    model = multilayer_perceptron(layers=[x_train.shape[1]+1, 24, 24, 24, 2])
    model.train(x_train, y_train, x_val, y_val)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
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


def test_mlp():
    _, _, x_val, y_val = load_data()
    x_val = x_val/x_val.max()
    with open('perceptron/architecture.pickle', 'rb') as file:
        layers = pickle.load(file)
    model = multilayer_perceptron(layers)
    acc, entropy = model.evaluate(x_val, y_val)
    print('Accuracy: ', round(acc, 3))
    print('Cross-entropy: ', round(entropy, 3))


if __name__ == '__main__':
    np.random.seed(2024)
    train_mlp()
    test_mlp()
