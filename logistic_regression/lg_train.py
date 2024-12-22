import pandas as pd
import numpy as np
import pickle
from parsers import train_parser, filter_nan


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit(X, y, lr, e):
    X_a = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros((X_a.shape[1], 1))
    while np.linalg.norm(grad := X_a.T @ (sigmoid(X_a @ w) - y) / X_a.shape[0]) > e:
        w -= lr * grad
        print(np.linalg.norm(grad))
    return w


def fit_faculty(df, columns_faculty, faculty_str, step, e):
    data = df[columns_faculty].to_numpy()
    y = data[:, -1]
    y[y != faculty_str] = 0
    y[y == faculty_str] = 1
    data[:, -1] = y
    data = data.astype(float)
    data = np.array(list(filter(filter_nan, data)))
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    normalization_coeff = []
    for i in range(0, len(X.T)-1):
        x = X[:, i]
        min_x = min(x)
        max_x = max(x)
        x = 2 * (x - min_x) / (max_x - min_x) - 1
        X[:, i] = x
        normalization_coeff.append((min_x, max_x))
    w = fit(X, y, step, e)
    return w, normalization_coeff


def train_net(args):
    df = pd.read_csv(args.filepath)

    faculties_dict = {'Ravenclaw': ['Herbology', 'Ancient Runes', 'History of Magic',
                                    'Transfiguration', 'Charms', 'Flying', 'Hogwarts House'],
                      'Hufflepuff': ['Ancient Runes', 'Charms', 'Hogwarts House'],
                      'Slytherin': ['Herbology', 'Divination', 'Hogwarts House'],
                      'Gryffindor': ['Ancient Runes', 'Muggle Studies', 'Charms', 'Hogwarts House']
                      }
    results = {}
    for faculty in faculties_dict:
        w, normalization_coeff = fit_faculty(
            df, faculties_dict[faculty], faculty, args.learning_rate, args.precision)
        results[faculty] = w, normalization_coeff
    with open('weights.pickle', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    train_net(train_parser().parse_args())
