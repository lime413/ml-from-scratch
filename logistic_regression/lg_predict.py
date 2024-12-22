import pandas as pd
import numpy as np
import pickle
from parsers import filepath_parser, filter_nan


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(X, w):
    return sigmoid(w[0] + X @ w[1:])


def generate_test_csv(args):
    with open('weights.pickle', 'rb') as f:
        train_dict = pickle.load(f)
    df_test = pd.read_csv('data/logistic_regression_test.csv')
    df_test = df_test.drop(['Index', 'Hogwarts House'], axis=1)

    faculties_dict = {'Ravenclaw': ['Herbology', 'Ancient Runes', 'History of Magic',
                                    'Transfiguration', 'Charms', 'Flying'],
                      'Hufflepuff': ['Ancient Runes', 'Charms'],
                      'Slytherin': ['Herbology', 'Divination'],
                      'Gryffindor': ['Ancient Runes', 'Muggle Studies', 'Charms']
                      }

    predictions = np.zeros(shape=(len(df_test), len(faculties_dict)))
    for k, faculty in enumerate(faculties_dict):
        X = np.nan_to_num(
            df_test[faculties_dict[faculty]].to_numpy().astype(float))
        for i in range(0, len(X.T)-1):
            x = X[:, i]
            min_x = train_dict[faculty][1][i][0]
            max_x = train_dict[faculty][1][i][1]
            x = 2 * (x - min_x) / (max_x - min_x) - 1
            X[:, i] = x
        predictions[:, k:k+1] = predict(X, train_dict[faculty][0])
    y = np.argmax(predictions, axis=1)

    y = y.astype(str)
    str_dict = {'0': 'Ravenclaw', '1': 'Hufflepuff',
                '2': 'Slytherin', '3': 'Gryffindor'}
    for idx in str_dict:
        y[y == idx] = str_dict[idx]

    df = pd.DataFrame({'Index': np.arange(0, len(y)), 'Hogwarts House': y})
    df.to_csv('houses.csv', index=False)
    print('house.csv generated')


def test_on_train_data(args):
    with open('weights.pickle', 'rb') as f:
        train_dict = pickle.load(f)
    df_test = pd.read_csv('data/logistic_regression_train.csv')
    y = df_test['Hogwarts House'].to_numpy()
    df_test = df_test.drop(['Index', 'Hogwarts House'], axis=1)

    faculties_dict = {'Ravenclaw': ['Herbology', 'Ancient Runes', 'History of Magic',
                                    'Transfiguration', 'Charms', 'Flying'],
                      'Hufflepuff': ['Ancient Runes', 'Charms'],
                      'Slytherin': ['Herbology', 'Divination'],
                      'Gryffindor': ['Ancient Runes', 'Muggle Studies', 'Charms']
                      }

    predictions = np.zeros(shape=(len(df_test), len(faculties_dict)))
    for k, faculty in enumerate(faculties_dict):
        X = np.nan_to_num(
            df_test[faculties_dict[faculty]].to_numpy().astype(float))
        for i in range(0, len(X.T)-1):
            x = X[:, i]
            min_x = train_dict[faculty][1][i][0]
            max_x = train_dict[faculty][1][i][1]
            x = 2 * (x - min_x) / (max_x - min_x) - 1
            X[:, i] = x
        predictions[:, k:k+1] = predict(X, train_dict[faculty][0])
    pred = np.argmax(predictions, axis=1)

    y[y == 'Ravenclaw'] = 0
    y[y == 'Hufflepuff'] = 1
    y[y == 'Slytherin'] = 2
    y[y == 'Gryffindor'] = 3
    y = y.astype(int)

    accuracy = np.sum((pred == y).astype(int))/len(y)
    # tp = 0
    # fp = 0
    # tn = 0
    # fn = 0
    # for i in range(0, len(y)):
    #     if pred[i]==1 and y[i]==1 :
    #         tp+=1
    #     elif pred[i]==1 and y[i]==0 :
    #         fp+=1
    #     elif pred[i]==0 and y[i]==0:
    #         tn+=1
    #     else:
    #         fn+=1
    # precision=tp/(tp+fp)
    # recall=tp/(tp+fn)
    # confusion_matrix = np.array([[tp, fp], [fn, tn]])

    print("Accuracy: " + str(accuracy))
    # print("Confusion matrix")
    # print("tp fp")
    # print("fn tn")
    # print(confusion_matrix)
    # print("Precision: " + str(round(precision,2)))
    # print("Recall: " + str(round(recall,2)))


if __name__ == '__main__':
    args = filepath_parser().parse_args()
    test_on_train_data(args)
    generate_test_csv(args)
