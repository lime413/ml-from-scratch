import time
import numpy as np
import pickle


class multilayer_perceptron():
    def __init__(self, layers):
        self.layers = np.asarray(layers)
        self.init_weights()
        self.train_loss = list()
        self.train_acc = list()
        self.val_loss = list()
        self.val_acc = list()
        self.train_time = list()
        self.total_time = list()

    def sigmoid(self, x):
        return 1./(1.+np.exp(-x))

    def softmax(self, x):
        exp = np.exp(x)
        return exp/exp.sum(axis=1, keepdims=True)

    def binary_cross_entropy(self, y_pred, y):
        return (np.log(y_pred) * y + np.log(1-y_pred) * (1-y)).sum(axis=1).mean()

    def accuracy(self, y_pred, y):
        return np.all(y_pred == y, axis=1).mean()

    def sigmoid_derivative(self, k):
        # derivative of sigmoid, k=sigmoid(x)
        return k*(1-k)

    def probabilities_2_categories(self, x):
        categories = np.zeros((x.shape[0], self.layers[-1]))
        categories[np.arange(x.shape[0]), x.argmax(axis=1)] = 1
        return categories

    def init_weights(self):
        self.weights = list()
        for i in range(self.layers.shape[0]-1):
            self.weights.append(
                np.random.uniform(-1, 1, size=[self.layers[i], self.layers[i+1]]))

    def init_hidden_values(self, batch_size):
        self.hidden_values = [np.empty((batch_size, layer))
                              for layer in self.layers]

    def feed_forward(self, batch):
        hidden_value = batch
        self.hidden_values[0] = batch
        for i, weights in enumerate(self.weights):
            hidden_value = self.sigmoid(hidden_value.dot(weights))
            self.hidden_values[i+1] = hidden_value
        self.out = self.softmax(self.hidden_values[-1])

    def back_propagation(self, batch_y):
        delta_t = (self.out - batch_y) * \
            self.sigmoid_derivative(self.hidden_values[-1])
        for i in range(1, len(self.weights)+1):
            self.weights[-i] -= self.lr * \
                (self.hidden_values[-i-1].T.dot(delta_t)) / self.batch_size
            delta_t = self.sigmoid_derivative(
                self.hidden_values[-i-1]) * (delta_t.dot(self.weights[-i].T))

    def predict_categories(self, x):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        self.init_hidden_values(x.shape[0])
        self.feed_forward(x)
        return self.probabilities_2_categories(self.out)

    def predict_probabilities(self, x):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        self.init_hidden_values(x.shape[0])
        self.feed_forward(x)
        return self.out

    def evaluate(self, x, y):
        with open('perceptron/weights.pickle', 'rb') as file:
            self.weights = pickle.load(file)
        y = np.squeeze(np.eye(self.layers[-1])[y.astype(np.int8).reshape(-1)])
        acc = self.accuracy(self.predict_categories(x), y)
        entropy = self.binary_cross_entropy(self.predict_probabilities(x), y)
        return acc, entropy

    def train(self, x_train, y_train, x_val, y_val, batch_size=4, num_epochs=50, lr=0.1):
        x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
        y_train = np.squeeze(
            np.eye(self.layers[-1])[y_train.astype(np.int8).reshape(-1)])
        x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
        y_val = np.squeeze(
            np.eye(self.layers[-1])[y_val.astype(np.int8).reshape(-1)])
        n_samples = x_train.shape[0]
        self.lr = lr
        self.batch_size = batch_size
        for epoch in range(num_epochs):
            start = time.time()

            self.init_hidden_values(self.batch_size)
            shuffle = np.random.permutation(n_samples)
            train_loss = 0
            train_acc = 0
            X_batches = np.array_split(
                x_train[shuffle], n_samples/self.batch_size)
            Y_batches = np.array_split(
                y_train[shuffle], n_samples/self.batch_size)
            for batch_x, batch_y in zip(X_batches, Y_batches):
                self.feed_forward(batch_x)
                train_loss += self.binary_cross_entropy(self.out, batch_y)
                train_acc += self.accuracy(
                    self.probabilities_2_categories(self.out), batch_y)
                self.back_propagation(batch_y)

            train_loss = (train_loss/len(X_batches))
            train_acc = (train_acc/len(X_batches))
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            train_time = round(time.time()-start, 3)
            self.train_time.append(train_time)

            self.init_hidden_values(x_val.shape[0])
            self.feed_forward(x_val)
            val_loss = self.binary_cross_entropy(self.out, y_val)
            val_acc = self.accuracy(
                self.probabilities_2_categories(self.out), y_val)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            total_time = round(time.time()-start, 3)
            self.total_time.append(total_time)

            if (epoch+1) % 5 == 0:
                print(f'Epoch {epoch+1}: loss = {train_loss.round(3)} | acc = {train_acc.round(3)} | \
                    val_loss = {val_loss.round(3)} | val_acc = {val_acc.round(3)} | train_time = {train_time} | tot_time = {total_time}')

        with open('perceptron/weights.pickle', 'wb') as file:
            pickle.dump(self.weights, file)
        with open('perceptron/architecture.pickle', 'wb') as file:
            pickle.dump(self.layers, file)
