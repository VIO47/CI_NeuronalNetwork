import numpy as np

class Perceptron:
    def __int__(self):
        self.weights = np.zeros(shape=(input.shape[1], 2))
        self.bias = np.zeros(shape=(2,))

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs = 5):
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.range(0, epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.weights))
                if p != target:
                    error = p - target
                    self.weights += -self.learning_rate * error + x

    def predict(self, X):
        return self.step(np.dot(X, self.weights))