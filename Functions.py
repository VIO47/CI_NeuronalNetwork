import numpy as np

class Functions:

    def __int__(self):
        pass

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    #Pass a vector for this function and the index of the value inside the vector
    def softmax(self, Z, index):
        sum = 0
        for i in range(7):
            sum += np.linalg.norm(np.exp(Z[i]))
        return np.exp(Z[index]) / sum

    def tanh(self, Z):
        return np.tanh(Z)

    def sigmoid_back(self, Z, activation):
        sig = Functions.sigmoid(Z)
        return activation * sig * (1 - sig)

    def step(self, Z):
        return 1 if Z > 0 else 0

    def step_back(self, Z):
        return 0 if Z > 0 else np.Inf

    def relu_back(self, Z, activation):
        copy = np.array(activation, copy = True)
        copy[Z <= 0] = 0
        return copy

    def softmax_back(self, Z):
        jacobian_matrix = np.zeros((len(Z), len(Z)))
        softmax_x = Functions.softmax(Z)
        for i in range(len(Z)):
            for j in range(len(Z)):
                if i == j:
                    jacobian_matrix[i][j] = softmax_x[i] * (1 - softmax_x[i])
                else:
                    jacobian_matrix[i][j] = -softmax_x[i] * softmax_x[j]
        return jacobian_matrix.diagonal()

    def tanh_back(self, Z):
        return 1 - np.tanh(Z)**2

    def cross_entropy_back(self, y_hat, y):
        batch_size = y.shape[0]
        return -(y / y_hat) / batch_size

