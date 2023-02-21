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
        for i in range(index):
            sum += np.linalg.norm(np.exp(Z[i]))
        return np.exp(Z[index]) / sum

    def tanh(self, Z):
        return np.tanh(Z)

    def sigmoid_back(self, Z, activation):
        sig = Functions.sigmoid(Z)
        return activation * sig * (1 - sig)

    def relu_back(self, Z, activation):
        copy = np.array(activation, copy = True)
        copy[Z <= 0] = 0
        return copy

    def softmax_back(self, Z):
        j_matrix = np.diag(Z)
        for i in range(len(j_matrix)):
            for j in range(len(j_matrix)):
                if(i == j):
                    j_matrix[i][j] = Z[i] * (1 - Z[i])
                else:
                    j_matrix[i][j] = -Z[i] * Z[j]

        return j_matrix

    def tanh_back(self, Z):
        return 1 - np.tanh(Z)**2

