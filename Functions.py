import numpy as np

class Functions:

    def __int__(self):
        pass

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        e_x = np.exp(Z - np.amax(Z))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def tanh(self, Z):
        return np.tanh(Z)

    def sigmoid_back(self, dA, Z):
        sig = Functions.sigmoid(self, Z)
        x = sig * (1 - sig)
        d = dA * x
        return d

    def step(self, Z):
        return 1 if Z > 0 else 0

    def step_back(self, Z):
        return 0 if Z > 0 else np.Inf

    def relu_back(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    # def softmax_back(self, dA, Z):
    #     jacobian_matrix = np.zeros((Z.shape[0], Z.shape[0]))
    #
    #     # for each column obj compute its jacob mat 7x7
    #     for obj in range(Z.shape[1]):
    #         softmax_x = Functions.softmax(self, Z.T[obj])
    #         mini_jacob_mat = np.zeros((Z.shape[0], Z.shape[0]))
    #         for i in range(Z.shape[0]):
    #             for j in range(Z.shape[0]):
    #                 if i == j:
    #                     mini_jacob_mat[i][j] = np.dot(softmax_x[i], (1 - softmax_x[i]).T)
    #                 else:
    #                     jacobian_matrix[i][j] = - np.dot(softmax_x[i], softmax_x[j].T)
    #         jacobian_matrix += mini_jacob_mat
    #     return (jacobian_matrix / Z.shape[1]) * dA
       # exp_z = np.exp(Z)
       # sum = exp_z.sum()
       # return np.round(exp_z / sum, 3)

    def tanh_back(self, Z):
        return 1 - np.tanh(Z)**2

    def cat_cross_entropy_loss(self, y_hat, y):
        return (- y * np.log(y_hat + 1e-8)).mean()

    def accuracy(self, y_hat, y):
        idx_pred = np.argmax(y_hat, axis=0)
        idx_real = np.argmax(y, axis=0)
        s = 0
        for i in range(idx_pred.size):
            if idx_real[i] == idx_pred[i]: s += 1
        return s / idx_pred.size


