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

    def shuffle_arrays(self, arrays, set_seed=-1):
        """Shuffles arrays in-place, in the same order, along axis=0

        Parameters:
        -----------
        arrays : List of NumPy arrays.
        set_seed : Seed value if int >= 0, else seed is random.
        """
        seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed
        for arr in arrays:
            rstate = np.random.RandomState(seed)
            rstate.shuffle(arr)

    def train_test_split(self, X, y):
        ratio_train = 0.85

        rows = X.shape[0]
        train_size = int(rows * ratio_train)

        X_train = X[0:train_size]
        y_train = y[0:train_size]

        X_test = X[train_size:]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test

    def normalize_data(self, X):
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        # Normalize the data
        X_norm = (X - means) / stds
        return X_norm