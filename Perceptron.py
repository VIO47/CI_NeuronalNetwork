import numpy as np
from Functions import Functions as Func

class Perceptron:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size, 1)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def perceptron_forward_prop(self, A_prev, W_curr, b_curr, activation, index):
        Z_curr = np.dot(A_prev, W_curr) + b_curr.T
        if (activation == "relu"):
            activation_function = Func.relu(self, Z_curr)
        elif (activation == "tanh"):
            activation_function = Func.tanh(self, Z_curr)
        elif (activation == "sigmoid"):
            activation_function = Func.sigmoid(self, Z_curr)
        elif (activation == "step"):
            activation_function = Func.step(self, Z_curr)
        elif (activation == "softmax"):
            activation_function = Func.softmax(self, Z_curr, index)
        else:
            raise Exception("Unsupported activation function")

        return activation_function, Z_curr

    def loss(self, y_hat, y):
        loss = 0
        for (computed, target) in zip(y_hat, y):
            loss -= target * np.log(computed)
        return loss
    def accuracy(self, y_hat, y):
        return np.mean(np.argmax(y_hat, axis=1) == y)

    def perceptron_back_propag(self, loss, Z, A_prev):
        if (self.activation_function == "relu"):
            deriv = Func.relu_back(loss, Z)
        elif (self.activation_function == "tanh"):
            deriv = Func.tanh_back(loss)
        elif (self.activation_function == "sigmoid"):
            deriv = Func.sigmoid_back(loss, Z)
        elif (self.activation_function == "step"):
            deriv = Func.step_back(loss)
        elif (self.activation_function == "softmax"):
            deriv = Func.softmax_back(loss)
        else:
            raise Exception("Unsupported activation function")

        dW_curr = np.dot(deriv, A_prev.T) / A_prev.shape[1]
        db_curr = np.sum(deriv, axis = 1, keepdims = True) / A_prev.shape[1]
        dA_prev = np.dot(self.weights, deriv)

        return dW_curr, db_curr
    def update(self, dW_curr, db_curr):
        self.bias -= self.learning_rate * db_curr
        self.weights -= self.learning_rate * dW_curr

