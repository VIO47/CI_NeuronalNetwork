import numpy as np
import Functions as Func


class NeuralNetwork:
    # Change to reshape the ANN and change the used activation xfunctions
    #nn_architecture = [
    #    {"input_dim": 10, "output_dim": 8, "activation": "relu"},
    #    {"input_dim": 8, "output_dim": 6, "activation": "relu"},
    #    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    #    {"input_dim": 4, "output_dim": 2, "activation": "relu"},
    #    {"input_dim": 2, "output_dim": 1, "activation": "relu"}
    #]


    def __init__(self, nn_structure, learning_rate):
        self.learning_rate = learning_rate
        np.random.seed(50)
        number_layers = len(nn_structure)
        self.weights = {}
        self.bias = {}

        for index, layer in enumerate(nn_structure):
            layer_index = index + 1
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]

            self.weights[str(layer_index)] = np.random.randn(input_size, output_size)
            self.bias[str(layer_index)] = np.random.randn(output_size, 1)

    def accuracy(self, y_hat, y):
        return np.mean(np.argmax(y_hat, axis=1) == y)

    # Have used binary cross-entropy as it's the most used loss function for classification problems
    def loss(self, y_hat, y):
        m = y_hat.shape[1]
        return -1 * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))

    def loss_perceptron(self, y_hat, y):
        return y_hat - y

    def inference(self, X):
        return Func.step(np.dot(self.weights["1"], X) + self.bias["1"])

    def update_weights(self, X, y):
        loss = NeuralNetwork.loss_perceptron(NeuralNetwork.inference(X), y)
        self.bias["1"] += self.bias["1"] + self.learning_rate * loss
        self.weights["1"] += self.learning_rate * loss * X

    def layer_forward_prop(self, A_prev, W_curr, b_curr, activation):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        if (activation is "relu"):
            activation_function = Func.relu(Z_curr)
        elif (activation is "tanh"):
            activation_function = Func.tanh(Z_curr)
        elif (activation is "sigmoid"):
            activation_function = Func.sigmoid(Z_curr)
        elif (activation is "step"):
            activation_function = Func.step(Z_curr)
        else:
            raise Exception("Unsupported activation function")

        return activation_function(Z_curr), Z_curr

    # z = W * x + bias
    def full_forward_prop(self, X, nn_architecture):
        A_curr = X
        memory_activation = {}
        memory_intermediate = {}

        for index, layer in enumerate(nn_architecture):
            layer_index = index + 1
            A_prev = A_curr
            activation_function = layer["activation"]
            W_curr = self.weights[str(layer_index)]
            bias = self.bias[str(layer_index)]

            A_curr, Z_curr = NeuralNetwork.layer_forward_prop(A_prev, W_curr, bias, activation_function)
            memory_activation[str(index)] = A_prev
            memory_intermediate[str(layer_index)] = Z_curr

            return A_curr, memory_activation, memory_intermediate
