import numpy as np
from Functions import Functions as Func


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

    def loss(self, y_hat, y):
        loss = 0
        for (computed, target) in zip(y_hat, y):
            loss -= target * np.log(computed)
        return loss

    def loss_perceptron(self, y_hat, y):
        return 1.0 * (y - y_hat)

    def inference(self, X):
        return Func.step(self, np.dot(self.weights["1"].T, X) + self.bias["1"])

    def update_weights(self, X, loss):
        self.bias["1"] += self.learning_rate * loss
        self.weights["1"] += self.learning_rate * loss * np.array(X).reshape(2, 1)

   # def update_weights_final_layer(self, X, loss):
    #    #derivative for softmax
    #    if(self.activation_function == "softmax")
    #    self.bias -= self.learning_rate *

    #def update_weights_hidden_layer(self, X, loss):
        #derivative for relu
    def train(self, X, Y, epochs = 30):
        accuracy_arr = []
        for epoch in range(epochs):
            accuracy = 0
            for (x, target) in list(zip(X, Y)):
                Z = NeuralNetwork.inference(self, x)
                if Z != target:
                    loss = NeuralNetwork.loss_perceptron(self, Z, target)
                    NeuralNetwork.update_weights(self, x, loss)
                else:
                    accuracy += 1
            accuracy_arr.append(accuracy / 4.0)
        return accuracy_arr



    def perceptron_forward_prop(self, A_prev, W_curr, b_curr, activation):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        if (activation == "relu"):
            activation_function = Func.relu(Z_curr)
        elif (activation == "tanh"):
            activation_function = Func.tanh(Z_curr)
        elif (activation == "sigmoid"):
            activation_function = Func.sigmoid(Z_curr)
        elif (activation == "step"):
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
