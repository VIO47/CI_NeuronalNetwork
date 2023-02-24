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


    def __init__(self, structure, learning_rate):
        self.learning_rate = learning_rate
        self.structure = structure
        np.random.seed(50)
        number_layers = len(structure)
        self.weights = {}
        self.bias = {}

        for index, layer in enumerate(structure):
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
    def train_perceptron(self, X, Y, epochs = 30):
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



    def layer_forward_prop(self, A_prev, W_curr, b_curr, activation):
        Z_curr = np.dot(A_prev, W_curr) + b_curr.T
        if (activation == "relu"):
            activation_function = Func.relu(self, Z_curr)
        elif (activation == "tanh"):
            activation_function = Func.tanh(self, Z_curr)
        elif (activation == "sigmoid"):
            activation_function = Func.sigmoid(self, Z_curr)
        elif (activation == "step"):
            activation_function = Func.step(self, Z_curr)
        else:
            raise Exception("Unsupported activation function")

        return activation_function, Z_curr

    # z = W * x + bias
    def full_forward_prop(self, X):
        A_curr = X
        memory_activation = {}
        memory_intermediate = {}

        for index, layer in enumerate(self.structure):
            layer_index = index + 1
            A_prev = A_curr

            activation_function = layer["activation"]
            W_curr = self.weights[str(layer_index)]
            bias = self.bias[str(layer_index)]

            A_curr, Z_curr = NeuralNetwork.layer_forward_prop(self, A_prev, W_curr, bias, activation_function)
            memory_activation[str(index)] = A_prev
            memory_intermediate[str(layer_index)] = Z_curr

            return A_curr, memory_activation, memory_intermediate

    def accuracy(self,y_hat, y):
        return (y_hat == y).all(axis = 0).mean()

    def choose_class(self, y_hat):
        return np.argmax(y_hat)


    def layer_back_prop(self, loss, W_curr, bias, Z_curr, A_prev, activation):
        if(activation == "relu"):
            activation_function = Func.relu_back(Z_curr, loss)
        elif (activation == "softmax"):
            activation_function = Func.softmax_back(Z_curr)
        elif(activation == "tanh"):
            activation_function = Func.tanh_back(Z_curr)
        else:
            raise Exception("Unsupported activation function")

        dZ_curr = activation_function
        dW_curr = np.dot(dZ_curr, A_prev.T) / A_prev.shape[1]
        dbias_curr = np.sum(dZ_curr, axis = 1) / A_prev.shape[1]
        loss_prev = np.dot(W_curr.T, dZ_curr)

        return loss_prev, dW_curr, dbias_curr

    def full_back_prop(self, y_hat, y, memory_activation, memory_intermediate):
        new_weights = {}
        new_bias = {}

        dloss_prev = Func.cross_entropy_back(self, y_hat, y)

        for prev_layer, layer in reversed(list(enumerate(self.structure))):
            curr_layer = prev_layer + 1
            activation = layer["activation"]

            dloss_curr = dloss_prev
            print(prev_layer)
            #print(memory_activation[str(prev_layer)])
            A_prev = memory_activation[str(prev_layer)]
            Z_curr = memory_intermediate[str(curr_layer)]
            W_curr = self.weights[str(curr_layer)]
            bias = self.bias[str(curr_layer)]

            dloss_prev, dW_curr, dbias_curr = NeuralNetwork.layer_back_prop(dloss_curr, W_curr, bias, Z_curr, A_prev, activation)

            #Update process incorporated in back-propagation
            self.weights[str(curr_layer)] = dW_curr
            self.bias[str(curr_layer)] = dbias_curr

    def train(self, X, y, batch_size, epochs = 50):
        accuracy_hystory = []

        for i in range(epochs):
            #batch_X = np.array_split(X, batch_size)[:-1] + [X[-int(round(len(X)/batch_size)):]]
            #batch_y = np.array_split(y, batch_size)[:-1] + [y[-int(round(len(y)/batch_size)):]]
            batch_X, batch_y = NeuralNetwork.split_in_batches(self, X, y, batch_size)
            for (mini_batch_X, mini_batch_y) in zip(batch_X, batch_y):
                y_hat, aux_activation, aux_intermediate = NeuralNetwork.full_forward_prop(self, mini_batch_X)
                predicted_y = NeuralNetwork.choose_class(self, y_hat)
                accuracy = NeuralNetwork.accuracy(self, predicted_y, mini_batch_y)
                accuracy_hystory.append(accuracy)

                print(aux_activation)
                NeuralNetwork.full_back_prop(self, predicted_y, mini_batch_y, aux_activation, aux_intermediate)

        return accuracy_hystory

    def split_in_batches(self, X, Y, batch_size):
        batches_X = []
        batches_Y = []
        for i in range(int(len(X) / batch_size + 1)):
            mini_X = X[i * batch_size : (i + 1) * batch_size]
            batches_X.append(mini_X)
            mini_Y = Y[i * batch_size : (i + 1) * batch_size]
            batches_Y.append(mini_Y)

        return batches_X, batches_Y



