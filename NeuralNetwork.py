import numpy as np
from Functions import Functions as Func

class NeuralNetwork:
    def __init__(self, structure, learning_rate, seed = None):
        self.learning_rate = learning_rate
        self.structure = structure
        np.random.seed(seed)
        self.number_layers = len(structure)
        self.weights = {}
        self.bias = {}

        for index, layer in enumerate(structure):
            layer_index = index + 1
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]
            self.bias[layer_index] = np.zeros([output_size, 1])

            if layer_index == 3:
                #self.weights[layer_index] = np.random.randn(output_size, input_size)
                #self.weights[layer_index] = np.random.uniform(-np.sqrt(6 / (input_size + output_size)), np.sqrt(6 / (input_size + output_size)), [output_size, input_size])
                self.weights[layer_index] = np.random.normal(0, 1 / input_size, [output_size, input_size])
            else:
                self.weights[layer_index] = np.random.normal(0, 2 / input_size, [output_size, input_size])
                #self.weights[layer_index] = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)

    def layer_forward_prop(self, W_curr, A_prev, B_curr, activation):
        f = Func()
        Z_curr = np.dot(W_curr, A_prev) + B_curr

        if (activation == "relu"):
            activated = Func.relu(f, Z_curr)
        elif (activation == "tanh"):
            activated = Func.tanh(f, Z_curr)
        elif (activation == "sigmoid"):
            activated = Func.sigmoid(f, Z_curr)
        elif (activation == "softmax"):
            activated = Func.softmax(f, Z_curr)
        else:
            raise Exception("Unsupported activation function")

        return activated, Z_curr

    # z = W * x + bias
    def full_forward_prop(self, X):
        A_curr = X
        memory_activation = {}
        memory_intermediate = {}

        for index, layer in enumerate(self.structure):
            layer_index = index + 1
            A_prev = A_curr

            activation_function = layer["activation"]
            W_curr = self.weights[layer_index]
            B_curr = self.bias[layer_index]

            A_curr, Z_curr = NeuralNetwork.layer_forward_prop(self, W_curr, A_prev, B_curr, activation_function)

            memory_activation[index] = A_prev
            memory_intermediate[layer_index] = Z_curr

        return A_curr, memory_activation, memory_intermediate

    def final_layer_back_prop(self, W_curr, Z_curr, A_prev, y):
        A_prev_cols = A_prev.shape[1]
        dZ_curr = Z_curr - y
        dW_curr = np.dot(dZ_curr, A_prev.T) / A_prev_cols
        dB_curr = np.sum(dZ_curr, axis=1, keepdims=True) / A_prev_cols
        dA_prev = np.dot(W_curr.T, dZ_curr)
        return dA_prev, dW_curr, dB_curr

    def layer_back_prop(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        f = Func()

        if(activation == "relu"):
            activated = Func.relu_back(f, dA_curr, Z_curr)
        elif(activation == "tanh"):
            activated = Func.tanh_back(f, Z_curr)
        elif(activation == "sigmoid"):
            activated = Func.sigmoid_back(f, dA_curr, Z_curr)
        else:
            raise Exception("Unsupported activation function")

        A_prev_cols = A_prev.shape[1]
        dZ_curr = activated
        dW_curr = np.dot(dZ_curr, A_prev.T) / A_prev_cols
        dB_curr = np.sum(dZ_curr, axis = 1, keepdims=True) / A_prev_cols
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, dB_curr

    def full_back_prop(self, y_hat, y, memory_activation, memory_intermediate):
        f = Func()
        gradients_weights = {}
        gradients_biases = {}
        y = y.reshape(y_hat.shape)
        ok = 0

        dA_prev = 0
        for prev_layer, layer in reversed(list(enumerate(self.structure))):
            curr_layer = prev_layer + 1
            activation = layer["activation"]

            dA_curr = dA_prev
            A_prev = memory_activation[prev_layer]
            Z_curr = memory_intermediate[curr_layer]
            W_curr = self.weights[curr_layer]

            if ok == 0:
                dA_prev, dW_curr, dB_curr = NeuralNetwork.final_layer_back_prop(self, W_curr, Z_curr, A_prev, y)
                ok = 1
            else:
                dA_prev, dW_curr, dB_curr = NeuralNetwork.layer_back_prop(self, dA_curr, W_curr, Z_curr, A_prev, activation)

            gradients_weights[curr_layer] = dW_curr
            gradients_biases[curr_layer] = dB_curr

        return gradients_weights, gradients_biases

    def update(self, gradients_weights, gradients_biases):
        for layer_idx, layer in enumerate(self.structure):
            curr_idx = layer_idx +1
            self.weights[curr_idx] -= self.learning_rate * gradients_weights[curr_idx]
            self.bias[curr_idx] -= self.learning_rate * gradients_biases[curr_idx]

    def train(self, X, y, batch_size=None, epochs = 1000):
        accuracy_history = []
        loss_history = []
        y_hat_history = []

        for i in range(epochs):
            f = Func()
            aux_X = np.copy(X)
            aux_y = np.copy(y)
            Func.shuffle_arrays(f, [aux_X, aux_y])
            aux_X = aux_X.T
            aux_y = aux_y.T
            # batch_X, batch_y = NeuralNetwork.split_in_batches(self, X, y, batch_size)
            # for (mini_batch_X, mini_batch_y) in zip(batch_X, batch_y):
            #     y_hat, aux_activation, aux_intermediate = NeuralNetwork.full_forward_prop(self, mini_batch_X)
            #
            #     loss = Func.cat_cross_entropy_loss(f, y_hat, mini_batch_y)
            #     loss_history.append(loss)
            #     accuracy = Func.accuracy(f, y_hat, mini_batch_y)
            #     accuracy_history.append(accuracy)
            #
            #     gradients_weights, gradients_biases = NeuralNetwork.full_back_prop(self, y_hat, mini_batch_y, aux_activation, aux_intermediate)
            #     NeuralNetwork.update(self, gradients_weights, gradients_biases)
            y_hat, aux_activation, aux_intermediate = NeuralNetwork.full_forward_prop(self, aux_X)

            loss = Func.cat_cross_entropy_loss(f, y_hat, aux_y)
            loss_history.append(loss)
            accuracy = Func.accuracy(f, y_hat, aux_y)
            accuracy_history.append(accuracy)
            y_hat_history.append(y_hat)

            gradients_weights, gradients_biases = NeuralNetwork.full_back_prop(self, y_hat, aux_y, aux_activation, aux_intermediate)
            NeuralNetwork.update(self, gradients_weights, gradients_biases)

        return loss_history, y_hat_history, accuracy_history

    def predict(self, X, y, make_accuracy):
        f = Func()
        acc_history = []
        result = X.T
        if(make_accuracy):
            y = y.T

        for layer_idx, layer in enumerate(self.structure):
            curr_idx = layer_idx + 1
            result = np.dot(self.weights[curr_idx], result) + self.bias[curr_idx]

        if(make_accuracy):
            acc_history.append(Func.accuracy(f, result, y))
        y_hat = np.argmax(result, axis = 0)
        return y_hat, acc_history

    def k_fold_cross_validation(self, X, y, k, epochs, batch_size=None):
        """
        Perform k-fold cross validation on a neural network model.

        Parameters:
        X (numpy array): The input data.
        y (numpy array): The target data.
        k (int): The number of folds to use.
        model_fn (function): A function that returns a compiled Keras model.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size to use during training.

        Returns:
        results (list): A list of the validation accuracies for each fold.
        """

        fold_size = len(X) // k
        accuracies = []

        for i in range(k):
            print(f"Processing fold {i + 1}/{k}...")
            start = i * fold_size
            end = (i + 1) * fold_size

            x_val = X[start:end]
            y_val = y[start:end]

            x_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            loss_train, y_hat_train, accuracy_train = NeuralNetwork.train(self, x_train, y_train, epochs)
            y_hat_validation, accuracy_validation = NeuralNetwork.predict(self, x_val, y_val, True)
            print(accuracy_validation[-1])
            accuracies.append(accuracy_validation[-1])

        return np.array(accuracies).mean()

    # def split_in_batches(self, X, Y, batch_size):
    #     batches_X = []
    #     batches_Y = []
    #     for i in range(int(len(X) / batch_size + 1)):
    #         mini_X = X[i * batch_size : (i + 1) * batch_size]
    #         batches_X.append(mini_X)
    #         mini_Y = Y[i * batch_size : (i + 1) * batch_size]
    #         batches_Y.append(mini_Y)
    #
    #     return batches_X, batches_Y



