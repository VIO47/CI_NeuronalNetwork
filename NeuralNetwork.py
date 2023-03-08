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

    def train(self, X, y, x_val=None, y_val=None, epochs=1000):
        accuracy_train_history = []
        loss_train_history = []
        y_hat_history = []
        loss_validation_history = []
        accuracy_validation_history = []

        for i in range(epochs):
            f = Func()
            aux_X = np.copy(X)
            aux_y = np.copy(y)
            Func.shuffle_arrays(f, [aux_X, aux_y])
            aux_X = aux_X.T
            aux_y = aux_y.T
            y_hat, aux_activation, aux_intermediate = NeuralNetwork.full_forward_prop(self, aux_X)
            y_hat_val, aux_activation_val, aux_intermediate_val = NeuralNetwork.full_forward_prop(self, x_val.T)

            loss_validation = Func.cat_cross_entropy_loss(f, y_hat_val, y_val.T)
            loss_validation_history.append(loss_validation)

            loss_train = Func.cat_cross_entropy_loss(f, y_hat, aux_y)
            loss_train_history.append(loss_train)

            accuracy_validation = Func.accuracy(f, y_hat_val, y_val.T)
            accuracy_validation_history.append(accuracy_validation)

            accuracy_train = Func.accuracy(f, y_hat, aux_y)
            accuracy_train_history.append(accuracy_train)

            y_hat_history.append(y_hat)

            gradients_weights, gradients_biases = NeuralNetwork.full_back_prop(self, y_hat, aux_y, aux_activation,
                                                                     aux_intermediate)
            NeuralNetwork.update(self, gradients_weights, gradients_biases)

        return y_hat_history, accuracy_train_history, accuracy_validation_history, loss_train_history, loss_validation_history

    def predict(self, X, y, make_accuracy):
        f = Func()
        accuracy = 0
        result = X.T
        if (make_accuracy):
            y = y.T

        for layer_idx, layer in enumerate(self.structure):
            curr_idx = layer_idx + 1
            result = np.dot(self.weights[curr_idx], result) + self.bias[curr_idx]

        if (make_accuracy):
            accuracy = Func.accuracy(f, result, y)
        y_hat = np.argmax(result, axis=0)
        return y_hat, accuracy

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
        loss_train_mean = np.zeros(epochs)
        loss_validation_mean = np.zeros(epochs)
        accuracies_validation_mean = np.zeros(epochs)
        accuracies_train_mean = np.zeros(epochs)

        for i in range(k):
            print(f"Processing fold {i + 1}/{k}...")
            start = i * fold_size
            end = (i + 1) * fold_size

            x_val = X[start:end]
            y_val = y[start:end]

            x_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])

            y_hat_train, accuracy_train, accuracy_validation, loss_train, loss_validation = NeuralNetwork.train(self, x_train,
                                                                                                      y_train, x_val,
                                                                                                      y_val, epochs)

            for i in range(epochs):
                loss_train_mean[i] = loss_train_mean[i] + loss_train[i]
                loss_validation_mean[i] = loss_validation_mean[i] + loss_validation[i]
                accuracies_validation_mean[i] = accuracies_validation_mean[i] + accuracy_validation[i]
                accuracies_train_mean[i] = accuracies_train_mean[i] + accuracy_train[i]

        for i in range(len(loss_train_mean)):
            loss_train_mean[i] = loss_train_mean[i] / k
            loss_validation_mean[i] = loss_validation_mean[i] / k
            accuracies_validation_mean[i] = accuracies_validation_mean[i] / k
            accuracies_train_mean[i] = accuracies_train_mean[i] / k

        return accuracies_train_mean, accuracies_validation_mean, loss_train_mean, loss_validation_mean




