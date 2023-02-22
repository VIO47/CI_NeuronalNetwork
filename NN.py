import numpy as np
from Perceptron import Perceptron as p
class ANN:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward_propag(self, X):
        A_curr = X
        memory_A = []
        memory_Z = []
        for layer in self.layers:
            auxiliar_Z = []
            auxiliar_A = []
            for output in range(len(layer)):
                activation, new_Z = layer[output].perceptron_forward_prop(A_curr, layer[output].weights, layer[output].bias, layer[output].activation_function, output)
                auxiliar_Z.append(new_Z)
                auxiliar_A.append(activation)
            if layer[output].output_size == 7:
                memory_A.append(auxiliar_A)
                memory_Z.append(auxiliar_Z)
                return memory_A, memory_Z
            else:
                print(auxiliar_A)
                memory_A.append(auxiliar_A)
                memory_Z.append(auxiliar_Z)
                A_curr = auxiliar_A

    def loss(self, y_hat, y):
        loss = 0
        for (computed, target) in zip(y_hat, y):
            loss -= target * np.log(computed)
        return loss

    def backward_propag(self, memory_A, memory_Z, Y):
        reverted = self.layers.reverse()
        for index_layer in reversed(range(len(self.layers) -1, 1)):
            error = ANN.loss(self, memory_A[index_layer], Y)
            p_curr = self.layers[index_layer - 1]
            for perceptron_index in range(len(p_curr)):
                perceptron = p_curr[perceptron_index]
                dW_curr, db_curr = perceptron.perceptron_back_propag(error, memory_Z[index_layer], memory_A[index_layer])
                perceptron.update(dW_curr, db_curr, self.learning_rate)

    def accuracy(self, y_hat, y):
        acc = 0
        for (real, pred) in zip(y, y_hat):
            print(pred)
            if(pred == real):
                acc += 1
        return acc/len(y_hat)

    def train(self, X, Y, epochs = 5):
        list = []
        for epoch in range(epochs):
            memory_A, memory_Z = ANN.forward_propag(self, X)
            #print(len(memory_A[len(self.layers) - 1][0]))
            predicted = max(memory_A[len(self.layers) - 1])
            accuracy = ANN.accuracy(self, predicted, Y)
            ANN.backward_propag(self, memory_A, memory_Z, Y)
            list.append(accuracy)
        return list
