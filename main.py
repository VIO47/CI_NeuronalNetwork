# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork as nn
import matplotlib.pyplot as plt
from Perceptron import Perceptron as p
from NN import ANN as ANN

def train_ANN():
 X = np.loadtxt("data/features.txt", dtype = 'f', delimiter = ',')
 Y = np.loadtxt("data/targets.txt", dtype = 'i')
 X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

 ann_struct = [[p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu"), p(10, 9, "relu")],
               [p(9, 8, "relu"), p(9, 8, "relu"), p(9, 8, "relu"), p(9,8 , "relu"), p(9, 8, "relu"), p(9, 8, "relu"), p(9, 8, "relu"), p(9, 8, "relu")],
               [p(8, 7, "softmax"), p(8, 7, "softmax"), p(8, 7, "softmax"), p(8, 7, "softmax"), p(8, 7, "softmax"), p(8, 7, "softmax"), p(8, 7, "softmax")]]
 ann = ANN(ann_struct, 0.1)
 X_train = X_train.reshape([X_train.shape[0], -1])
 #X_val = X_test.reshape([X_val.shape[0], -1])
 X_test = X_test.reshape([X_test.shape[0], -1])
 accuracy =  ann.train(X_train, y_train)

 fig, axs = plt.subplot
 axs[0].set_title("ANN Training")
 axs[0].plot(accuracy, 'tab:red')
def train_test():
    #X = np.loadtxt("features.txt", dtype = 'i', delimiter = ',')
    #Y = np.loadtxt("targets.txt", dtype = 'i')
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    X = [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]]

    Y_and = [0, 0, 0, 1]
    Y_or = [0, 1, 1, 1]
    Y_xor = [0, 1, 1, 0]

   # nn_structure = [
   #     {p(0.1, 2, 1, "step")}
    #]

    neural_network = p(0.1, 2, 1, "step")
    accuracy_and = neural_network.train(X, Y_and)
    accuracy_or = neural_network.train(X, Y_or)
    accuracy_xor = neural_network.train(X, Y_xor)


    fig, axs = plt.subplots(3)
    fig.suptitle("Accuracy over epochs")
    fig.tight_layout(pad=2.0)
    axs[0].set_title("AND gate")
    axs[0].plot(accuracy_and, 'tab:red')
    axs[1].set_title("OR gate")
    axs[1].plot(accuracy_or, 'tab:green')
    axs[2].set_title("XOR gate")
    axs[2].plot(accuracy_xor, 'tab:blue')
    for ax in axs.flat:
        ax.set(xlabel='epoch', ylabel='accuracy')
    plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_ANN()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
