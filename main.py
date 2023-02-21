# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork as nn
import matplotlib.pyplot as plt

def train_test():
    #X = np.loadtxt("features.txt", dtype = 'i', delimiter = ',')
    #Y = np.loadtxt("targets.txt", dtype = 'i')
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    X = [[0.0, 0.0],
         [0, 1],
         [1, 0],
         [1, 1]]

    Y_and = [0, 0, 0, 1]
    Y_or = [0, 1, 1, 1]
    Y_xor = [0, 1, 1, 0]

    nn_structure = [
        {"input_dim": 2, "output_dim": 1, "activation": "step"}
    ]

    neural_network = nn(nn_structure, 0.1)
    loss_arr = neural_network.train(X, Y_and)
    plt.plot(loss_arr)
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
