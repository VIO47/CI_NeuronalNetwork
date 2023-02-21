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
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]]

    Y_and = [0, 0, 0, 1]
    Y_or = [0, 1, 1, 1]
    Y_xor = [0, 1, 1, 0]

    nn_structure = [
        {"input_dim": 2, "output_dim": 1, "activation": "step"}
    ]

    neural_network = nn(nn_structure, 0.1)
    loss_arr_and = neural_network.train(X, Y_and)
    loss_arr_or = neural_network.train(X, Y_or)
    loss_arr_xor = neural_network.train(X, Y_xor)


    fig, axs = plt.subplots(3)
    fig.suptitle("Error over epochs")
    fig.tight_layout(pad=2.0)
    axs[0].set_title("AND gate")
    axs[0].plot(loss_arr_and, 'tab:red')
    axs[1].set_title("OR gate")
    axs[1].plot(loss_arr_or, 'tab:green')
    axs[2].set_title("XOR gate")
    axs[2].plot(loss_arr_xor, 'tab:blue')
    for ax in axs.flat:
        ax.set(xlabel='epoch', ylabel='error')
    plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
