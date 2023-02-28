# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from NeuralNetwork import NeuralNetwork as nn
from Functions import Functions as func
import matplotlib.pyplot as plt
from Perceptron import Perceptron as p

def train_ANN():
    f = func()
    X = np.loadtxt("data/features.txt", dtype='f', delimiter=',')
    Y = np.loadtxt("data/targets.txt", dtype='i')
    # X, Y = shuffle(X, Y)
    func.shuffle_arrays(f, [X, Y])
    X_train, X_valid, X_test, y_train, y_valid, y_test = func.train_test_valid_split(f, X, Y)

    one_hot_encode_y_train = []
    for y in y_train:
        arr = list(np.zeros(7))
        arr[y - 1] = 1
        one_hot_encode_y_train.append(arr)

    one_hot_encode_y_test = []
    for y in y_test:
        arr = list(np.zeros(7))
        arr[y - 1] = 1
        one_hot_encode_y_test.append(arr)


    structure = [
        {"input_dim": 10, "output_dim": 9, "activation": "relu"},
        {"input_dim": 9, "output_dim": 8, "activation": "relu"},
        {"input_dim": 8, "output_dim": 7, "activation": "softmax"}
    ]

    ann = nn(structure, 0.1)
    loss_train, pred_train, acc_train = ann.train(X_train, np.array(one_hot_encode_y_train), 3)
    pred_test, acc_test = ann.predict(X_test, np.array(one_hot_encode_y_test))

    plt.plot(loss_train, 'tab:red')
    plt.plot(acc_train, 'tab:blue')
    plt.show()

    plt.plot(acc_test, 'tab:blue')
    plt.show()

    print(acc_train[-1])
    print(acc_test[-1])


def train_test():

    X = [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]]

    Y_and = [0, 0, 0, 1]
    Y_or = [0, 1, 1, 1]
    Y_xor = [0, 1, 1, 0]

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
    # f = func()
    # a = [1, 2, 3, 4]
    # b = [[1, 2], [3, 4], [5, 6], [7, 8]]
    # func.shuffle_arrays(f, [a, b])
    # print(a)
    # print(b)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
