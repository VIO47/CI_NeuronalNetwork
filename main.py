# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from NeuralNetwork import NeuralNetwork as nn
from Functions import Functions as func
import matplotlib.pyplot as plt
from Perceptron import Perceptron as p


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    This function calculates and plots the confusion matrix for a neural network with 7 labels.
    y_true: list or array of true labels.
    y_pred: list or array of predicted labels.
    classes: list of class names, ordered by their corresponding numerical label.
    """

    # Calculate the confusion matrix
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        true_class = int(y_true[i]) - 1
        pred_class = int(y_pred[i])
        confusion_matrix[true_class][pred_class] += 1

    # Normalize the confusion matrix
    row_sums = confusion_matrix.sum(axis=1)
    normalized_matrix = confusion_matrix / row_sums[:, np.newaxis]
    # Plot the confusion matrix
    plt.imshow(normalized_matrix, cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add the normalized values to the plot
    threshold = normalized_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, '{:.2f}'.format(normalized_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if normalized_matrix[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.show()

def train_ANN():
    f = func()
    X = np.loadtxt("data/features.txt", dtype='f', delimiter=',')
    Y = np.loadtxt("data/targets.txt", dtype='i')
    unknown = np.loadtxt("data/unknown.txt", dtype='f', delimiter=',')
    func.normalize_data(f, X)
    func.normalize_data(f, Y)
    # X, Y = shuffle(X, Y)
    func.shuffle_arrays(f, [X, Y])
    X_train, X_test, y_train, y_test = func.train_test_split(f, X, Y)

    one_hot_encode_y_train = []
    for y in y_train:
        arr = list(np.zeros(7))
        arr[y - 1] = 1
        one_hot_encode_y_train.append(arr)
    one_hot_encode_y_train = np.array(one_hot_encode_y_train)

    one_hot_encode_y_test = []
    for y in y_test:
        arr = list(np.zeros(7))
        arr[y - 1] = 1
        one_hot_encode_y_test.append(arr)
    one_hot_encode_y_test = np.array(one_hot_encode_y_test)


    structure = [
        {"input_dim": 10, "output_dim": 32, "activation": "relu"},
        {"input_dim": 32, "output_dim": 7, "activation": "softmax"}
    ]

    ann = nn(structure, 0.01)
    accuracies_train = ann.k_fold_cross_validation(X_train, one_hot_encode_y_train, 10, 3000)
    y_hat_test, accuracies_test = ann.predict(X_test, one_hot_encode_y_test, True)
    class_names = ['1', '2', '3', '4', '5', '6', '7']
    plot_confusion_matrix(y_test, y_hat_test, class_names)

    #Predict the unknownn labels
    y_predict, accuracies_empty = ann.predict(unknown, [], False)
    file = open("predictions.txt", "w+")
    for label in y_predict:
        file.write(label)
    file.close()


    #loss_train, pred_train, acc_train = ann.train(X_train, np.array(one_hot_encode_y_train))
    #pred_test, acc_test = ann.predict(X_test, np.array(one_hot_encode_y_test))

    # plt.plot(loss_train, 'tab:red')
    # plt.plot(accuracies_train, 'tab:blue')
    # plt.show()

    plt.title("Test performance")
    plt.plot(accuracies_test, 'tab:blue')
    plt.show()

    print(accuracies_train)
    print(accuracies_test[-1])


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



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
