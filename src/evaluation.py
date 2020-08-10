import numpy as np
import matplotlib.pyplot as plt


def accuracy(Ytrue, preds):
    num_data_points = Ytrue.shape[0]
    correct_predictions = np.count_nonzero(Ytrue == preds)
    accuracy = correct_predictions / num_data_points
    return accuracy


def plot_loss_and_accuracy(history):
    xaxis = history['Epochs']
    plt.plot(xaxis, history['Train_loss'], label="Train loss")
    plt.plot(xaxis, history['Val_loss'], label="Validation loss")
    plt.plot(xaxis, history['Val_performance'], label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylim(top=1, bottom=0)
    plt.show()
