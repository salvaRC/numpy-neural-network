from abc import abstractmethod, ABC
import numpy as np


class Loss(ABC):
    @abstractmethod
    def __call__(self, true_labels, predictions):
        pass

    @abstractmethod
    def derivative(self, true_labels, predictions):
        pass


class SquaredLoss(Loss):
    def __call__(self, true_labels, predictions):
        diff = true_labels - predictions
        loss = diff.T @ diff
        return loss / 2

    def derivative(self, true_labels, predictions):
        return predictions - true_labels  # turn this around and do gradient ASCENT, works too ;)


class CrossEntropy(Loss):
    def __init__(self):
        from src.activation_functions import Softmax
        self.softmax = Softmax()

    def __call__(self, true_labels, predictions):
        y = true_labels.argmax(axis=1) if true_labels.ndim > 1 else true_labels
        m = y.shape[0]
        p = self.softmax(predictions)
        log_likelihood = -np.log(p[list(range(m)), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def derivative(self, true_labels, predictions):
        y = true_labels.argmax(axis=1) if true_labels.ndim > 1 else true_labels
        m = y.shape[0]
        grad = self.softmax(predictions)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad
