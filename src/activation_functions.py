from abc import abstractmethod, ABC
import numpy as np


class Activation(ABC):
    """
    Base class for all activation functions
    """

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Linear(Activation):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))


class ReLU(Activation):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        if not isinstance(x, np.ndarray):
            return 0 if x < 0 else 1
        x = x.copy()
        neg = x < 0
        x[neg] = 0
        x[~neg] = 1
        return x


class Sin(Activation):
    def __call__(self, x):
        return np.sin(x)

    def derivative(self, x):
        return np.cos(x)


class Cos(Activation):
    def __call__(self, x):
        return np.cos(x)

    def derivative(self, x):
        return -np.sin(x)


class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative(self, x):
        pass
