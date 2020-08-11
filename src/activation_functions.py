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


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        if not isinstance(x, np.ndarray):
            return 0 if x < 0 else 1
        x = x.copy()
        negative_dims = x < 0
        x[negative_dims] = self.alpha
        x[~negative_dims] = 1
        return x


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)


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
