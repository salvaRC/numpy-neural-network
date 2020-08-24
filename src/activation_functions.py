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

    def __str__(self):
        return "Linear"


class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))

    def __str__(self):
        return "Sigmoid"


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

    def __str__(self):
        return f"Leaky ReLU - alpha={self.alpha}"


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)

    def __str__(self):
        return "ReLU"


class Sin(Activation):
    def __call__(self, x):
        return np.sin(x)

    def derivative(self, x):
        return np.cos(x)

    def __str__(self):
        return "Sine"


class Cos(Activation):
    def __call__(self, x):
        return np.cos(x)

    def derivative(self, x):
        return -np.sin(x)

    def __str__(self):
        return "Cosine"


class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative(self, x):
        pass

    def __str__(self):
        return "Softmax"


class Tanh(Activation):
    def __call__(self, x):
        exp_x = np.exp(x)
        exp__x = np.exp(-x)
        return (exp_x - exp__x) / (exp_x + exp__x)

    def derivative(self, x):
        return 1 - self(x) ** 2

    def __str__(self):
        return "Tanh"


class MultiActivations(Activation):
    def __init__(self, dimensions, activations):
        self.activations = []
        neurons_per_act = int(np.ceil(dimensions / len(activations)))
        prev = 0
        for act in activations[:-1]:
            self.activations.append((act, prev, prev + neurons_per_act))
            prev += neurons_per_act
        self.activations.append((activations[-1], prev, dimensions))  # take the rest

    def __call__(self, z):
        x = z.copy()
        for act_func, s, e in self.activations:
            x[s:e] = act_func(x[s:e])
        return x

    def derivative(self, z):
        x = z.copy()
        for act_func, s, e in self.activations:
            x[s:e] = act_func.derivative(x[s:e])
        return x

    def __str__(self):
        return ", ".join([str(act) for act, _, _ in self.activations])
