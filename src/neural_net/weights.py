import numpy as np
from src.neural_net.layers import Layer


class Weight:
    def __init__(self, prev_layer: Layer, next_layer: Layer):
        self.previous = prev_layer
        self.next = next_layer
        self.dim_out, self.dim_in = next_layer.n_neurons, prev_layer.n_neurons
        self.W = np.random.randn(self.dim_out, self.dim_in) * 0.1
        self.bias = np.random.randn(self.dim_out, 1) * 0.1
        self.gradient_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)
        self.activation_func = self.next.activation_func

    def forward(self, x):
        self.previous.activation = x
        pre_activation = self.W @ x + self.bias
        self.next.pre_activation = pre_activation
        self.next.activation = self.activation_func(pre_activation)
        return self.next.activation

    def backward(self, lr):
        self.bias_acc += lr * self.next.gradient
        self.gradient_acc += lr * np.outer(self.next.gradient, self.previous.activation)

    def update(self, normalization):
        self.W = self.W - self.gradient_acc / normalization
        self.bias = self.bias - self.bias_acc / normalization
        self.gradient_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)

