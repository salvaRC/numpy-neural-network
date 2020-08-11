import numpy as np
from src.neural_net.layers import Layer


class Weight:
    def __init__(self, prev_layer: Layer, next_layer: Layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.dim_out, self.dim_in = next_layer.n_neurons, prev_layer.n_neurons
        self.W = np.random.randn(self.dim_out, self.dim_in) * 0.1
        self.bias = np.random.randn(self.dim_out, 1) * 0.1
        self.gradient_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)
        self.activation_func = self.next_layer.activation_func

    def forward(self, x):
        self.prev_layer.activation = x
        pre_activation = self.W @ x + self.bias
        self.next_layer.pre_activation = pre_activation
        self.next_layer.activation = self.activation_func(pre_activation)
        return self.next_layer.activation

    def backward(self, lr):
        self.bias_acc += lr * self.next_layer.gradient
        self.gradient_acc += lr * np.outer(self.next_layer.gradient, self.prev_layer.activation)

    def update(self, normalization):
        self.W = self.W - self.gradient_acc / normalization
        self.bias = self.bias - self.bias_acc / normalization
        self.gradient_acc = np.zeros(self.W.shape)
        self.bias_acc = np.zeros(self.bias.shape)

