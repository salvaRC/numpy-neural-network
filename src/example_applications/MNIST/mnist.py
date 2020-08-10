import numpy as np

from src.activation_functions import ReLU, Softmax, Sin, Cos
from src.evaluation import plot_loss_and_accuracy, accuracy
from src.loss_functions import SquaredLoss, CrossEntropy
from src.neural_net.layers import InputLayer, Layer
from src.neural_net.network import NeuralNetwork
from src.preprocessing import to_categorical, split
from keras.datasets import mnist

(X, Y), (Xtest, Ytest) = mnist.load_data()
# Reshape & Normalize
dimensions = X.shape[1] * X.shape[2]   # i.e. flattened
X = X.reshape((X.shape[0], dimensions)) / 255.0
Xtest = Xtest.reshape((Xtest.shape[0], dimensions)) / 255.0
(X, Y), (Xval, Yval) = split(X, Y, train_frac=0.9)

Y = to_categorical(Y)

hidden_layer_act = ReLU()
layers = [InputLayer(X.shape[1]),
          Layer(25, hidden_layer_act),
          Layer(25, hidden_layer_act),
          Layer(10, Softmax())]

nn = NeuralNetwork(layers)
nn.compile(loss_function=SquaredLoss(), metric=accuracy)
history = nn.fit(X, Y, Xval, Yval, learning_rate=0.01, n_epochs=100, batch_size=32)
nn.print_predict(Xtest, Ytest)
plot_loss_and_accuracy(history)
