import numpy as np
from src.activation_functions import ReLU, Softmax, LeakyReLU
from src.evaluation import plot_loss_and_accuracy, accuracy
from src.loss_functions import SquaredLoss
from src.neural_net.layers import InputLayer, Layer
from src.neural_net.network import NeuralNetwork
from src.preprocessing import to_categorical, split


X, Y = np.loadtxt('small_mnist/mnist_small_train_in.txt', delimiter=','),\
       np.loadtxt('small_mnist/mnist_small_train_out.txt', delimiter=',')

Xtest, Ytest = np.loadtxt('small_mnist/mnist_small_test_in.txt', delimiter=','),\
               np.loadtxt('small_mnist/mnist_small_test_out.txt', delimiter=',')

(X, Y), (Xval, Yval) = split(X, Y, train_frac=0.9)
Y = to_categorical(Y)

hidden_layer_act = LeakyReLU(alpha=0.01)
layers = [InputLayer(X.shape[1]),
          Layer(25, hidden_layer_act),
          Layer(25, hidden_layer_act),
          Layer(10, Softmax())]

nn = NeuralNetwork(layers)
nn.compile(SquaredLoss(), metric=accuracy)
history = nn.fit(X, Y, Xval, Yval, learning_rate=0.003, n_epochs=100, batch_size=16)
nn.print_predict(Xtest, Ytest)
plot_loss_and_accuracy(history)
