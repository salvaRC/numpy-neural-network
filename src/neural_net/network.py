import numpy as np
from src.neural_net.weights import Weight
from src.preprocessing import shuffle


class NeuralNetwork:
    def __init__(self, layers: list):
        self.metric = None
        self.layers = layers
        self.weights = []
        self.y = None
        self.loss_f = None
        self.loss = -1
        self.lr = 0.01
        self.batch_size = None
        for i, layer in enumerate(self.layers[:-1]):
            weight = Weight(layer, self.layers[i + 1])
            self.weights.append(weight)
            layer.next_layer = self.layers[i + 1]
            layer.weight = weight

    def forward(self, x):
        activation = x
        for weight in self.weights:
            activation = weight.forward(activation)
        return activation

    def backprop(self):
        assert self.loss_f is not None and self.y is not None
        preds = self.layers[-1].activation
        loss = self.loss_f.derivative(self.y, preds)

        self.layers[-1].gradient = loss
        self.loss += float(self.loss_f(self.y, preds))
        rev_layers = list(reversed(self.layers))
        for layer in rev_layers[1:-1]:
            layer.backward()
        for weight in self.weights:
            weight.backward(self.lr)

    def compile(self, loss_function, metric):
        self.loss_f = loss_function
        self.metric = metric

    def update(self):
        for weight in self.weights:
            weight.update(self.batch_size)

    def fit(self, X, Y, X_val=None, Y_val=None, learning_rate=0.01, n_epochs=25, batch_size=64):
        self.lr = learning_rate
        self.batch_size = batch_size
        num_samples = X.shape[0]
        train_losses = []
        val_losses = []
        val_metrics = []
        for epoch in range(n_epochs):
            X, Y = shuffle(X, Y)
            self.loss = 0
            for i, (x, y) in enumerate(zip(X, Y)):
                self.y = y.reshape(-1, 1)
                self.forward(x.reshape(-1, 1))
                self.backprop()
                if i % batch_size == 0 and i != 0:
                    self.update()
            self.loss /= num_samples

            if X_val is not None and Y_val is not None:
                tr_loss, val_loss, val_metric = self.print_predict(X_val, Y_val, epoch=epoch)
                train_losses.append(tr_loss)
                val_losses.append(val_loss)
                val_metrics.append(val_metric)
            else:
                train_losses.append(self.loss)
                print("Epoch {} - Loss = {:.3f}".format(epoch, self.loss))
        return {"Train_loss": train_losses, "Val_loss": val_losses,
                "Val_performance": val_metrics, "Epochs": list(range(n_epochs))}

    def print_predict(self, X, Y, epoch=-1):
        if X is None or Y is None:
            return
        train_loss = self.loss
        preds = self.predict(X.T)
        val_loss = np.sum(self.loss_f(Y, preds)) / Y.shape[0]
        metric_val = self.metric(Y, preds)
        print("Epoch {} - Train loss = {:.3f} Validation Loss = {:.3f}, performance: {:.4f}"
              .format(epoch, train_loss, val_loss, metric_val))
        return train_loss, val_loss, metric_val

    def predict_proba(self, X):
        probs = self.forward(X)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=0)
        return preds
