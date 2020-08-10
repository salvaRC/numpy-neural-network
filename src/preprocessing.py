import numpy as np


def to_categorical(Y):
    n_classes = len(np.unique(Y))
    Y_cat = np.zeros((Y.shape[0], n_classes))
    for i, y in enumerate(Y):
        Y_cat[i, int(y)] = 1
    return Y_cat


def shuffle(x, y, seed=77):
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(x))
    rng.shuffle(permutation)
    return x[permutation], y[permutation]


def split(X, Y, train_frac=0.8, seed=77):
    X, Y = shuffle(X, Y, seed=seed)
    upto = int(train_frac * X.shape[0])
    return (X[:upto, :], Y[:upto]), (X[upto:, :], Y[upto:])
