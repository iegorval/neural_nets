import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def zero_initializer(num_of_features):
    w = np.zeros((num_of_features, 1))
    b = 0
    return w, b


def log_cost(a, y):
    m = y.shape[1]
    j = (-1 / m) * (np.log(a).dot(y.T) + np.log(1 - a).dot((1 - y).T))
    return j


def linear_function(X, W, b):
    y_estim = W.T.dot(X) + b
    return y_estim


def forward_prop_1layer(W, b, X, y):
    a = sigmoid(linear_function(X, W, b))
    cost = np.squeeze(log_cost(a, y))
    return a, cost


def backward_prop_1layer(X, a, y):
    m = X.shape[1]
    dW = (1 / m) * (X.dot((a - y).T))
    db = (1 / m) * (np.sum(a - y))
    return dW, db


def gradient_descent_optimizer(X, y, W, b, alpha, num_iterations):
    costs = []
    i = 1
    for i in range(num_iterations):
        a, cost = forward_prop_1layer(W, b, X, y)
        dW, db = backward_prop_1layer(X, a, y)
        # update rule
        W -= alpha * dW
        b -= alpha * db
        costs.append(cost)
    params = (W, b, dW, db)
    return params, costs


def predict(W, b, X):
    m = X.shape[1]
    y_pred = np.zeros((1, m))
    # W = W.reshape(X.shape[0], 1)
    a = sigmoid(linear_function(X, W, b))
    for i in range(a.shape[1]):
        if a[0, i] > 0.5:
            y_pred[0, i] = 1
        else:
            y_pred[0, i] = 0
