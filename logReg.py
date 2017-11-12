import numpy as np


def get_prediction_error(y, y_hat):
    """
    :param y: actual output
    :param y_hat: predicted output
    :return: LSE between predicted and actual output
    """
    errors = (1/(2*y.shape[0])) * np.sum(np.square(y - y_hat))
    return errors


def get_predictions(x, w):
    """
    :param x: input data
    :param w: weights for the features of the regression
    :return: vector of predictions of the output variables
    """
    assert x.shape[1] == w.shape[0]
    return x.dot(w)


def get_weights_ne(x, y, l):
    """
    :param x: input data
    :param y: output data
    :param l: regularization parameter
    :return: vector of weights for LR
    """
    dot_product = x.T.dot(x)
    s = (l * np.eye(dot_product.shape[0], dot_product.shape[1]) + dot_product)
    w = np.linalg.inv(s).dot(x.T.dot(y))
    return w


def get_xy_arrays(f_name, pwr):
    """
    :param f_name: file that should be used for getting inputs and labels
    :param pwr: order of the polynomial used for the regression
    :return: array x for features of all the examples  +
             array y for the corresponding labels
    """
    x = np.array([])
    y = np.array([])
    is_first_line = True
    n = 0
    m = 0
    for line in open(f_name, 'r'):
        in_list = [int(x) for x in line.strip().split()]
        if is_first_line:
            m = len(in_list) - 1
        for i in range(pwr):
            x = np.append(x, [j ** (i+1) for j in in_list[:-1]])
        y = np.append(y, [in_list[-1]])
        n += 1
    x = x.reshape((n, m*pwr))
    x = np.insert(x, 0, [1], 1)
    y = y.reshape((y.shape[0], 1))
    return x, y


def regression(f_name, pwr):
    """
    :param f_name: file that should be used for getting inputs and labels
    :param pwr: order of the polynomial used for the regression
    :return: pred - vector of predictions of the output variables
             lse - least square error for the prediction
             y - actual output variables
    """
    # PREPARE THE VARIABLES FOR INPUTS AND LABELS
    x, y = get_xy_arrays(f_name, pwr)

    # GET WEIGHTS FROM THE NORMAL EQUATION
    lambd = 0  # case with no regularization
    weights = get_weights_ne(x, y, lambd)

    # GET PREDICTION FOR THE OUTPUT
    pred = get_predictions(x, weights)

    # CHECK PREDICTION ERROR
    lse = get_prediction_error(y, pred)
    return pred, lse, y


if __name__ == "__main__":
    file_name = "data/electrn.dat"

    # CHECK WHAT ORDER OF POLYNOMIAL WILL GIVE BETTER RESULT
    min_pwr = 1
    max_pwr = 10
    min_error = -1
    best_pwr = min_pwr
    best_predictions = np.array([])
    y = np.array([])
    for i in range(min_pwr, max_pwr):
        predictions, prediction_error, y = regression(file_name, i)
        if min_error == -1:
            min_error = prediction_error
            best_predictions = predictions
        elif prediction_error < min_error:
            min_error = prediction_error
            best_pwr = i
            best_predictions = predictions

    # PRINT RESULT
    print("The best order of polynomial to use for the regression is ", best_pwr)
    print("LSE: ", min_error)
    print("Predictions from the linear regression vs actual outputs:")
    for i in range(best_predictions.shape[0]):
        print("Prediction: ", best_predictions[i][0], "; actual output: ", y[i][0])
