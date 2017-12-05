import nn
import utils
import matplotlib.pyplot as plt

# global variables
num_of_examples = 400
num_of_train = 300
num_of_test = num_of_examples - num_of_train
num_of_features = 2
num_iterations = 100
alpha = 1

if __name__ == '__main__':
    sample_data = utils.linear_sep_data(num_of_examples)
    train_data = sample_data[:num_of_train]
    test_data = sample_data[num_of_train:]
    plt.plot(train_data.T[0], train_data.T[1], 'ro')
    plt.show()
    # X, y = utils.load_train_set(num_of_examples)
    X = train_data.T[:2]
    y = train_data.T[2]
    X_test = test_data.T[:2]
    y_test = test_data.T[2]
    X = X.reshape((2, num_of_train))
    y = y.reshape((1, num_of_train))
    X_test = X_test.reshape((2, num_of_test))
    y_test = y_test.reshape((1, num_of_test))
    # X  = utils.normalize_features(X)
    # X = X.T
    # y = y.reshape((1, num_of_examples))
    W, b = nn.zero_initializer(num_of_features)
    optimal_params, costs = nn.gradient_descent_optimizer(X, y, W, b, alpha, num_iterations)
    W_opt = optimal_params[0]
    b_opt = optimal_params[1]
    print(costs)
    y_pred = nn.predict(W_opt, b_opt, X_test)
    print('real labels', y_test)
    print('pred labels', y_pred)
    
