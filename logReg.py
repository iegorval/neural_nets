import nn
import utils

# global variables
num_of_examples = 100
num_of_features = 3197
num_iterations = 100
alpha = 0.001

if __name__ == '__main__':
    X, y = utils.load_train_set(num_of_examples)
    X = utils.normalize_features(X)
    X = X.T
    y = y.reshape((1, num_of_examples))
    W, b = nn.zero_initializer(num_of_features)
    optimal_params, costs = nn.gradient_descent_optimizer(X, y, W, b, alpha, num_iterations)
    print(costs)
    
