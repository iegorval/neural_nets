import tensorflow as tf
import utils

# global variables
num_of_examples = 400
num_of_train = 300
num_of_test = num_of_examples - num_of_train
num_of_features = 2
alpha = 1


def initialize_params():
    w = tf.get_variable('W', [1, num_of_features],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', [1, 1],
                        initializer=tf.zeros_initializer())
    return w, b


def linear_function(x, w, b):
    x = tf.reshape(x, [num_of_features, 1])
    y_est = tf.add(tf.matmul(w, x), b)
    return y_est


if __name__ == '__main__':
    # trainSetX, trainSetY = utils.load_train_set()
    # testSetX, testSetY = utils.load_test_set()
    sample_data = utils.linear_sep_data(num_of_examples)
    train_data = sample_data[:num_of_train]
    test_data = sample_data[num_of_train:]
    X_train = train_data.T[:2]
    y_train = train_data.T[2]
    X_test = test_data.T[:2]
    y_test = test_data.T[2]
    X_train = X_train.reshape((num_of_train, 2))
    y_train = y_train.reshape((num_of_train, 1))
    X_test = X_test.reshape((num_of_test, 2))
    y_test = y_test.reshape((num_of_test, 1))

    # create placeholders & variables
    X = tf.placeholder(tf.float32, shape=(num_of_features,))
    y = tf.placeholder(tf.float32, shape=(1,))
    W, b = initialize_params()

    # predict y
    y_estim = linear_function(X, W, b)
    y_estim = tf.reshape(y_estim, [1, ])
    y_pred = tf.sigmoid(y_estim)

    # set the optimizer
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_mean)

    # training phase
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for idx in range(num_of_train):
            cur_x, cur_y = X_train[idx], y_train[idx]
            _, c = sess.run([optimizer, loss_mean], feed_dict={X: cur_x, y: cur_y})
