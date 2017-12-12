import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils_rnn
from random import randint


alpha = 0.001
n_input = 3
n_hidden = 512
num_epochs = 100
step = 5


def RNN(X, W, b):
    X = tf.reshape(X, [-1, n_input])
    X = tf.split(X, n_input, 1)
    RNN_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(RNN_cell, X, dtype=tf.float32)
    pred = tf.matmul(outputs[-1], W) + b
    return pred


if __name__ == '__main__':
    data_set = utils_rnn.load_data()
    X_num, word_dict, reverse_dict = utils_rnn.word2idx(data_set)
    N = len(X_num)
    vocab_size = len(word_dict)

    W = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    b = tf.Variable(tf.random_normal([vocab_size]))

    # initialize placeholders for input/output
    X = tf.placeholder(tf.float32, [n_input])
    y = tf.placeholder(tf.float32, [vocab_size])

    # predict and compare
    y_prediction = RNN(X, W, b)
    predicted_num = tf.argmax(y_prediction, 0)
    expected_num = tf.argmax(y, 0)
    correct_prediction = tf.equal(predicted_num, expected_num)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # cost and optimizer
    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(cost_op)

    # initialize
    init = tf.global_variables_initializer()

    # training phase
    with tf.Session() as session:
        total_cost = 0
        total_accuracy = 0
        session.run(init)
        costs = []
        for epoch in range(num_epochs):
            for i in range(N):
                n_whole_example = len(X_num[i])
                offset = randint(0, n_whole_example-n_input-1)
                cur_example = X_num[i][offset:offset+n_input]
                one_hot = np.zeros([vocab_size], dtype=float)
                one_hot[X_num[i][offset + n_input]] = 1.0
                _, acc, cost, prediction = session.run([optimizer, accuracy, cost_op, y_prediction],
                                                          feed_dict={X:cur_example, y:one_hot})
                #total_cost += cost
                #total_accuracy += acc
                #if (epoch % step == 0) and (i==N-1):
                #    step_accuracy = total_accuracy/(step+N)
                #    step_cost =  total_cost/(step+N)
                #    print("Accuracy:", step_accuracy, "Cost:", step_cost)
                #    costs.append(step_cost)
                #    total_accuracy = 0
                #    total_cost = 0

    #plt.plot(costs)
    #plt.show()
