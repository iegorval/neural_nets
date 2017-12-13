import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import utils_rnn
from random import randint
import os


alpha = 0.001
n_input = 3
n_hidden = 512
num_epochs = 1500
step = 1000




def RNN(x, W, b):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x,n_input,1)
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], W) + b


if __name__ == '__main__':
    data_set = utils_rnn.load_data()
    X_num, word_dict, reverse_dict = utils_rnn.word2idx(data_set)
    N = len(X_num)
    vocab_size = len(word_dict)

    W = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    b = tf.Variable(tf.random_normal([vocab_size]))

    # initialize placeholders for input/output
    X = tf.placeholder(tf.float32, [None, n_input, 1])
    y = tf.placeholder(tf.float32, [None, vocab_size])

    # predict and compare
    y_prediction = RNN(X, W, b)
    correct_pred = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # cost and optimizer
    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(cost_op)

    # create a saver
    saver = tf.train.Saver()

    # initialize
    init = tf.global_variables_initializer()

    # training phase
    with tf.Session() as session:
        total_cost = 0
        total_correct = 0
        total_iter_step = 0
        current_step = 0

        session.run(init)
        costs = []

        for epoch in range(num_epochs):
            for i in range(N):
                n_whole_example = len(X_num[i])
                if n_whole_example >= 4:
                    current_step += 1
                    offset = randint(0, n_whole_example-n_input-1)
                    cur_example = X_num[i][offset:offset+n_input]
                    cur_example = np.reshape(np.array(cur_example), [-1, n_input, 1])

                    one_hot = np.zeros([vocab_size], dtype=float)
                    one_hot[X_num[i][offset+n_input]] = 1.0
                    one_hot = np.reshape(one_hot, [1, -1])

                    _, acc, cost, prediction = session.run([optimizer, accuracy, cost_op, y_prediction],
                                                          feed_dict={X:cur_example, y:one_hot})
                    total_cost += cost
                    total_correct += acc
                    total_iter_step += 1
                    if total_iter_step % step == 0:
                        step_accuracy = (total_correct/step) * 100
                        step_cost =  total_cost/step
                        print("Iteration:", total_iter_step, "; Average Accuracy:", step_accuracy, "%; Average Cost:", step_cost)
                        costs.append(step_cost)
                        total_correct = 0
                        total_cost = 0
        saver.save(session, os.path.join(os.getcwd(), 'RNN-model'))

    plt.plot(costs)
    plt.show()
