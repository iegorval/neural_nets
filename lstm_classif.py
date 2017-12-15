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
max_iterations = 50000
step = 1000

class WordPredictor():
    def __init__(self, fname):
        self.get_data(fname)
        self.W = tf.Variable(tf.random_normal([n_hidden, self.vocab_size]))
        self.b = tf.Variable(tf.random_normal([self.vocab_size]))

    def get_data(self, fname):
        data_set = utils_rnn.load_data(fname)
        self.X_num, self.word_dict, self.reverse_dict = utils_rnn.word2idx(data_set)
        self.N = len(self.X_num)
        self.vocab_size = len(self.word_dict)

    def predict(self, x, W, b):
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(x,n_input,1)
        rnn_cell = rnn.BasicLSTMCell(n_hidden)
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], W) + b

    def build(self):
        # initialize placeholders for input/output
        self.X = tf.placeholder(tf.float32, [None, n_input, 1])
        self.y = tf.placeholder(tf.float32, [None, self.vocab_size])

        # predict and compare
        self.y_prediction = self.predict(self.X, self.W, self.b)
        correct_pred = tf.equal(tf.argmax(self.y_prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # cost and optimizer
        self.cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_prediction, labels=self.y))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(self.cost_op)

        # create a saver
        self.saver = tf.train.Saver()

    def train(self):
        # initialize
        init = tf.global_variables_initializer()

        # training phase
        with tf.Session() as session:
            total_cost = 0
            total_correct = 0
            total_iter_step = 1
            current_step = 0

            session.run(init)
            costs = []

            while(total_iter_step <= max_iterations):
                for i in range(self.N):
                    n_whole_example = len(self.X_num[i])
                    if n_whole_example >= 4:
                        current_step += 1
                        offset = randint(0, n_whole_example - n_input - 1)
                        cur_example = self.X_num[i][offset:offset + n_input]
                        cur_example = np.reshape(np.array(cur_example), [-1, n_input, 1])

                        one_hot = np.zeros([self.vocab_size], dtype=float)
                        one_hot[self.X_num[i][offset + n_input]] = 1.0
                        one_hot = np.reshape(one_hot, [1, -1])

                        _, acc, cost, prediction = session.run([self.optimizer, self.accuracy, self.cost_op, self.y_prediction],
                                                               feed_dict={self.X: cur_example, self.y: one_hot})
                        total_cost += cost
                        total_correct += acc
                        total_iter_step += 1
                        if total_iter_step % step == 0:
                            step_accuracy = (total_correct / step) * 100
                            step_cost = total_cost / step
                            print("Iteration:", total_iter_step, "; Average Accuracy:", step_accuracy,
                                  "%; Average Cost:", step_cost)
                            costs.append(step_cost)
                            total_correct = 0
                            total_cost = 0
            self.saver.save(session, os.path.join(os.getcwd(), 'RNN-model'))
        plt.plot(costs)
        plt.show()


if __name__ == '__main__':
    wp = WordPredictor('data/story.txt')
    wp.build()
    wp.train()
