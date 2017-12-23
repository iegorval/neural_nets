import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import utils_rnn
import os

alpha = 0.001
n_input = 3
n_units = 512
max_iterations = 30000
step = 300

class SignalPredictor():
    def __init__(self):
        #self.data = utils_rnn.toy_problem(500)
        data = []
        with open("data/HENON.DAT", 'r') as f:
            for line in f:
                format_line = float(line.strip('\n'))
                data.append(format_line)
        self.data = np.array(data)
        self.n_data = self.data.shape[0]
        self.W = tf.Variable(tf.random_normal([n_units, 1]))
        self.b = tf.Variable(tf.random_normal([1]))

    def predict(self, x, W, b):
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(x,n_input,1)
        self.rnn_cell = rnn.BasicLSTMCell(n_units, reuse=tf.AUTO_REUSE)
        outputs, states = rnn.static_rnn(self.rnn_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], W) + b

    def build(self):
        # initialize placeholders for input/output
        self.X = tf.placeholder(tf.float32, [None, n_input, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])

        # predict
        self.y_prediction = self.predict(self.X, self.W, self.b)

        # cost and optimizer
        self.cost_op = tf.reduce_sum(tf.pow(self.y_prediction-self.y, 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(self.cost_op)

        # create a saver
        self.saver = tf.train.Saver()

    def train(self):
        # prepare graphs
        fig, (cost_gr, sin_gt) = plt.subplots(2, 1)
        #sin_gt.plot(self.data, 'r', label="Given data")
        sin_gt.plot(self.data[:501], 'r', label="Given data")

        # initialize
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # training phase
            total_cost = 0
            total_iter_step = 1
            current_step = 0

            session.run(init)
            costs = []

            while(total_iter_step <= max_iterations):
                for i in range(self.n_data-(n_input+1)):
                    current_step += 1
                    cur_example = self.data[i:i+n_input]
                    cur_example = np.reshape(np.array(cur_example), [-1, n_input, 1])
                    next_value = self.data[i+n_input]
                    next_value = np.reshape(np.array(next_value), [1, -1])

                    _, cost, prediction, W_opt, b_opt = session.run([self.optimizer, self.cost_op, self.y_prediction, self.W, self.b],
                                                           feed_dict={self.X: cur_example, self.y: next_value})
                    total_cost += cost
                    total_iter_step += 1
                    if total_iter_step % step == 0:
                        step_cost = total_cost / step
                        print("Iteration:", total_iter_step, "; Average Cost:", step_cost)
                        costs.append(step_cost)
                        total_cost = 0
                    if total_iter_step > max_iterations:
                        break
            # self.saver.save(session, os.path.join(os.getcwd(), 'RNN-model'))
            self.W_opt = np.array(W_opt)
            self.b_opt = np.array(b_opt)
            cost_gr.plot(list(range(1, max_iterations, step)), costs)
            #cost_gr.title("Cost for sin prediction")
            cost_gr.set_xlabel("Iterations")
            cost_gr.set_ylabel("Cost")

            # testing phase
            print("Start testing...")
            cur_X = np.array(self.data[:n_input])
            cur_X = np.reshape(np.array(cur_X), [-1, n_input, 1])
            tmp = self.data[n_input]
            tmp = np.reshape(np.array(tmp), [1, -1])
            predicted_signal = cur_X
            #for i in range(self.n_data-(n_input+1)):
            for i in range(500):
                cost, prediction = session.run([self.cost_op, self.y_prediction], feed_dict={self.X: cur_X, self.y: tmp})
                cur_X = np.reshape(np.append(cur_X[0][1:], np.array(self.data[i+n_input])), [-1, n_input, 1])
                #cur_X = np.reshape(np.append(cur_X[0][1:], np.array(prediction)), [-1, n_input, 1])
                predicted_signal = np.append(predicted_signal, np.array(prediction))
            sin_gt.plot(predicted_signal, 'b', label="Predicted data")
            sin_gt.legend(loc = "upper right")
            plt.show()
            print("The end.")

    def test(self):
        #session = self.session
        pass


if __name__ == '__main__':
    sp = SignalPredictor()
    sp.build()
    sp.train()
