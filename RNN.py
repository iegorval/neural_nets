import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils_rnn
from sklearn.utils import shuffle




def RNN(X, W, b):
    X = tf.reshape(X, [-1, n_input])
    X = tf.split(X, n_input, 1)
    RNN_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(RNN_cell, X, dtype=tf.float32)
    return tf.matmul(outputs[-1], W) + b

if __name__ == '__main__':
    X = utils_rnn.load_data()
    _, word_dict, reverse_dict = utils_rnn.word2idx(X)

    n_hidden = 3
    n_input = 3
    vocab_size = len(word_dict)
    W = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    b = tf.Variable(tf.random_normal([vocab_size]))

    numeric_input = [[word_dict[str(X[i])]] for i in range(offset, offset+n_input)
    symbols_onehot = np.zeros([vocab_size], dtype=float)
    symbols_onehot[word_dict[str(X[offset+n_inpu])]] = 1.0
