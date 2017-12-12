import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils_rnn
from sklearn.utils import shuffle
import gru


alpha = 0.001
max_iter = 500
n_input = 3
n_hidden = 512

class LayerRNN:
    def __init__(self, word_embedding, hidden_layer_sizes, dict_length, activation_f, session,
                 W_embed, W_in, W_out, W_rec, b_rec, b_out):
        self.D = word_embedding
        self.M = hidden_layer_sizes
        self.V = dict_length
        self.f = activation_f
        self.session = session

        self.We = W_embed
        self.Wi = W_in
        self.Wo = W_out
        self.Wr = W_rec
        self.br = b_rec
        self.bo = b_out
        self.params = [self.We, self.Wi, self.Wo, self.Wr, self.br, self.bo]


def RNN(X, W, b):
    X = tf.reshape(X, [-1, n_input])
    X = tf.split(X, n_input, 1)
    RNN_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(RNN_cell, X, dtype=tf.float32)
    return tf.matmul(outputs[-1], W) + b

if __name__ == '__main__':
    dataset = utils_rnn.load_data()
    X_num, word_dict, reverse_dict = utils_rnn.word2idx(dataset)

    offset = 0
    vocab_size = len(word_dict)
    W = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    b = tf.Variable(tf.random_normal([vocab_size]))

    X = tf.placeholder(tf.int32, [None, n_input, 1])
    y = tf.placeholder(tf.int32, [None, vocab_size])
    print(X, y)


    #numeric_input = [[word_dict[str(X[i])]] for i in range(offset, offset+n_input)]
    #symbols_onehot = np.zeros([vocab_size], dtype=float)
    #symbols_onehot[word_dict[str(X[offset+n_input])]] = 1.0
