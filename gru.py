# GATED RECURRENT UNIT CLASS
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class GRU:
    def __init__(self, input_size, output_size, activation_function):
        self.Ni = input_size
        self.No = output_size
        self.f = activation_function

        self.Wxr = tf.get_variable('Wxr', shape=[self.Ni, self.No], initializer=xavier_initializer())
        self.Whr = tf.get_variable('Whr', shape=[self.No, self.No], initializer=xavier_initializer())
        self.br  = tf.get_variable('br',  shape=[self.No], initializer=tf.zeros_initializer())
        self.Wxz = tf.get_variable('Wxz', shape=[self.Ni, self.No], initializer=xavier_initializer())
        self.Whz = tf.get_variable('Whz', shape=[self.No, self.No], initializer=xavier_initializer())
        self.bz  = tf.get_variable('bz',  shape=[self.No], initializer=tf.zeros_initializer())
        self.Wxh = tf.get_variable('Wxh', shape=[self.Ni, self.No], initializer=xavier_initializer())
        self.Whh = tf.get_variable('Whh', shape=[self.No, self.No], initializer=xavier_initializer())
        self.bh  = tf.get_variable('bh',  shape=[self.No], initializer=tf.zeros_initializer())
        self.h0  = tf.get_variable('h0',  shape=[self.No], initializer=tf.zeros_initializer())

    def recurrence(self, x_cur, h_pred):
        r  = tf.nn.sigmoid(x_cur.dot(self.Wxr) + h_pred.dot(self.Whr) + self.br)
        z  = tf.nn.sigmoid(x_cur.dot(self.Wxz) + h_pred.dot(self.Whz) + self.bz)
        h1 = self.f(x_cur.dot(self.Wxh) + (r*h_pred).dot(self.Whh) + self.bh)
        h  = (1-z)*h_pred + z*h1
        return h

    def propagate(self, x):
        h, _ = tf.scan(
            fn=self.recurrence,
            elems=x,
            initializer=self.h0,
        )
