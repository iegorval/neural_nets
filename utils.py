import tensorflow as tf
import numpy as np


def read_file(f_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    # read - reads a single line from the file
    key, value = reader.read(f_queue)
    # default values (also seem to determine the type of the column)
    record_defaults = [[1.0] for j in range(3198)]
    record_defaults[0][0] = 1
    # decode_csv - parses the result into a list of tensors
    columns = tf.decode_csv(value, record_defaults=record_defaults)
    feat = tf.stack(columns[1:])
    col8 = columns[0]
    return feat, col8


def _load_set(max_num_records, f_name):
    f_queue = tf.train.string_input_producer([f_name])
    features, labels = read_file(f_queue)
    set_x = np.empty((0, 3197), float)
    set_y = np.empty((0,), int)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        # tf.train.start_queue_runners - to populate the queue before run/eval to execute read
        # otherwise read will block while it waits for file names from the queue
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(max_num_records):
            example, label = sess.run([features, labels])
            set_x = np.append(set_x, [example], axis=0)
            set_y = np.append(set_y, [label], axis=0)
        coord.request_stop()
        coord.join(threads)
    return set_x, set_y


def load_train_set(max_num_records=100):
    train_file = 'data/exoTrain.csv'
    train_x, train_y = _load_set(max_num_records, train_file)
    return train_x, train_y


def load_test_set(max_num_records=100):
    test_file = 'data/exoTest.csv'
    test_x, test_y = _load_set(max_num_records, test_file)
    return test_x, test_y


def normalize_features(X):
    num_of_examples = X.shape[0]
    num_of_features = X.shape[1]
    for i in range(num_of_examples):
        mean_of_row = np.mean(X[i])
        std_of_row = np.std(X[i])
        for j in range(num_of_features):
            X[i][j] -= mean_of_row
            X[i][j] /= std_of_row
    return X
