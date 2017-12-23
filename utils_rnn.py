import string
import re
import numpy as np


def load_data(fname, max_sentences_count=1000):
    sentence_count = 0
    data = []
    with open(fname, 'r') as f:
        text = f.read()
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    for sentence in sentences:
        data.append(sentence)
        sentence_count += 1
        if sentence_count > max_sentences_count:
            break
    return data


def word2idx(data):
    word_dict = {}
    numeric_sentences = []
    table = {ord(char): None for char in string.punctuation}
    for line in data:
        line = line.strip().replace('\n', '')
        if line:
            numeric_sentence = []
            words = line.lower().translate(table).split()
            for word in words:
                if word not in word_dict:
                    word_dict[word] = len(word_dict)
                numeric_sentence.append(word_dict[word])
            numeric_sentences.append(numeric_sentence)
    reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return numeric_sentences, word_dict, reverse_dict


def sin(x, T):
    return np.sin(12.0 * np.pi * x / T)


def cos(x, T):
    return np.cos(17.0 * np.pi * x / T) / 3


def toy_problem(T, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x, T) # + cos(x, T) + noise
