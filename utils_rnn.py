import string


def load_data(max_words_count=100):
    cur_words_count = 0
    data = []
    with open('data/en_US.news.txt', 'r') as f_news:
        for line in f_news:
            cur_words_count += 1
            if cur_words_count > max_words_count:
                break
            data.append(line)
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
