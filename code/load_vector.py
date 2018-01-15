# -*- coding: UTF-8 -*-

import os
import numpy as np
import pickle
from keras.preprocessing import text
from gensim.models import KeyedVectors

# first, build index mapping words in the embeddings set
# to their embedding vector
# w2v or glove
def build_words_mapping(filename, type='w2v'):
    print('Indexing word vectors.')
    embeddings_index = {}
    if type == 'glove':
        f = open(os.path.join(filename))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    elif type == 'w2v':
        embeddings_index = KeyedVectors.load_word2vec_format(filename, binary=True)
        # print(model['model'])
    else:
        print('No find type:{}'.format(type))
    return embeddings_index


# second, prepare text samples and their labels
def build_txt_dict(filename):
    print('Processing text dataset')
    texts = []  # list of text samples
    dataset = open(filename).readlines()
    for i in range(len(dataset)//4):
        texts.append(dataset[4*i+2])
    words = []
    for line in texts:
        line_temp = text.text_to_word_sequence(line,
                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                               lower=False,
                                               split=" ")
        # print(line_temp)
        # input()
        words.extend(line_temp)
    # print(len(words))
    words = set(words)
    # for word in words:
    #     print(word)
    print('Found %s texts.' % len(words))
    return words


def generate_mini_model(embeddings_index, words, filename):
    embeddings_map = {}
    for word in words:
        try:
            vec = np.asarray(embeddings_index[word], dtype='float32')
        except:
            vec = np.asarray([0.0]*300, dtype='float32')
        embeddings_map[word] = vec
    f = open(filename, 'wb')
    pickle.dump(embeddings_map, f)
    f.close()
    print('generate finish')


if __name__ == '__main__':
    BASE_DIR = 'D:/file/intent_model/'
    TEXT_DATA_DIR = BASE_DIR + 'SLU/data/temp/'
    words1 = build_txt_dict(filename=TEXT_DATA_DIR+'finish.txt')
    filename = BASE_DIR + 'word_embedding/GoogleNews-vectors-negative300.bin'
    embeddings_map1 = build_words_mapping(filename=filename, type='w2v')
    filename = TEXT_DATA_DIR+'w2v_min_model.plk'
    generate_mini_model(embeddings_map1, words1, filename)

    # 测试
    # f = open(filename, 'rb')
    # data = pickle.load(f)
    # print(len(data.keys()))
    # while True:
    #     word = input()
    #     print(data[word])
