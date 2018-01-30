# -*- coding: UTF-8 -*-
import os
import pickle
import numpy as np
from copy import deepcopy
from gensim.models import KeyedVectors

from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, SimpleRNN, Dense, LSTM
from keras.layers import Add, Dot, Concatenate, Multiply
from keras.layers.core import Lambda, Activation
from keras.optimizers import RMSprop
from code.show_loss import LossHistory

np.random.seed(1024)  # for reproducibility

BASE_DIR = 'D:/file/intent_model/'
TEXT_DATA_DIR = BASE_DIR + 'SLU/data/temp/'
MINI_VECTOR = TEXT_DATA_DIR + 'glove_min_model.plk'
MAX_SEQUENCE_LENGTH = 100  # 最长的句子长度
MAX_NB_WORDS = 1200  # 最大的词汇量
MAX_NB_LABELS = 17  # 标签的数量 需要+1 因为没有0标签
MAX_MEMORY = 5

EMBEDDING_DIM = 50
TYPE = 'glove'
VECTOR_DIR = BASE_DIR + 'word_embedding/glove.6B/glove.6B.50d.txt'


# VECTOR_DIR = BASE_DIR + 'word_embedding/GoogleNews-vectors-negative300.bin'


class V2C(object):
    def __init__(self, v2c_filename):
        super(V2C, self).__init__()
        self.type = TYPE
        self.dim = EMBEDDING_DIM
        self.vector_path = v2c_filename

    # 加载词向量模型
    # w2v or glove
    def build_words_mapping(self):
        print('Indexing word vectors.')
        embeddings_index = {}
        if self.type == 'glove':
            f = open(os.path.join(self.vector_path), 'rb')
            for line in f:
                values = line.split()
                word = values[0].decode('utf-8')
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
        elif self.type == 'w2v':
            embeddings_index = KeyedVectors.load_word2vec_format(self.vector_path, binary=True)
        else:
            print('No find type:{}'.format(type))
        return embeddings_index


class Dateset(object):
    def __init__(self, text_path):
        super(Dateset, self).__init__()
        self.tokenizer_t = self.get_token_t(text_path)
        self.tokenizer_l = self.get_token_l(text_path)

    # 获取全部数据集的文本token
    @staticmethod
    def get_token_t(text_file):
        texts = []  # list of text samples
        data = open(text_file).readlines()
        for i in range(len(data) // 4):
            texts.append(data[4 * i + 2])

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                              filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)
        tokenizer.fit_on_texts(texts)
        return tokenizer

    # 获取全部数据集的文本token
    @staticmethod
    def get_token_l(text_file):
        labels = []  # list of text samples
        data = open(text_file).readlines()
        for i in range(len(data) // 4):
            labels.append(data[4 * i + 3].strip('>>>> '))

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                              filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)  # 将filters中的_去掉
        tokenizer.fit_on_texts(labels)
        return tokenizer

    # maxs, maxw, maxl
    # 确定数据集最长句子，词汇量，标签数量
    def get_max_value(self, text_file):
        print('确定数据集最长句子，词汇量，标签数量')
        data = open(text_file).readlines()
        texts = []
        labels = []
        for i in range(len(data) // 4):
            texts.append(data[4 * i + 2])
            labels.append(data[4 * i + 3].strip('>>>> '))
        word_index = self.tokenizer_t.word_index
        sequences = self.tokenizer_t.texts_to_sequences(texts)
        label_index = self.tokenizer_l.word_index
        max_s = 0
        for line in sequences:
            max_s = max(max_s, len(line))
        max_w = len(word_index.keys())
        max_l = len(label_index.keys())
        return max_s, max_w, max_l

    # 将类别映射到二值列表上
    @staticmethod
    def to_categorical(y, num_classes=None):
        temp = [0 for _ in range(num_classes)]
        for i in y:
            temp[i] = 1
        return temp

    @staticmethod
    def tran_m(memory_list):
        memory = [list() for _ in range(MAX_MEMORY)]
        for i in range(len(memory_list)):
            for j in range(MAX_MEMORY):
                memory[j].append(list(memory_list[i][j]))
        return np.array(memory)

    # 从finish.txt加载数据，建立文本和标签的索引
    def load_data(self, text_file):
        data = open(text_file).readlines()
        # 获取文本列表和对应的标签列表
        texts = []
        labels = []
        user = []
        for i in range(len(data) // 4):
            user.append(data[4 * i + 1])
            texts.append(data[4 * i + 2])
            labels.append(data[4 * i + 3].strip('>>>> '))
        sequences_t = self.tokenizer_t.texts_to_sequences(texts)
        sequences_l = self.tokenizer_l.texts_to_sequences(labels)
        samples = len(data) // 4
        temp = [list() for _ in range(samples)]
        for i in range(samples):
            for j in range(1, MAX_MEMORY + 1):
                if i - j >= 0 and int(data[i * 4]) == int(data[(i - j) * 4]):
                    temp[i].append(sequences_t[i - j])
                else:
                    temp[i].append([0])

        u = [[np.float32(1)] if t == "user" else [np.float32(0)] for t in user]
        x = pad_sequences(sequences_t, maxlen=MAX_SEQUENCE_LENGTH)
        t = [pad_sequences(temp[k], maxlen=MAX_SEQUENCE_LENGTH) for k in range(samples)]
        y = [self.to_categorical(label, num_classes=MAX_NB_LABELS) for label in sequences_l]
        m = self.tran_m(t)
        data = [x, np.array(m), np.array(y), np.array(u)]
        return data

    # 生成embedding的索引embedding_matrix
    def embedding(self, w2v_file, update=False):
        # 加载词向量模型
        word_index = self.tokenizer_t.word_index
        if update:
            print('更新词向量模型文件')
            v2c = V2C(VECTOR_DIR)
            v2c_model = v2c.build_words_mapping()
            embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))
            for word, i in word_index.items():
                if i > MAX_NB_WORDS:
                    continue
                try:
                    temp = v2c_model.get(word)
                    if temp is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[i] = np.asarray(temp, dtype='float32')
                except:
                    pass
            f = open(w2v_file, 'wb')
            pickle.dump(embedding_matrix, f)
            f.close()
        else:
            f = open(w2v_file, 'rb')
            embedding_matrix = pickle.load(f)
            f.close()
        return embedding_matrix


class Keras_MAC(object):
    def __init__(self):
        super(Keras_MAC, self).__init__()
        self.valid_split = 0.2
        self.batch_size = 64
        self.nb_epoch = 100
        self.loss_path = TEXT_DATA_DIR + 'loss_d_100.png'
        self.old_model = ''
        self.new_model = TEXT_DATA_DIR + 'model_d_100.h5'

    def build_model(self, train_data, embedding_matrix):
        print('Build model...')
        embedding_layer = Embedding(output_dim=EMBEDDING_DIM,
                                    input_dim=MAX_NB_WORDS + 1,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix],
                                    trainable=False,
                                    name='embedding')
        rnn_layer = LSTM(EMBEDDING_DIM,
                         activation='relu',
                         dropout=0.2,
                         recurrent_dropout=0.2,
                         return_sequences=False,
                         name='RNN')
        main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input')
        embedding_l = embedding_layer(main_input)
        rnn_l = rnn_layer(embedding_l)

        output1 = Dense(EMBEDDING_DIM, activation='sigmoid', name='dense')(rnn_l)
        output2 = Dense(MAX_NB_LABELS, activation='sigmoid', name='classifier')(output1)
        model = Model(inputs=[main_input
                              ], outputs=[output2])

        # 打印模型
        model.summary()
        # 编译模型
        print('compile...')

        rmsprop = RMSprop(lr=0.001)
        model.compile(loss='binary_crossentropy',
                      optimizer=rmsprop,
                      metrics=['binary_accuracy'])
        # 创建一个实例history
        history = LossHistory()

        train_x, train_m, train_y, train_u = train_data
        index = np.arange(len(train_x))
        np.random.shuffle(index)

        if self.old_model != '':
            model.load_weights(self.old_model)

        print('fit...')
        model.fit([train_x[index]
                   ],
                  train_y[index],
                  batch_size=self.batch_size,
                  epochs=self.nb_epoch,
                  validation_split=self.valid_split,
                  callbacks=[history]
                  )
        # 绘制acc-loss曲线
        history.loss_plot('epoch', pic_path=self.loss_path)
        # 保存模型
        model.save(self.new_model)
        del model

    def p_rnn(self, valid_data):
        model = load_model(self.new_model)
        valid_x, valid_m, valid_y = valid_data

        intermediate_layer_model1 = Model([model.get_layer('main_input').input,
                                           model.get_layer('aux_input1').input,
                                           model.get_layer('aux_input2').input,
                                           model.get_layer('aux_input3').input,
                                           model.get_layer('aux_input4').input,
                                           model.get_layer('aux_input5').input
                                           ],
                                          model.get_layer('attention').get_output_at(0))

        intermediate_layer_model2 = Model([model.get_layer('main_input').input,
                                           model.get_layer('aux_input1').input,
                                           model.get_layer('aux_input2').input,
                                           model.get_layer('aux_input3').input,
                                           model.get_layer('aux_input4').input,
                                           model.get_layer('aux_input5').input
                                           ],
                                          model.get_layer('ctrl').get_output_at(0))

        while True:
            i = int(input('i>>>>>'))
            # j = int(input('j>>>>>'))
            index = np.arange(i, i + 1)

            intermediate_output1 = intermediate_layer_model1.predict([valid_x[index],
                                                                      valid_m[0][index],
                                                                      valid_m[1][index],
                                                                      valid_m[2][index],
                                                                      valid_m[3][index],
                                                                      valid_m[4][index]
                                                                      ])
            intermediate_output2 = intermediate_layer_model2.predict([valid_x[index],
                                                                      valid_m[0][index],
                                                                      valid_m[1][index],
                                                                      valid_m[2][index],
                                                                      valid_m[3][index],
                                                                      valid_m[4][index]
                                                                      ])
            for i in range(len(index)):
                print('5:', intermediate_output1[i])
                print('a:', intermediate_output2[i])

    def valid_rnn(self, valid_data):
        model = load_model(self.new_model)
        valid_x, valid_m, valid_y, valid_u = valid_data
        out = model.predict([valid_x],
                            batch_size=self.batch_size)
        best_threshold = [0.5 for _ in range(MAX_NB_LABELS)]
        p = [0.0 for _ in range(MAX_NB_LABELS)]
        r = [0.0 for _ in range(MAX_NB_LABELS)]
        sum_TP, sum_p, sum_f = 0, 0, 0
        for j in range(1, valid_y.shape[1]):
            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(valid_y.shape[0]):
                if out[i, j] >= best_threshold[j]:
                    if valid_y[i, j] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if valid_y[i, j] == 1:
                        FN += 1
                    else:
                        TN += 1
            print(j, TP, FN, FP, TN)
            p[j] = TP / (TP + FP + 0.1)
            r[j] = TP / (TP + FN + 0.1)
            print(p[j])
            print(r[j])
            sum_TP += TP
            sum_p += (FP + TP)
            sum_f += (FN + TP)
        print(sum_TP / sum_p)
        print(sum_TP / sum_f)
        del model

    def test_rnn(self, valid_data):
        model = load_model(self.new_model)
        valid_x, valid_m, valid_y, valid_u = valid_data
        while True:
            i = int(input('i>>>>>'))
            # j = int(input('j>>>>>'))
            index = np.arange(i, i + 1)
            out = model.predict([valid_x[index]
                                 ],
                                batch_size=self.batch_size)
            y_test = valid_y[index]
            best_threshold = [0.5 for _ in range(MAX_NB_LABELS)]
            y_pred = np.array(
                [[1 if out[i, j] >= best_threshold[j] else 0
                  for j in range(y_test.shape[1])]
                 for i in range(len(y_test))])
            print(out)
            for i, prob in enumerate(y_test[0]):
                if prob:
                    print(i)
            print(y_test)
            print(y_pred)
