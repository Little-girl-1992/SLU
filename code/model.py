# -*- coding: UTF-8 -*-
import os
import pickle
import numpy as np
from copy import deepcopy
from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, SimpleRNN, Dense
from keras.layers import Add, Dot, Concatenate, Multiply
from code.show_loss import LossHistory

np.random.seed(1024)  # for reproducibility

BASE_DIR = 'D:/file/intent_model/'
VECTOR_DIR = BASE_DIR + 'word_embedding/'
TEXT_DATA_DIR = BASE_DIR + 'SLU/data/temp/'
MAX_SEQUENCE_LENGTH = 100  # 最长的句子长度
MAX_NB_WORDS = 1100  # 最大的词汇量
MAX_NB_LABELS = 51  # 标签的数量 需要+1 因为没有0标签
EMBEDDING_DIM = 300
MAX_MEMORY = 5
VALIDATION_SPLIT = 0.2
batch_size = 32
nb_epoch = 50


# 获取文本
def processing_text(text_file):
    print('Processing text dataset')
    texts = []  # list of text samples
    dataset = open(text_file).readlines()
    for i in range(len(dataset) // 4):
        texts.append(dataset[4 * i + 2])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n',
                          lower=False,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return word_index, sequences


# 获取类别
def processing_label(text_file):
    print('Processing text dataset')
    labels = []  # list of text samples
    dataset = open(text_file).readlines()
    for i in range(len(dataset) // 4):
        labels.append(dataset[4 * i + 3].strip('>>>> '))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n',
                          lower=False,
                          split=" ",
                          char_level=False)  # 将filters中的_去掉
    tokenizer.fit_on_texts(labels)
    sequences = tokenizer.texts_to_sequences(labels)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return word_index, sequences


# 按id查找text和label
def processing_id(text_file):
    dataset = open(text_file).readlines()
    samples = 50
    id_text_list = [list() for _ in range(samples)]
    id_label_list = [list() for _ in range(samples)]
    for index in range(0, samples):
        id_text_list[int(dataset[index * 4])].append(dataset[index * 4 + 2])
        id_label_list[int(dataset[index * 4])].append(dataset[index * 4 + 3])
    return id_text_list, id_label_list


# maxs, maxw, maxl
def get_max_value(text_file):
    word_index, sequences = processing_text(text_file)
    label_index, _ = processing_label(text_file)
    maxs = 0
    for line in sequences:
        maxs = max(maxs, len(line))
    maxw = len(word_index.keys())
    maxl = len(label_index.keys())
    return maxs, maxw, maxl


# 将类别映射到二值列表上
def to_categorical(y, num_classes=None):
    temp = [0 for _ in range(num_classes)]
    for i in y:
        temp[i] = 1
    return temp


def tran_m(memory_list):
    memory = [list() for _ in range(MAX_MEMORY)]
    for i in range(len(memory_list)):
        for j in range(MAX_MEMORY):
            memory[j].append(list(memory_list[i][j]))
    return np.array(memory)


# 从finish.txt加载数据，建立文本和标签的索引
def load_data(text_file):
    _, sequences = processing_text(text_file=text_file)
    _, labels    = processing_label(text_file=text_file)
    dataset = open(text_file).readlines()
    samples = len(dataset)//4
    temp = [list() for _ in range(samples)]
    for i in range(samples):
        for j in range(1, MAX_MEMORY + 1):
            if i-j >= 0 and int(dataset[i*4]) == int(dataset[(i-j)*4]):
                temp[i].append(sequences[i-j])
            else:
                temp[i].append([0])

    x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    t = [pad_sequences(temp[k], maxlen=MAX_SEQUENCE_LENGTH) for k in range(samples)]
    y = [to_categorical(label, num_classes=MAX_NB_LABELS) for label in labels]
    m = tran_m(t)
    data = [x, np.array(m), np.array(y)]
    return data


# 生成embedding的索引embedding_matrix
def embedding(text_file, w2v_file):
    # 加载词向量模型
    f = open(w2v_file, 'rb')
    w2v_mini = pickle.load(f)
    f.close()
    word_index, _ = processing_text(text_file=text_file)
    embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = w2v_mini.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def train_rnn(train_data, valid_data, embedding_matrix, save_path=''):
    print('Build model...')

    embedding_layer = Embedding(output_dim=EMBEDDING_DIM,
                                input_dim=MAX_NB_WORDS + 1,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=False,
                                name='embedding')
    rnn_layer = SimpleRNN(EMBEDDING_DIM,
                          activation='relu',
                          dropout=0.2,
                          recurrent_dropout=0.2,
                          return_sequences=False,
                          name='RNN')
    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input')
    embedding_l = embedding_layer(main_input)
    rnn_l = rnn_layer(embedding_l)

    aux_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='aux_input1')
    embedding_r1 = embedding_layer(aux_input1)
    rnn_r1 = rnn_layer(embedding_r1)

    aux_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='aux_input2')
    embedding_r2 = embedding_layer(aux_input2)
    rnn_r2 = rnn_layer(embedding_r2)

    p_layer = Dot(axes=1, name='att')
    p1 = p_layer([rnn_l, rnn_r1])
    p2 = p_layer([rnn_l, rnn_r2])

    assert p_layer.get_output_at(0) == p1
    assert p_layer.get_output_at(1) == p2

    h1 = Multiply()([p1, rnn_r1])
    h2 = Multiply()([p2, rnn_r2])

    h = Add()([h1, h2])

    c = Concatenate(axis=1)([h, rnn_l])

    a = Dense(1, activation='sigmoid', name='ctrl')(c)

    hh = Multiply()([a, h])

    added = Add()([hh, rnn_l])

    output1 = Dense(EMBEDDING_DIM, activation='relu', name='dense1')(added)
    output2 = Dense(MAX_NB_LABELS, activation='softmax', name='dense2')(output1)
    model = Model(inputs=[main_input, aux_input1, aux_input2], outputs=[output2])

    # 打印模型
    model.summary()
    input()
    # 编译模型
    print('compile...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # 创建一个实例history
    history = LossHistory()

    train_x, train_m, train_y = train_data
    index = np.arange(len(train_x))
    np.random.shuffle(index)

    print('fit...')
    model.fit([train_x[index], train_m[0][index], train_m[1][index]], train_y[index],
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=VALIDATION_SPLIT,
              callbacks=[history]
              )
    # 绘制acc-loss曲线
    history.loss_plot('epoch', pic_path=save_path+'loss.png')
    # 保存模型
    model.save(save_path+'model_01.h5')
    del model


def valid_rnn(valid_data, save_path=''):
    model = load_model(save_path+'model_01.h5')
    valid_x, valid_m, valid_y = valid_data
    score, acc = model.evaluate([valid_x, valid_m[0], valid_m[1]], valid_y,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def test_rnn(valid_data, save_path=''):
    model = load_model(save_path + 'model_01.h5')
    valid_x, valid_m, valid_y = valid_data
    dict_label, _ = processing_label(text_file=filename1)
    print(dict_label)
    while True:
        i = int(input('i>>>>>'))
        j = int(input('j>>>>>'))
        index = np.arange(i, j)
        p = model.predict([valid_x[index], valid_m[0][index], valid_m[1][index]],
                          batch_size=2)
        print(p)
        print(valid_y[index])


if __name__ == '__main__':
    filename1 = TEXT_DATA_DIR + 'train.txt'
    filename2 = TEXT_DATA_DIR + 'valid.txt'
    filename3 = TEXT_DATA_DIR + 'w2v_min_model.plk'
    embedding_matrix_global = embedding(text_file=filename1, w2v_file=filename3)

    # 加载训练数据和测试数据
    train_data_g = load_data(text_file=filename1)
    valid_data_g = load_data(text_file=filename2)
    # print(train_data_g[0][151])
    # print(valid_data_g[0][151])
    # 训练模型
    train_rnn(train_data_g, valid_data_g, embedding_matrix_global, save_path=TEXT_DATA_DIR)

    # valid_rnn(valid_data_g, save_path=TEXT_DATA_DIR)

    # test_rnn(train_data_g, save_path=TEXT_DATA_DIR)

    # print(get_max_value(filename1))

    # yg = [1,2,3,6]
    # nb = 6
    # print(to_categorical(yg, num_classes=nb))
