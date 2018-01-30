import numpy as np
from code import multi_turn as mo
# from code import test as mo
np.random.seed(1024)

BASE_DIR = 'D:/file/intent_model/'
TEXT_DATA_DIR = BASE_DIR + 'SLU/data/temp/'
MINI_VECTOR = TEXT_DATA_DIR + 'glove_min_model.plk'
VECTOR_DIR = BASE_DIR + 'word_embedding/glove.6B/glove.6B.50d.txt'


def test_v2c():
    v2c = model.V2C(VECTOR_DIR)
    v2c_model = v2c.build_words_mapping()
    while True:
        word = input('>>>')
        print(v2c_model.get(word))


def test_dateset():
    filename1 = TEXT_DATA_DIR + 'train.txt'
    dataset = model.Dateset(text_path=filename1)
    print(dataset.get_max_value(text_file=filename1))
    print(dataset.tokenizer_l.word_index)
    # print(dataset.tokenizer_t.word_index)
    train_data_g = dataset.load_data(text_file=filename1)
    embedding = dataset.embedding(w2v_file=MINI_VECTOR, update=True)
    while True:
        i = int(input('>>>:'))
        print(train_data_g[0][i])
        print(train_data_g[2][i])

        print(embedding[i])


def test_model():
    filename1 = TEXT_DATA_DIR + 'train.txt'
    filename2 = TEXT_DATA_DIR + 'valid.txt'
    dataset = mo.Dateset(text_path=filename1)
    print(dataset.tokenizer_l.word_index)
    print(dataset.get_max_value(text_file=filename1))
    train_data_g = dataset.load_data(text_file=filename1)
    valid_data_g = dataset.load_data(text_file=filename2)
    embedding_matrix = dataset.embedding(w2v_file=MINI_VECTOR, update=False)

    keras_mac = mo.Keras_MAC()
    # keras_mac.build_model(train_data_g, embedding_matrix)
    # keras_mac.p_rnn(train_data_g)
    # keras_mac.valid_rnn(train_data_g, dtype='valid')
    keras_mac.test_rnn(train_data_g)
    # keras_mac.vis_rnn()


if __name__ == '__main__':
    # test_v2c()
    test_model()

