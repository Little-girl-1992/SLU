# SLU
in SLU\data\temp\train.txt:
you can see:
i+1: paragraph id
i+2: lecturer id
i+3: content of dialogue
i+4: intents(labels)

first, you need configuration SLU/code/conf.ini
MAX_MEMORY: number of memory(frozen)
MAX_NB_LABELS: number of label set in train dataset
MAX_NB_WORDS: number of word in train dataset
MAX_SEQUENCE_LENGTH: length of max sequence in train dataset

TEXT_DATA_DIR: path of data for save
TYPE: type of the word embedding
EMBEDDING_DIM: dim of word embedding
MINI_VECTOR: save path of word embedding dict for all words in train dataset
VECTOR_DIR: the model of word embedding that you can get in https://nlp.stanford.edu/projects/glove/

second, you need execute SLU/code/multi_turn.py
python multi_turn.py train.txt valid.txt [train, valid, test]
exp:
if you want train model, you can:
python multi_turn.py D:\file\intent_model\SLU\data\temp\train.txt '' train
if you want valid model, you can:
python multi_turn.py D:\file\intent_model\SLU\data\temp\train.txt D:\file\intent_model\SLU\data\temp\valid.txt valid
if you want test model, you can:
python multi_turn.py D:\file\intent_model\SLU\data\temp\train.txt D:\file\intent_model\SLU\data\temp\valid.txt test

