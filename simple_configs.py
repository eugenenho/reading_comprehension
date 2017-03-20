# DATA PARAMETERS
TRAIN_BATCH_SIZE = 12
SAVE_PREDICTIONS_FREQUENCY = 10
NUM_EPOCS = 2
LEARNING_RATE = 0.003
DROPOUT = 0.8
SMALL_DATA_SET = True

# Text params
QUESTION_MAX_LENGTH = 50
PASSAGE_MAX_LENGTH = 100
OUTPUT_MAX_LENGTH = 30
MAX_NUM_PASSAGES = 10

# Embedding params
VOCAB_SIZE = 20000#228999 #MAX VALUE
EMBEDDING_DIM = 300
MAX_DATA_SIZE = 5000 #-1 to set no limit on data size

# model params
HIDDEN_DIM = 300

# directories
GLOVE_DIR = './download/dwr/'
TEXT_DATA_DIR = './data/marco/vocab.dat'
EMBEDDING_MAT_DIR = './data/marco/embeddings' + str(EMBEDDING_DIM) + '.npy'
SAVE_MODEL_DIR = './data/model.weights'
LOG_FILE_DIR = './log.txt'
