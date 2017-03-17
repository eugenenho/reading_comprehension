# DATA PARAMETERS
TRAIN_BATCH_SIZE = 128
SAVE_PREDICTIONS_FREQUENCY = 10
NUM_EPOCS = 10
LEARNING_RATE = 0.001

PRED_BATCH_SIZE = 256

# Text params
QUESTION_MAX_LENGTH = 50
PASSAGE_MAX_LENGTH = 100
OUTPUT_MAX_LENGTH = 15

# Embedding params
MAX_NB_WORDS = 50000#228999 #MAX VALUE
EMBEDDING_DIM = 300
MAX_DATA_SIZE = -1 #-1 to set no limit on data size

# model params
HIDDEN_DIM = 100
DEPTH = 1

# directories
GLOVE_DIR = './download/dwr/'
TEXT_DATA_DIR = './data/marco/vocab.dat'
EMBEDDING_MAT_DIR = './data/marco/embeddings' + str(EMBEDDING_DIM) + '.npy'
SAVE_MODEL_DIR = './baseline_model'
LOG_FILE_DIR = './log.txt'