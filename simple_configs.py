# DATA PARAMETERS
TRAIN_BATCH_SIZE = 32
SAVE_PREDICTIONS_FREQUENCY = 10
NUM_EPOCS = 10
LEARNING_RATE = 0.001

# Text params
QUESTION_MAX_LENGTH = 50
PASSAGE_MAX_LENGTH = 150
INPUT_MAX_LENGTH = 200
OUTPUT_MAX_LENGTH = 25

# Embedding params
MAX_NB_WORDS = 228999
EMBEDDING_DIM = 50

# model params
HIDDEN_DIM = 10
DEPTH = 1

# directories
GLOVE_DIR = './download/dwr/'
TEXT_DATA_DIR = './data/marco/vocab.dat'
EMBEDDING_MAT_DIR = './data/marco/embeddings' + str(EMBEDDING_DIM) + '.npy'
SAVE_MODEL_DIR = './baseline_model'
LOG_FILE_DIR = './log.txt'