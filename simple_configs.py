# DATA PARAMETERS
TRAIN_BATCH_SIZE = 10
SAVE_PREDICTIONS_FREQUENCY = 10
FULL_PASS_ITERS = 2

# Text params
INPUT_MAX_LENGTH = 1000
OUTPUT_MAX_LENGTH = 25

# Embedding params
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100

# model params
HIDDEN_DIM = 50
DEPTH = 1

# directories
GLOVE_DIR = './download/dwr/'
TEXT_DATA_DIR = './data/marco/vocab.dat'
SAVE_MODEL_DIR = './baseline_model'