INPUT_MAX_LENGTH = 1000
OUTPUT_MAX_LENGTH = 15
TEST_PREDICTIONS_FREQUENCY = 10
TRAIN_BATCH_SIZE = 10
NUM_QUESTIONS = 74093
MAX_NB_WORDS = 200000
MAX_NUM_ITERS = int(float(NUM_QUESTIONS) / float(TRAIN_BATCH_SIZE)) + 1
FULL_PASS_ITERS = 10
GLOVE_DIR = './download/dwr/'
TEXT_DATA_DIR = './data/marco/vocab.dat'
INPUT_MAX_LENGTH = 1000
OUTPUT_MAX_LENGTH = 15
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
DEPTH = 2