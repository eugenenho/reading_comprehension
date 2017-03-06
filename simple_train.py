import copy
from tqdm import tqdm 
import numpy as np
import cPickle
import simple_model
import time

INPUT_MAX_LENGTH = 1000
OUTPUT_MAX_LENGTH = 15
TEST_PREDICTIONS_FREQUENCY = 10
TRAIN_BATCH_SIZE = 50
NUM_QUESTIONS = 74093
MAX_NB_WORDS = 200000
MAX_NUM_ITERS = int(float(NUM_QUESTIONS) / float(TRAIN_BATCH_SIZE)) + 1
FULL_PASS_ITERS = 10

def save_model(nn_model):
    nn_model.save_weights('./model_save', overwrite=True)

def train_model():
    nn_model = simple_model.build_model()

    # get data
    X_train, Y_train = build_data()

    start_time = time.time()
    sents_batch_iteration = 1
    for iter_num in range(1, FULL_PASS_ITERS + 1):
        print 'Full-data-pass iteration num: ' + str(iter_num)

        for i in tqdm(range(1, MAX_NUM_ITERS + 1)):
            X_batch = X_train[i:(i + 1)*TRAIN_BATCH_SIZE]
            Y_batch = Y_train[i:(i + 1)*TRAIN_BATCH_SIZE, :]
            nn_model.fit(X_batch, Y_batch, batch_size=TRAIN_BATCH_SIZE, nb_epoch=1, verbose=1)

            if sents_batch_iteration % TEST_PREDICTIONS_FREQUENCY == 0:
                save_model(nn_model)

            sents_batch_iteration += 1

        print 'Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / iter_num)
        if iter_num > 10: break

    save_model(nn_model)


def build_data():
    X_train = np.zeros((NUM_QUESTIONS, INPUT_MAX_LENGTH))
    train_passages = cPickle.load(open("./data/marco/train.ids.passage.pkl","rb"))
    train_questions = cPickle.load(open("./data/marco/train.ids.question.pkl", "rb" ))

    for i, tp in enumerate(train_passages):
        curr_input_list = train_questions[i]
        for p in tp:
            curr_input_list.extend(p)

        # remove words witout embeddings
        for w in reversed(curr_input_list):
            if int(w) > MAX_NB_WORDS: curr_input_list.remove(w)

        if len(curr_input_list) < INPUT_MAX_LENGTH:
            curr_input_list.extend( [0] * (INPUT_MAX_LENGTH - len(curr_input_list)) )

        X_train[i] = np.array(curr_input_list[:INPUT_MAX_LENGTH])

    Y_train_indexes = np.zeros((NUM_QUESTIONS, OUTPUT_MAX_LENGTH))
    answers_list = cPickle.load(open("./data/marco/train.ids.answer.pkl","rb" ))
    for i, ans in enumerate(answers_list):
        ans = ans[0] if len(ans) == 1 else []
        for w in reversed(ans): 
            if int(w) > MAX_NB_WORDS: ans.remove(w)
        if len(ans) < OUTPUT_MAX_LENGTH: ans.extend( [0] * (OUTPUT_MAX_LENGTH - len(ans)) )
        Y_train_indexes[i] = np.array(ans[:OUTPUT_MAX_LENGTH])

    Y_train_indexes = Y_train_indexes.astype(int)
    Y_train = np.zeros((NUM_QUESTIONS, OUTPUT_MAX_LENGTH, MAX_NB_WORDS))

    for r, row in enumerate(Y_train_indexes):
        for c, i in enumerate(row):
            if i > MAX_NB_WORDS: print r, c, i
            Y_train[r][c][i] = 1

    print 'Made Data'
    return X_train, Y_train


train_model()























