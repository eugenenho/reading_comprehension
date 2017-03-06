import copy
from tqdm import tqdm 
import numpy as np
import cPickle
import time
from scipy.sparse import lil_matrix

import simple_model
import simple_configs

def save_model(nn_model):
    nn_model.save_weights('./model_save', overwrite=True)

# This is the main function for the model right now.
# Builds model from the other file, gets data, and runs a fit
# Then saves model weights
def train_model():
    nn_model = simple_model.build_model()

    # get data
    X_train = build_X_data()
    Y_train_indexes = build_Y_indexes()

    start_time = time.time()
    sents_batch_iteration = 1
    # for iter_num in range(1, simple_configs.FULL_PASS_ITERS + 1):
    # print 'Full-data-pass iteration num: ' + str(iter_num)
    # FIT
    for i in tqdm(range(1, simple_configs.MAX_NUM_ITERS + 1)):

        start_index = i * simple_configs.TRAIN_BATCH_SIZE
        end_index = start_index + simple_configs.TRAIN_BATCH_SIZE

        X_batch = X_train[start_index:end_index]
        Y_batch = get_Y_batch(i, Y_train_indexes, start_index, end_index)

        nn_model.fit(X_batch, Y_batch, batch_size=simple_configs.TRAIN_BATCH_SIZE, nb_epoch=1, verbose=1)

        if sents_batch_iteration % simple_configs.TEST_PREDICTIONS_FREQUENCY == 0:
            save_model(nn_model)

        sents_batch_iteration += 1

    print 'Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / iter_num)

    save_model(nn_model)


# This constructs the data from the pickled objects
def build_X_data():
    # X DATA...
    X_train = np.zeros((simple_configs.NUM_QUESTIONS, simple_configs.INPUT_MAX_LENGTH))
    train_passages = cPickle.load(open("./data/marco/train.ids.passage.pkl","rb"))
    train_questions = cPickle.load(open("./data/marco/train.ids.question.pkl", "rb" ))

    # FOR EACH PASSAGE, CONCATENATE [QUESTION, P1, P2, ..., PN]
    # pad to len 1000 with zeros (still need to figure out masking in the model)
    for i, tp in enumerate(train_passages):
        curr_input_list = train_questions[i]
        for p in tp:
            curr_input_list.extend(p)

        # remove words witout embeddings (really uncommon words)
        for w in reversed(curr_input_list):
            if int(w) > simple_configs.MAX_NB_WORDS: curr_input_list.remove(w)

        # padding
        if len(curr_input_list) < simple_configs.INPUT_MAX_LENGTH:
            curr_input_list.extend( [0] * (simple_configs.INPUT_MAX_LENGTH - len(curr_input_list)) )

        X_train[i] = np.array(curr_input_list[:simple_configs.INPUT_MAX_LENGTH])

    print 'Made X data'
    return X_train

def build_Y_indexes():
    # Y DATA....
    answers_list = cPickle.load(open("./data/marco/train.ids.answer.pkl","rb" ))
    Y_train_indexes = np.zeros((simple_configs.NUM_QUESTIONS, simple_configs.OUTPUT_MAX_LENGTH))
    for i, ans in enumerate(answers_list):
        # weird thing here, the answer is stores as a list of lists
        ans = ans[0] if len(ans) == 1 else []
        # remove words witout embeddings (really uncommon words)
        for w in reversed(ans): 
            if int(w) > simple_configs.MAX_NB_WORDS: ans.remove(w)
        # pad / truncate values
        if len(ans) < simple_configs.OUTPUT_MAX_LENGTH: ans.extend( [0] * (simple_configs.OUTPUT_MAX_LENGTH - len(ans)) )
        # add to matrix
        Y_train_indexes[i] = np.array(ans[:simple_configs.OUTPUT_MAX_LENGTH])

    Y_train_indexes = Y_train_indexes.astype(int)
    return Y_train_indexes

def get_Y_batch(batch_num, Y_train_indexes, start_index, end_index):
    # Now we need to convert this into a matrix of 1-hot vectors.  So it will be 3D with dimentions below
    Y_train = np.zeros((simple_configs.TRAIN_BATCH_SIZE, simple_configs.OUTPUT_MAX_LENGTH, simple_configs.MAX_NB_WORDS))
    for r, row in enumerate(Y_train_indexes[start_index:end_index]):
        for c, i in enumerate(row):
            Y_train[r][c][i] = 1

    print 'Made Y batch', Y_train.shape
    return Y_train

train_model()























