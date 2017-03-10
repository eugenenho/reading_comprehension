import numpy as np
import cPickle

from simple_configs import TRAIN_BATCH_SIZE, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS

class DataHolder:

    # pass in data_set ('train', 'val', 'dev')
    def __init__(self, DATA_SET):
        print 'initializing Data Holder'
        self.data_set = str(DATA_SET).lower()
        self.data_size = -1 # this is set in build_X_data

        self.X_data = self.build_X_data()
        self.Y_data = self.build_Y_data() 

        self.start_index = 0
        self.end_index = TRAIN_BATCH_SIZE

    def get_num_iterations(self):
        num_batches = float(self.data_size) / float(TRAIN_BATCH_SIZE)
        if num_batches % 1 != 0: num_batches += 1
        return int(num_batches) 

    # This constructs the data from the pickled objects
    def build_X_data(self):
        # X DATA...
        train_passages = cPickle.load(open("./data/marco/" + self.data_set + ".ids.passage.pkl","rb"))
        train_questions = cPickle.load(open("./data/marco/" + self.data_set + ".ids.question.pkl", "rb" ))

        self.data_size = len(train_questions)
        X_train = np.zeros((self.data_size, INPUT_MAX_LENGTH))

        # FOR EACH PASSAGE, CONCATENATE [QUESTION, P1, P2, ..., PN]
        # pad to len 1000 with zeros (still need to figure out masking in the model)
        for i, tp in enumerate(train_passages):
            curr_input_list = train_questions[i]
            for p in tp:
                curr_input_list.extend(p)

            # remove words witout embeddings (really uncommon words)
            for w in reversed(curr_input_list):
                if int(w) > MAX_NB_WORDS: curr_input_list.remove(w)

            # padding
            if len(curr_input_list) < INPUT_MAX_LENGTH:
                curr_input_list.extend( [0] * (INPUT_MAX_LENGTH - len(curr_input_list)) )

            X_train[i] = np.array(curr_input_list[:INPUT_MAX_LENGTH])

        print 'built x data'
        return X_train

    def build_Y_data(self):
        # Y DATA....
        answers_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.answer.pkl","rb" ))
        Y_train_indexes = np.zeros((self.data_size, OUTPUT_MAX_LENGTH))
        for i, ans in enumerate(answers_list):
            # weird thing here, the answer is stores as a list of lists
            ans = ans[0] if len(ans) == 1 else []
            # remove words witout embeddings (really uncommon words)
            for w in reversed(ans): 
                if int(w) > MAX_NB_WORDS: ans.remove(w)
            # pad / truncate values
            if len(ans) < OUTPUT_MAX_LENGTH: ans.extend( [0] * (OUTPUT_MAX_LENGTH - len(ans)) )
            # add to matrix
            Y_train_indexes[i] = np.array(ans[:OUTPUT_MAX_LENGTH])

        Y_train_indexes = Y_train_indexes.astype(int)

        print 'built y data'
        return Y_train_indexes

    # start from any point in the data set
    def reset_batch_position(self, start_index = 0):
        self.start_index = start_index
        self.end_index = start_index + TRAIN_BATCH_SIZE

    # Get batch of data from X and Y.
    # Returns None, None after a complete iteration over the data set
    def get_batch_data(self):
        if self.start_index > self.data_size:
            print 'finished a full iteration over the data set'
            self.reset_batch_position()
            return None, None

        X_data = self.get_X_batch()
        Y_data = self.get_Y_batch()

        self.start_index += TRAIN_BATCH_SIZE
        self.end_index += TRAIN_BATCH_SIZE if self.end_index < self.data_size else self.data_size

        return X_data, Y_data

    def get_X_batch(self):
        return self.X_data[self.start_index:self.end_index]

    # Now we need to convert this into a matrix of 1-hot vectors.  So it will be 3D with dimentions below
    def get_Y_batch(self):
        batch_size = self.end_index - self.start_index
        Y_train = np.zeros((batch_size, OUTPUT_MAX_LENGTH, MAX_NB_WORDS))
        for r, row in enumerate(self.Y_data[self.start_index:self.end_index]):
            for c, i in enumerate(row):
                Y_train[r][c][i] = 1
        return Y_train