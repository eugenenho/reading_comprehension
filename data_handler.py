import numpy as np
import cPickle

from sklearn.preprocessing import OneHotEncoder

from simple_configs import TRAIN_BATCH_SIZE, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS

class DataHolder:

    # pass in data_set ('train', 'val', 'dev')
    def __init__(self, DATA_SET):
        print 'initializing Data Holder'
        self.data_set = str(DATA_SET).lower()
        self.data_size = -1 # this is set in build_X_data

        found = False
        try:
            self.X_data = np.load("./data/marco/" + self.data_set + ".xdata.npy")
            self.Y_data = np.load("./data/marco/" + self.data_set + ".ydata.npy")
            found = True
        except Exception as e:
            print 'could not find premade matrix. creating it...'

        if not found:
            self.X_data = self.build_X_data()
            self.Y_data = self.build_Y_data() 

        self.data_size = self.X_data.shape[0]
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
        X_data = np.zeros((self.data_size, INPUT_MAX_LENGTH))

        # FOR EACH PASSAGE, CONCATENATE [QUESTION, P1, P2, ..., PN]
        # pad to len 1000 with zeros (still need to figure out masking in the model)
        for i, tp in enumerate(train_passages):
            curr_input_list = train_questions[i]
            for p in tp:
                curr_input_list.extend(p)

            # padding
            if len(curr_input_list) < INPUT_MAX_LENGTH:
                curr_input_list.extend([0] * (INPUT_MAX_LENGTH - len(curr_input_list)) )

            X_data[i] = np.array(curr_input_list[:INPUT_MAX_LENGTH])
        np.save("./data/marco/" + self.data_set + ".xdata", X_data)
        print 'built x data'
        return X_data

    def build_Y_data(self):
        # Y DATA....
        answers_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.answer.pkl","rb" ))
        Y_data_indexes = np.zeros((self.data_size, OUTPUT_MAX_LENGTH))
        for i, ans in enumerate(answers_list):
            # weird thing here, the answer is stores as a list of lists
            ans = ans[0] if len(ans) == 1 else []
 
            # pad / truncate values
            if len(ans) < OUTPUT_MAX_LENGTH: ans.extend( [0] * (OUTPUT_MAX_LENGTH - len(ans)) )
            # add to matrix
            Y_data_indexes[i] = np.array(ans[:OUTPUT_MAX_LENGTH])

        Y_data_indexes = Y_data_indexes.astype(int)
        np.save("./data/marco/" + self.data_set + ".ydata", Y_data_indexes)

        print 'built y data'
        return Y_data_indexes

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
        return self.build_Y_batch(self.Y_data[self.start_index:self.end_index])

    def build_Y_batch(self, Y_batch_full):
        # enc = OneHotEncoder()
        # return enc.fit(Y_batch_full)
        batch = self.end_index - self.start_index
        Y_data = np.zeros((batch, OUTPUT_MAX_LENGTH, 228999))
        for r, row in enumerate(self.Y_data[self.start_index:self.end_index]):
            for c, i in enumerate(row):
                Y_data[r][c][i] = 1
        return Y_data

    def get_full_data(self):
        print 'building full Y data'
        Y_data = np.zeros((self.Y_data.shape[0], OUTPUT_MAX_LENGTH, MAX_NB_WORDS))
        for r, row in enumerate(self.Y_data[self.start_index:self.end_index]):
            for c, i in enumerate(row):
                if i > MAX_NB_WORDS: continue
                Y_data[r][c][i] = 1
        return self.X_data, Y_data














