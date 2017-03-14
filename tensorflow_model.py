import time
import tensorflow as tf
import numpy as np

from progbar import Progbar

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder

from simple_configs import NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

# MASKING AND DROPOUT!!!, and save as we go, and data memory handling
class TFModel():
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, QUESTION_MAX_LENGTH), name="questions")
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, PASSAGE_MAX_LENGTH), name="passages")
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, OUTPUT_MAX_LENGTH), name="answers")
        self.start_token_placeholder = tf.placeholder(tf.float32, shape=(None, MAX_NB_WORDS), name="starter_token")

    def create_feed_dict(self, questions_batch, passages_batch, start_token_batch, answers_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.questions_placeholder : questions_batch,
            self.passages_placeholder : passages_batch,
            self.start_token_placeholder : start_token_batch
        }
        if answers_batch is not None: feed_dict[self.answers_placeholder] = answers_batch
        return feed_dict

    def add_embedding(self, placeholder):  
        embed_vals = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embed_vals, placeholder)
        return embeddings
        # embeddings = tf.reshape(embeddings, [-1, self.config.n_features * self.config.embed_size])
        # return embeddings

    def add_prediction_op(self): 
        questions = self.add_embedding(self.questions_placeholder)
        passages = self.add_embedding(self.passages_placeholder)
        with tf.variable_scope("question"):
            q_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            q_outputs, q_state_tuple = tf.nn.dynamic_rnn(q_cell, questions, dtype=tf.float32)

        with tf.variable_scope("passage"):
            p_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            p_outputs, p_state_tuple = tf.nn.dynamic_rnn(p_cell, passages, initial_state=q_state_tuple, dtype=tf.float32)

        q_last = tf.slice(q_outputs, [0, QUESTION_MAX_LENGTH - 1, 0], [-1, 1, -1])
        p_last = tf.slice(p_outputs, [0, PASSAGE_MAX_LENGTH - 1, 0], [-1, 1, -1])
        q_p_hidden = tf.concat(2, [q_last, p_last]) # SHAPE: [BATCH, 1, 2*HIDDEN_DIM]

       
        preds = list()
        with tf.variable_scope("decode"):
            d_cell = tf.nn.rnn_cell.LSTMCell(2*HIDDEN_DIM) # Make decoder cell with hidden dim

            # Make starter token input
            inp = self.start_token_placeholder # STARTER TOKEN, SHAPE: [BATCH, 1, MAX_NB_WORDS]
            
            # make initial state for LSTM cell
            h_0 = tf.reshape(q_p_hidden, [-1, 2*HIDDEN_DIM]) # hidden state from passage and question
            c_0 = tf.reshape(tf.zeros((2*HIDDEN_DIM)), [-1, 2*HIDDEN_DIM]) # empty memory SHAPE [BATCH, 2*HIDDEN_DIM]
            h_t = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            
            for time_step in range(OUTPUT_MAX_LENGTH):
                o_t, h_t = d_cell(inp, h_t)

                U = tf.get_variable('U', shape=(2 * HIDDEN_DIM, MAX_NB_WORDS), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable('b', shape=(MAX_NB_WORDS, ), dtype=tf.float32)
                o_t = tf.matmul(o_t, U) + b # SHAPE: [BATCH, MAX_NB_WORDS]

                # o_drop_t = tf.nn.dropout(h_t, self.dropout_placeholder)
                # y_t = tf.matmul(o_drop_t, U) + b_2

                y_t = o_t
                inp = y_t

                preds.append(y_t)
                tf.get_variable_scope().reuse_variables()

            packed_preds = tf.pack(preds, axis=2)
            preds = tf.transpose(packed_preds, perm=[0, 2, 1])
        return preds

    def add_loss_op(self, preds):
        y = tf.one_hot(self.answers_placeholder, MAX_NB_WORDS)
        loss_mat = tf.nn.softmax_cross_entropy_with_logits(preds, y)
        loss = tf.reduce_mean(loss_mat)

        return loss

    def add_training_op(self, loss):        
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        return train_op

    def train_on_batch(self, sess, questions_batch, passages_batch, start_token_batch, answers_batch):
        """Perform one step of gradient descent on the provided batch of data."""
        print 'traiing on one batch'
        feed = self.create_feed_dict(questions_batch, passages_batch, start_token_batch, answers_batch=answers_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        print 'loss:', loss
        return loss

    def run_epoch(self, sess, q_data, p_data, a_data, s_t):
        prog = Progbar(target=1 + int(len(q_data) / TRAIN_BATCH_SIZE))
        
        losses = list()
        print 'make batches'
        batches_list = self.minibatches(q_data, p_data, a_data, s_t, TRAIN_BATCH_SIZE)
        for i, batch in enumerate(batches_list):
            q_batch = batch[0]
            p_batch = batch[1]
            a_batch = batch[2]
            s_t_batch = batch[3]

            loss = self.train_on_batch(sess, q_batch, p_batch, s_t_batch, a_batch)
            losses.append(loss)

            prog.update(i + 1, [("train loss", loss)])

        return losses

    def minibatches(self, q_data, p_data, a_data, s_t, batch_size):
        time_start = time.time()
        start = 0
        end = batch_size
        batches = list()
        while True:
            s_t_batch = np.zeros((end - start, MAX_NB_WORDS))
            batches.append( (q_data[start:end], p_data[start:end], a_data[start:end], s_t_batch) )

            start += batch_size
            end = min( start + batch_size, len(q_data) )

            if start >= 1000: break

        print 'split batches done, and took', time.time() - time_start, 'seconds'
        return batches


    def fit(self, sess, q_data, p_data, a_data, s_t):
        losses = []
        for epoch in range(NUM_EPOCS):
            print "Epoch:", epoch + 1, "out of", NUM_EPOCS
            loss = self.run_epoch(sess, q_data, p_data, a_data, s_t)
            losses.append(loss)
        return losses

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, embeddings):
        self.pretrained_embeddings = embeddings
        self.build()

if __name__ == "__main__":
    data = TFDataHolder('train')
    embeddings = EmbeddingHolder().get_embeddings_mat()
    q_data, p_data, a_data, s_t = data.get_full_data()
    with tf.Graph().as_default():
        print "Building model..."
        start = time.time()
        model = TFModel(embeddings)
        print "took", time.time() - start, "seconds"

        init = tf.global_variables_initializer()
        print 'initialzed variables'
        with tf.Session() as session:
            session.run(init)
            print 'ran init, fitting.....'
            losses = model.fit(session, q_data, p_data, a_data, s_t)

    print 'losses list:', losses

















