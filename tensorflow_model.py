import time
import tensorflow as tf
import numpy as np

from progbar import Progbar

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder

from simple_configs import NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR


class TFModel():
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, QUESTION_MAX_LENGTH), name="questions")
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, PASSAGE_MAX_LENGTH), name="passages")
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, OUTPUT_MAX_LENGTH, MAX_NB_WORDS), name="answers")

    def create_feed_dict(self, questions_batch, passages_batch, answers_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.questions_placeholder : questions_batch,
            self.passages_placeholder : passages_batch
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
        q_p_hidden = tf.concat(2, [q_last, p_last])

        with tf.variable_scope("decode"):
            U = tf.get_variable('U', shape=(2 * HIDDEN_DIM, MAX_NB_WORDS * OUTPUT_MAX_LENGTH), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable('b', shape=(MAX_NB_WORDS*OUTPUT_MAX_LENGTH, ), dtype=tf.float32)
            
            q_p_hidden = tf.reshape(q_p_hidden, [-1, 2 * HIDDEN_DIM])
            
            output = tf.reshape(tf.matmul(q_p_hidden, U) + b, [OUTPUT_MAX_LENGTH, -1, MAX_NB_WORDS])
            print output

            preds = tf.nn.ctc_beam_search_decoder(output, OUTPUT_MAX_LENGTH, beam_width=100, top_paths=1, merge_repeated=True)
        return 'preds'

    def add_loss_op(self, preds):
        y = self.answers_placeholder   
        loss_mat = tf.nn.softmax_cross_entropy_with_logits(preds, self.answers_placeholder)
        loss = tf.reduce_mean(loss_mat)

        return loss

    def add_training_op(self, loss):        
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        return train_op

    def train_on_batch(self, sess, questions_batch, passages_batch, answers_batch):
        """Perform one step of gradient descent on the provided batch of data."""
        feed = self.create_feed_dict(questions_batch, passages_batch, answers_batch=answers_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        print 'loss:', loss
        return loss

    def run_epoch(self, sess, q_data, p_data, a_data):
        prog = Progbar(target=1 + int(len(q_data) / TRAIN_BATCH_SIZE))
        
        losses = list()
        batches_list = self.minibatches(q_data, p_data, a_data, TRAIN_BATCH_SIZE)
        for i, batch in enumerate(batches_list):
            q_batch = batch[0]
            p_batch = batch[1]
            a_batch = batch[2]

            loss = self.train_on_batch(sess, q_batch, p_batch, a_batch)
            print 'here'
            losses.append(loss)

            prog.update(i + 1, [("train loss", loss)])

        return losses

    def minibatches(self, q_data, p_data, a_data, batch_size):
        time_start = time.time()
        start = 0
        end = batch_size
        batches = list()
        while True:
            batches.append( (q_data[start:end], p_data[start:end], a_data[start:end]) )

            end = min( start + batch_size, len(q_data) )
            start += batch_size

            if start >= len(q_data): break

        print 'split batches done, and took', time.time() - time_start, 'seconds'
        return batches


    def fit(self, sess, q_data, p_data, a_data):
        losses = []
        for epoch in range(NUM_EPOCS):
            print "Epoch:", epoch + 1, "out of", NUM_EPOCS
            loss = self.run_epoch(sess, q_data, p_data, a_data)
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
    q_data, p_data, a_data = data.get_full_data()
    with tf.Graph().as_default():
        print "Building model..."
        start = time.time()
        model = TFModel(embeddings)
        print "took", time.time() - start, "seconds"

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            losses = model.fit(session, q_data, p_data, a_data)

    print 'losses list:', losses

















