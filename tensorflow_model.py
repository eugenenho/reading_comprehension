import time
import tensorflow as tf
import numpy as np

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder

from simple_configs import NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR


class TFModel():
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.questions_placeholder = tf.placeholder(tf.int64, shape=(None, QUESTION_MAX_LENGTH), name="questions")
        self.passages_placeholder = tf.placeholder(tf.int64, shape=(None, PASSAGE_MAX_LENGTH), name="passages")
        self.answers_placeholder = tf.placeholder(tf.int64, shape=(None, OUTPUT_MAX_LENGTH, MAX_NB_WORDS), name="answers")

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

        q_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
        encoded_questions = tf.nn.dynamic_rnn(q_cell, questions, dtype=tf.float64)
        print 'questions', encoded_questions
        
        p_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
        encoded_passages = tf.nn.dynamic_rnn(p_cell, passages, dtype=tf.float64)
        print 'passages', encoded_passages
        
        encoded_info = tf.concat(encoded_questions[0], encoded_passages[0])
        print 'info', encoded_info

        # preds = tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True)
        
        preds = encoded_questions
        return preds

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
        _, loss, _ = sess.run([self.train_op, self.loss, _], feed_dict=feed)
        return loss

    def run_epoch(self, sess, q_data, p_data, a_data):
        prog = Progbar(target=1 + int(len(train) / TRAIN_BATCH_SIZE))
        losses = []
        for i, batch in enumerate(minibatches(train, TRAIN_BATCH_SIZE)):
            loss, _ = self.train_on_batch(sess, *batch)
            losses.append(loss)
            prog.update(i + 1, [("train loss", loss)])

        return losses

    def fit(self, sess, q_data, p_data, a_data):
        losses = []
        for epoch in range(NUM_EPOCS):
            print "Epoch %d out of %d", epoch + 1, NUM_EPOCS
            loss, _ = self.run_epoch(sess, train)
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
        print "took %.2f seconds", time.time() - start

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            losses = model.fit(session, q_data, p_data, a_data)

    print 'losses list:', losses

















