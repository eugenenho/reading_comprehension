import time
import tensorflow as tf
import numpy as np

from progbar import Progbar

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder
from tf_lstm_attention_cell import LSTMAttnCell

from simple_configs import LOG_FILE_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

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
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, questions_batch, passages_batch, start_token_batch, answers_batch=None, dropout=0.5):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.questions_placeholder : questions_batch,
            self.passages_placeholder : passages_batch,
            self.start_token_placeholder : start_token_batch,
            self.dropout_placeholder : dropout
        }
        if answers_batch is not None: feed_dict[self.answers_placeholder] = answers_batch
        return feed_dict

    def add_embedding(self, placeholder):  
        embed_vals = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embed_vals, placeholder)
        return embeddings

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def encode_w_attn(self, inputs, sequence_length, prev_states, scope="", reuse=False):
        self.attn_cell = LSTMAttnCell(self.size, prev_states)
        with vs.variable_scope(scope, reuse):
            o, final_state = tf.nn.dynamic_rnn(self.attn_cell, inputs, dtype=tf.float32, sequence_length=self.seq_length(passages))
        return (o, final_state)

    def add_prediction_op(self): 
        questions = self.add_embedding(self.questions_placeholder)
        passages = self.add_embedding(self.passages_placeholder)

        # Question encoder
        with tf.variable_scope("question"): 
            q_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            q_outputs, _ = tf.nn.dynamic_rnn(q_cell, questions, dtype=tf.float32, sequence_length=self.seq_length(questions))

        # Passage encoder
        p_outputs, _ = encode_w_attn(passages, self.seq_length(passages), q_outputs)

        # with tf.variable_scope("passage"):
        #     p_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
        #     p_outputs, p_state_tuple = tf.nn.dynamic_rnn(p_cell, passages, initial_state=q_state_tuple, dtype=tf.float32, sequence_length=self.seq_length(passages))

        # Attention state encoder
        with tf.variable_scope("attention"): 
            a_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            a_outputs, _ = tf.nn.dynamic_rnn(a_cell, p_outputs, dtype=tf.float32)

        q_last = tf.slice(q_outputs, [0, QUESTION_MAX_LENGTH - 1, 0], [-1, 1, -1])
        p_last = tf.slice(p_outputs, [0, PASSAGE_MAX_LENGTH - 1, 0], [-1, 1, -1])
        a_last = tf.slice(a_outputs, [0, PASSAGE_MAX_LENGTH - 1, 0], [-1, 1, -1])
        q_p_a_hidden = tf.concat(2, [q_last, p_last, a_last]) # SHAPE: [BATCH, 1, 3*HIDDEN_DIM]
       
        preds = list()
        
        with tf.variable_scope("decode"):
            d_cell_dim = 3*HIDDEN_DIM
            d_cell = tf.nn.rnn_cell.LSTMCell(d_cell_dim) # Make decoder cell with hidden dim

            # Make starter token input
            inp = self.start_token_placeholder # STARTER TOKEN, SHAPE: [BATCH, MAX_NB_WORDS]
            
            # make initial state for LSTM cell
            h_0 = tf.reshape(q_p_a_hidden, [-1, d_cell_dim]) # hidden state from passage and question
            c_0 = tf.reshape(tf.zeros((d_cell_dim)), [-1, d_cell_dim]) # empty memory SHAPE [BATCH, 2*HIDDEN_DIM]
            h_t = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            
            for time_step in range(OUTPUT_MAX_LENGTH):
                o_t, h_t = d_cell(inp, h_t)

                U = tf.get_variable('U', shape=(d_cell_dim, MAX_NB_WORDS), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable('b', shape=(MAX_NB_WORDS, ), dtype=tf.float32)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b # SHAPE: [BATCH, MAX_NB_WORDS]

                inp = y_t

                preds.append(y_t)
                tf.get_variable_scope().reuse_variables()

            packed_preds = tf.pack(preds, axis=2)
            preds = tf.transpose(packed_preds, perm=[0, 2, 1])
        return preds

    def add_loss_op(self, preds):
        y = tf.one_hot(self.answers_placeholder, MAX_NB_WORDS)
        
        # CREATE MASKS HERE
        index_maxs = tf.argmax(preds, 2)
        check = tf.zeros(tf.shape(index_maxs), dtype=tf.int64) + 2
        stop_tokens = tf.to_int32( tf.equal(index_maxs, check) )
        stop_token_index = tf.to_int32( tf.argmax(stop_tokens, 1) + 1 )
        masks = tf.sequence_mask(stop_token_index, OUTPUT_MAX_LENGTH)

        loss_mat = tf.nn.softmax_cross_entropy_with_logits (preds, y)

        # apply masks
        masked_loss_mat = tf.boolean_mask(loss_mat, masks)

        loss = tf.reduce_mean(masked_loss_mat)
        return loss

    def add_training_op(self, loss):        
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        return train_op

    def train_on_batch(self, sess, questions_batch, passages_batch, start_token_batch, answers_batch):
        """Perform one step of gradient descent on the provided batch of data."""
        feed = self.create_feed_dict(questions_batch, passages_batch, start_token_batch, answers_batch=answers_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, data):
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        losses = list()
        i = 0
        batch = data.get_batch()
        while batch is not None:
            q_batch = batch[0]
            p_batch = batch[1]
            a_batch = batch[2]
            s_t_batch = batch[3]

            loss = self.train_on_batch(sess, q_batch, p_batch, s_t_batch, a_batch)
            losses.append(loss)

            prog.update(i + 1, [("train loss", loss)])

            batch = data.get_batch()
            if i % 1200 == 0 and i > 0:
                self.log.write('\nNow saving file...')
                saver.save(sess, './data/model.weights')
                self.log.write('\nSaved...')
            i += 1
        return losses

    def fit(self, sess, saver, data):
        losses = []
        for epoch in range(NUM_EPOCS):
            self.log.write("\nEpoch: " + str(epoch + 1) + " out of " + str(NUM_EPOCS))
            loss = self.run_epoch(sess, data)
            losses.append(loss)
            saver.save(sess, './data/model.weights')
        return losses

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, embeddings):
        self.pretrained_embeddings = embeddings
        self.log = open(LOG_FILE_DIR, "a")
        self.build()

if __name__ == "__main__":
    print 'Starting, and now printing to log.txt'
    data = TFDataHolder('train')
    embeddings = EmbeddingHolder().get_embeddings_mat()
    with tf.Graph().as_default():
        start = time.time()
        model = TFModel(embeddings)
        model.log.write("\nBuild graph took " + str(time.time() - start) + " seconds")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        model.log.write('\ninitialzed variables')
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=config) as session:
            session.run(init)
            model.log.write('\nran init, fitting.....')
            losses = model.fit(session, saver, data)

    model.log.write('\nlosses list: ' + losses)
    model.log.close()
















