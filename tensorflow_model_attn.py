import time
import tensorflow as tf
import numpy as np

from model import Model

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder
from tf_lstm_attention_cell import LSTMAttnCell
import get_predictions

from simple_configs import LOG_FILE_DIR, SAVE_MODEL_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, VOCAB_SIZE, LEARNING_RATE, HIDDEN_DIM

class TFModel(Model):

    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, QUESTION_MAX_LENGTH), name="questions")
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, PASSAGE_MAX_LENGTH), name="passages")
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, OUTPUT_MAX_LENGTH), name="answers")
        self.start_token_placeholder = tf.placeholder(tf.float32, shape=(None, EMBEDDING_DIM), name="starter_token")
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
        embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, placeholder)
        return embeddings

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def encode_w_attn(self, inputs, mask, prev_states, scope="", reuse=False):
        self.attn_cell = LSTMAttnCell(HIDDEN_DIM, prev_states)
        with tf.variable_scope(scope, reuse):
            o, final_state = tf.nn.dynamic_rnn(self.attn_cell, inputs, dtype=tf.float32, sequence_length=mask)
        return (o, final_state)

    def add_prediction_op(self): 
        questions = self.add_embedding(self.questions_placeholder)
        passages = self.add_embedding(self.passages_placeholder)

        # Question encoder
        with tf.variable_scope("question"): 
            q_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            q_outputs, _ = tf.nn.dynamic_rnn(q_cell, questions, dtype=tf.float32, sequence_length=self.seq_length(questions))

        # Passage encoder
        p_outputs, _ = self.encode_w_attn(passages, self.seq_length(passages), q_outputs)

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
            inp = self.start_token_placeholder # STARTER TOKEN, SHAPE: [BATCH, VOCAB_SIZE]

            # make initial state for LSTM cell
            h_0 = tf.reshape(q_p_a_hidden, [-1, d_cell_dim]) # hidden state from passage and question
            c_0 = tf.reshape(tf.zeros((d_cell_dim)), [-1, d_cell_dim]) # empty memory SHAPE [BATCH, 2*HIDDEN_DIM]
            h_t = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            
            for time_step in range(OUTPUT_MAX_LENGTH):
                o_t, h_t = d_cell(inp, h_t)

                U = tf.get_variable('U', shape=(d_cell_dim, VOCAB_SIZE), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable('b', shape=(VOCAB_SIZE, ), dtype=tf.float32)
                tf.summary.scalar('dropout_keep_probability', self.dropout_placeholder)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b # SHAPE: [BATCH, VOCAB_SIZE]

                # if self.testing:
                inp = tf.argmax(tf.nn.softmax(y_t), 1)
                inp = tf.nn.embedding_lookup(self.pretrained_embeddings, inp)
                # else: 
                #     inp = tf.slice(self.answers_placeholder, [0, time_step], [-1, 1]) 
                #     inp = tf.nn.embedding_lookup(self.pretrained_embeddings, inp)
                #     inp = tf.reshape(inp, [-1, EMBEDDING_DIM])

                preds.append(y_t)

                tf.get_variable_scope().reuse_variables()

            packed_preds = tf.pack(preds, axis=2)
            preds = tf.transpose(packed_preds, perm=[0, 2, 1])
        return preds

    def add_loss_op(self, preds):
        y = tf.one_hot(self.answers_placeholder, VOCAB_SIZE)
        
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
        tf.summary.scalar('cross_entropy_loss', cross_entropy)
        return loss

    def add_training_op(self, loss):        
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        return train_op

    def __init__(self, embeddings, testing=False):
        self.testing = testing
        self.pretrained_embeddings = tf.Variable(embeddings)
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
        # tensorboard code
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        tf.global_variables_initializer().run()
        # end of tensorboard code
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

            model.log.write("starting predictions now.....")
            preds = model.predict(session, saver, data)
            index_word = get_predictions.get_index_word_dict()
            preds = get_predictions.sub_in_word(preds, index_word)
            get_predictions.build_json_file(preds, './data/train_preds.json')


    model.log.close()
















