import time
import tensorflow as tf
import numpy as np

from model import Model
from progbar import Progbar

from embeddings_handler import EmbeddingHolder
from data_handler import DataHolder
from tf_lstm_attention_cell import LSTMAttnCell


from simple_configs import LOG_FILE_DIR, SAVE_MODEL_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, VOCAB_SIZE, LEARNING_RATE, HIDDEN_DIM, MAX_GRAD_NORM, ACTIVATION_FUNC 
PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4
FILE_TBOARD_LOG = 'L2 2attn dbl '

class TFModel(Model):
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, QUESTION_MAX_LENGTH), name="questions")
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, PASSAGE_MAX_LENGTH), name="passages")
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, OUTPUT_MAX_LENGTH), name="answers")
        self.start_token_placeholder = tf.placeholder(tf.int32, shape=(None,), name="starter_token")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, questions_batch, passages_batch, start_token_batch, dropout=0.5, answers_batch=None):
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
        used = tf.sign(sequence)
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def encode_w_attn(self, inputs, mask, prev_states, scope="", reuse=False):
        
        with tf.variable_scope(scope, reuse):
            attn_cell = LSTMAttnCell(HIDDEN_DIM, prev_states, HIDDEN_DIM)
            o, final_state = tf.nn.dynamic_rnn(attn_cell, inputs, dtype=tf.float32, sequence_length=mask)
        return (o, final_state)

    def add_prediction_op(self): 
        questions = self.add_embedding(self.questions_placeholder)
        passages = self.add_embedding(self.passages_placeholder)

        # Question encoder
        with tf.variable_scope("question"): 
            q_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            q_outputs, q_final_tuple = tf.nn.dynamic_rnn(q_cell, questions, dtype=tf.float32, sequence_length=self.seq_length(self.questions_placeholder))
            q_final_c, q_final_h = q_final_tuple
            q_final_h = tf.expand_dims(q_final_h, axis=1)

        # Passage encoder with attention
        p_outputs, p_final_tuple = self.encode_w_attn(passages, self.seq_length(self.passages_placeholder), q_outputs, scope = "passage_attn")
        p_final_c, p_final_h = p_final_tuple
        p_final_h = tf.expand_dims(p_final_h, axis=1)

        # Attention state encoder (Match LSTM layer variant)
        with tf.variable_scope("attention"): 
            a_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            a_outputs, a_final_tuple = tf.nn.dynamic_rnn(a_cell, p_outputs, dtype=tf.float32, sequence_length=self.seq_length(self.passages_placeholder))
            a_final_c, a_final_h = a_final_tuple
            a_final_h = tf.expand_dims(a_final_h, axis=1)

        # Concatenation of all final hidden states
        q_p_a_hidden = tf.concat(2, [q_final_h, p_final_h, a_final_h]) # SHAPE: [BATCH, 1, 3*HIDDEN_DIM]              
        tf.Print(q_p_a_hidden, [q_p_a_hidden], message="Final Hidden State:", summarize=75)
        preds = list()
        
        with tf.variable_scope("decoder"):
            d_cell_dim = 3 * HIDDEN_DIM
            
            # Run decoder with attention between DECODER and ATTENTION/Match LSTM layer 
            d_cell = LSTMAttnCell(d_cell_dim, a_outputs, HIDDEN_DIM)
            d_cell_second = tf.nn.rnn_cell.LSTMCell(d_cell_dim) # Make decoder cell with hidden dim

            # Create first-time-step input to LSTM (starter token)
            inp = self.add_embedding(self.start_token_placeholder) # STARTER TOKEN, SHAPE: [BATCH, EMBEDDING_DIM]

            # make initial state for first-layer LSTM cell
            h_0 = tf.reshape(q_p_a_hidden, [-1, d_cell_dim]) # hidden state from passage and question
            c_0 = tf.reshape(tf.zeros((d_cell_dim)), [-1, d_cell_dim]) # empty memory SHAPE [BATCH, 2*HIDDEN_DIM]
            
            h_t1 = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            h_t2 = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            
            # U and b for manipulating the output from LSTM to logit (LSTM output -> logit)
            U = tf.get_variable('U', shape=(d_cell_dim, VOCAB_SIZE), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable('b', shape=(VOCAB_SIZE, ), dtype=tf.float32)
            
            for time_step in range(OUTPUT_MAX_LENGTH):
                
                # first layer
                o_t1, h_t1 = d_cell(inp, h_t1)

                # second layer
                o_t2, h_t2 = d_cell_second(o_t1, h_t2)

                # dropout layer
                o_drop_t = tf.nn.dropout(o_t2, self.dropout_placeholder)

                # logit / softmax manipulation
                y_t = tf.matmul(o_drop_t, U) + b # SHAPE: [BATCH, VOCAB_SIZE]
                
                # limit vocab size to words that we have seen in question or passage and popular words
                # mask = self.get_vocab_masks()
                # y_t = tf.multiply(y_t, mask)
                
                y_t = tf.nn.softmax(y_t)
                
                if self.predicting:
                    inp_index = tf.argmax(y_t, 1)
                    inp = tf.nn.embedding_lookup(self.pretrained_embeddings, inp_index)
                else: 
                     inp = tf.slice(self.answers_placeholder, [0, time_step], [-1, 1]) 
                     inp = tf.nn.embedding_lookup(self.pretrained_embeddings, inp)
                     inp = tf.reshape(inp, [-1, EMBEDDING_DIM])

                preds.append(y_t)
                tf.get_variable_scope().reuse_variables()

            packed_preds = tf.pack(preds, axis=2)
            preds = tf.transpose(packed_preds, perm=[0, 2, 1])

        return preds

    def add_loss_op(self, preds):
        y = tf.one_hot(self.answers_placeholder, VOCAB_SIZE)
        
        ans_lengths = self.seq_length(self.answers_placeholder)
        mask = tf.sequence_mask(ans_lengths, OUTPUT_MAX_LENGTH)
        mask = tf.reshape(mask, [tf.shape(y)[0], OUTPUT_MAX_LENGTH, 1])
        base_zeros = tf.zeros(tf.shape(y), dtype=tf.int32) + tf.cast(mask, tf.int32)
        full_masks = tf.cast(tf.reshape(base_zeros, [-1, VOCAB_SIZE * OUTPUT_MAX_LENGTH]), tf.bool)

        y = tf.reshape(y, [-1, VOCAB_SIZE * OUTPUT_MAX_LENGTH])
        preds = tf.reshape(preds, [-1, VOCAB_SIZE * OUTPUT_MAX_LENGTH])

        # or bool mask
        masked_preds = tf.boolean_mask(preds, full_masks)
        masked_y = tf.boolean_mask(y, full_masks)

        loss_mat = tf.nn.l2_loss(masked_y - masked_preds)
        loss = tf.reduce_mean(loss_mat)
        tf.summary.scalar(FILE_TBOARD_LOG + 'Loss per Batch', loss)
        return loss

    def add_training_op(self, loss):        
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        tf.summary.scalar(FILE_TBOARD_LOG + 'LEARNING_RATE', loss)

        grad_var_pairs = optimizer.compute_gradients(loss)
        grads = [g[0] for g in grad_var_pairs]

        clipped_grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        grad_var_pairs = [(g, grad_var_pairs[i][1]) for i, g in enumerate(clipped_grads)]
        
        grad_norm = tf.global_norm(clipped_grads)
        tf.summary.scalar(FILE_TBOARD_LOG + 'Global Gradient Norm', grad_norm)

        return optimizer.apply_gradients(grad_var_pairs)

    def run_epoch(self, sess, merged, data):
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        losses = list()
        i = 0
        batch = data.get_selected_passage_batch()
        while batch is not None:
            q_batch = batch['question']
            p_batch = batch['passage']
            a_batch = batch['answer']
            s_t_batch = batch['start_token']
            dropout = batch['dropout']

            loss = self.train_on_batch(sess, merged, q_batch, p_batch, s_t_batch, dropout, a_batch)
            losses.append(loss)

            prog.update(i + 1, [("train loss", loss)])

            batch = data.get_selected_passage_batch()
            if i % 1200 == 0 and i > 0:
                self.log.write('\nNow saving file...')
                saver.save(sess, SAVE_MODEL_DIR)
                self.log.write('\nSaved...')
            i += 1
        return losses

    def predict(self, sess, data):
        self.predicting = True
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        preds = list()
        i = 0
        
        data.reset_iter()
        batch = data.get_selected_passage_batch(predicting=True)
        while batch is not None:
            q_batch = batch['question']
            p_batch = batch['passage']
            s_t_batch = batch['start_token']
            a_batch = np.zeros((q_batch.shape[0], OUTPUT_MAX_LENGTH), dtype=np.int32)
            dropout = batch['dropout']

            prediction = self.predict_on_batch(sess, q_batch, p_batch, s_t_batch, dropout, a_batch)
            preds.append(prediction)

            prog.update(i + 1, [("Predictions going...", 1)])

            batch = data.get_selected_passage_batch(predicting=True)
            i += 1

        return preds

    def predict_on_batch(self, sess, questions_batch, passages_batch, start_token_batch, dropout, answers_batch):
        feed = self.create_feed_dict(questions_batch, passages_batch, start_token_batch, dropout, answers_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        predictions = np.argmax(predictions, axis=2)
        return predictions

    def __init__(self, embeddings, predicting=False):
        self.predicting = predicting
        self.pretrained_embeddings = tf.Variable(embeddings)
        self.log = open(LOG_FILE_DIR, "a")
        self.build()

if __name__ == "__main__":
    print 'Starting, and now printing to log.txt'
    data = DataHolder('train')
    embeddings = EmbeddingHolder().get_embeddings_mat()
    with tf.Graph().as_default():
        start = time.time()
        model = TFModel(embeddings)
        model.log.write("\nBuild graph took " + str(time.time() - start) + " seconds")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        model.log.write('\ninitialzed variables')
        config = tf.ConfigProto()

        with tf.Session(config=config) as session:
            merged = tf.summary.merge_all()
            session.run(init)
            model.log.write('\nran init, fitting.....')
            losses = model.fit(session, saver, merged, data)

        model.train_writer.close()      
        model.test_writer.close()
        model.log.close()















