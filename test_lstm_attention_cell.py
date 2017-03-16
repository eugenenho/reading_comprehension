import time
import tensorflow as tf
import numpy as np

from progbar import Progbar

from tf_lstm_attention_cell import LSTMAttnCell

from simple_configs import LOG_FILE_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR


class Test():
    
    def create_cell(self):
        
        encoder_output = tf.ones([TRAIN_BATCH_SIZE, PASSAGE_MAX_LENGTH])

        with tf.variable_scope("decode"):
            d_cell = LSTMAttnCell(HIDDEN_DIM, encoder_output) # Make decoder cell with hidden dim

            # Make starter token input
            inp = tf.ones([TRAIN_BATCH_SIZE, 1000]) # STARTER TOKEN, SHAPE: [BATCH, MAX_NB_WORDS]
            
            # make initial state for LSTM cell
            h_0 = tf.ones([TRAIN_BATCH_SIZE, HIDDEN_DIM]) # hidden state from passage and question
            c_0 =tf.ones([TRAIN_BATCH_SIZE, HIDDEN_DIM]) # empty memory SHAPE [BATCH, 2*HIDDEN_DIM]
            h_t = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            
            for time_step in range(OUTPUT_MAX_LENGTH):
                o_t, h_t = d_cell(inp, h_t)

                U = tf.get_variable('U', shape=(2 * HIDDEN_DIM, MAX_NB_WORDS), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable('b', shape=(MAX_NB_WORDS, ), dtype=tf.float32)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b # SHAPE: [BATCH, MAX_NB_WORDS]

                inp = y_t

                preds.append(y_t)
                tf.get_variable_scope().reuse_variables()

            packed_preds = tf.pack(preds, axis=2)
            preds = tf.transpose(packed_preds, perm=[0, 2, 1])


if __name__ == "__main__":
    testInstance = Test();
    testInstance.create_cell();

