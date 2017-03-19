import tensorflow as tf

from tf_lstm_attention_cell import LSTMAttnCell
from simple_configs import TRAIN_BATCH_SIZE, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, VOCAB_SIZE, HIDDEN_DIM


class Test():
    
    def create_cell(self):
        
        encoder_output = tf.ones([TRAIN_BATCH_SIZE, PASSAGE_MAX_LENGTH])

        with tf.variable_scope("decode"):
            d_cell = LSTMAttnCell(HIDDEN_DIM, encoder_output) # Make decoder cell with hidden dim

            # Make starter token input
            inp = tf.ones([TRAIN_BATCH_SIZE, 1000]) # STARTER TOKEN, SHAPE: [BATCH, VOCAB_SIZE]
            
            # make initial state for LSTM cell
            h_0 = tf.ones([TRAIN_BATCH_SIZE, HIDDEN_DIM]) # hidden state from passage and question
            c_0 =tf.ones([TRAIN_BATCH_SIZE, HIDDEN_DIM]) # empty memory SHAPE [BATCH, 2*HIDDEN_DIM]
            h_t = tf.nn.rnn_cell.LSTMStateTuple(c_0, h_0)
            
            for time_step in range(OUTPUT_MAX_LENGTH):
                o_t, h_t = d_cell(inp, h_t)

                U = tf.get_variable('U', shape=(2 * HIDDEN_DIM, VOCAB_SIZE), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable('b', shape=(VOCAB_SIZE, ), dtype=tf.float32)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b # SHAPE: [BATCH, VOCAB_SIZE]

                inp = y_t

                preds.append(y_t)
                tf.get_variable_scope().reuse_variables()

            packed_preds = tf.pack(preds, axis=2)
            preds = tf.transpose(packed_preds, perm=[0, 2, 1])


if __name__ == "__main__":
    testInstance = Test();
    testInstance.create_cell();

