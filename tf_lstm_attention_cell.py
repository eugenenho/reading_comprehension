import time
import tensorflow as tf
import numpy as np

from progbar import Progbar


from simple_configs import LOG_FILE_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

class LSTMAttnCell(tf.nn.rnn_cell.LSTMCell):
	def __init__(self, num_units, encoder_output, scope = None):
		# encoder_output : output tensor from passage encoder
		# num_units : size for hidden_tilda

		self.hs = encoder_output
		super(LSTMAttnCell, self).__init__(num_units)

	def __call__(self, inputs, state, scope = None):
		lstm_out, lstm_state = super(LSTMAttnCell, self).__call__(inputs, state, scope)

		original_h, original_c = lstm_state

		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("Attn"):  # reuse = True???

				# GRAB h_t out of LSTM cell
				
				# h_t : [None x H]
				h_t = tf.nn.rnn_cell._linear(lstm_out, self._num_units, True, 1.0)

				## QUESTION: why self._num_units? vs self.num_units??

				# h_t_expanded : [None x 1 x H]
				h_t = tf.expand_dims(h_t, axis = 1)
				
				# self.hs = [None x max_time x H]
				scores = tf.reduce_sum(self.hs * h_t, reduction_indices = 2, keep_dims = True) # [None x max_time x 1]
				scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
				scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

				context = tf.reduce_sum(self.hs * scores, reduction_indices = 1) # [None x H]

				with tf.variable_scope("AttnConcat"):
					out = tf.nn.relu(tf.nn.rnn_cell._linear([context, lstm_out], self._num_units, True, 1.0))

		output_tuple = tf.nn.rnn_cell.LSTMStateTuple(out, original_c)
		return (out, output_tuple)



				# W_a : [2H x H]
				# W_a = tf.get_variable('W_a', shape=(2 * HIDDEN_DIM, HIDDEN_DIM), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)				
				# h_t_reshaped : [None x H]
				# h_t_reshaped = tf.matmul(h_t, W_a)




