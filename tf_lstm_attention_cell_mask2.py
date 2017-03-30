import tensorflow as tf
from simple_configs import LOG_FILE_DIR, SAVE_MODEL_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, VOCAB_SIZE, LEARNING_RATE, HIDDEN_DIM, MAX_GRAD_NORM, ACTIVATION_FUNC 

class LSTMAttnCell(tf.nn.rnn_cell.LSTMCell):
	def __init__(self, num_units, encoder_output, encoder_hidden_size, encoder_seq_length, scope = None, activation=tf.nn.tanh):
		# encoder_output : output tensor from passage encoder
		# num_units : size for hidden_tilda

		self.hs = encoder_output
		self.encoder_hidden_size = encoder_hidden_size
		self.encoder_seq_length = encoder_seq_length
		super(LSTMAttnCell, self).__init__(num_units, activation=activation)

	def __call__(self, inputs, state, scope = None):
		lstm_out, lstm_tuple = super(LSTMAttnCell, self).__call__(inputs, state, scope)

		original_c, original_h = lstm_tuple

##### DEBUG ######
#		original_h = tf.Print(original_h, [original_h], message = "original_h vector :", summarize = self._num_units * TRAIN_BATCH_SIZE)
##################

		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("Attn"):  # reuse = True???

				# GRAB h_t out of LSTM cell
				# h_t : [None x H]		[None x 3H]				
				h_t = tf.nn.rnn_cell._linear(original_h, self.encoder_hidden_size, True, 1.0)
				
				# h_t_expanded : [None x 1 x H]       [None x 1 x 3H]
				h_t = tf.expand_dims(h_t, axis = 1)
				
				# self.hs = [None x max_time x H]       
				scores = tf.reduce_sum(self.hs * h_t, reduction_indices = 2, keep_dims = True) # [None x max_time x 1]

##### DEBUG ######
#				scores = tf.Print(scores, [scores], message = "scores vector pre-processing :", summarize = TRAIN_BATCH_SIZE * PASSAGE_MAX_LENGTH)			
##################


				# get mask matrix before softmax process
				# mask = tf.sign(tf.abs(tf.identity(scores))) # [None x max_time x 1] same shape, but 0 for all 0, 1 for all non zero values
				
				mask = tf.sequence_mask(self.encoder_seq_length, tf.shape(self.hs)[1]) #[None x max_time] of True, False
				mask = tf.expand_dims(mask, 2) #[None x max_time x 1]
				mask = tf.cast(mask, tf.int32)


##### DEBUG ######
#				mask = tf.Print(mask, [mask], message = "mask vector pre-processing :", summarize = TRAIN_BATCH_SIZE * PASSAGE_MAX_LENGTH)			
##################

				# get usual softmax values
				scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
				scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

				# apply mask
				scores = scores * mask # [None x max_time x 1] same shape, but masking out all values that were initially 0

##### DEBUG ######
#				scores = tf.Print(scores, [scores], message = "scores vector: after scores * mask:", summarize = TRAIN_BATCH_SIZE * PASSAGE_MAX_LENGTH)			
##################			
				
				# apply factors
				scores = scores * (1 / tf.reduce_sum(scores, reduction_indices = 1, keep_dims=True))

##### DEBUG ######
#				scores = tf.Print(scores, [scores], message = "scores vector: after applying factors:", summarize = TRAIN_BATCH_SIZE * PASSAGE_MAX_LENGTH)			
##################			



				context = tf.reduce_sum(self.hs * scores, reduction_indices = 1) # [None x H]

				with tf.variable_scope("AttnConcat"):
					out = tf.nn.relu(tf.nn.rnn_cell._linear([context, original_h], self._num_units, True, 1.0))
				
##### DEBUG ######
#		out = tf.Print(out, [out], message = "out vector :", summarize = self._num_units * TRAIN_BATCH_SIZE)			
##################		

		output_tuple = tf.nn.rnn_cell.LSTMStateTuple(original_c, out)
		return (out, output_tuple)
