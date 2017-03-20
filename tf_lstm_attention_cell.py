import tensorflow as tf

class LSTMAttnCell(tf.nn.rnn_cell.LSTMCell):
	def __init__(self, num_units, encoder_output, encoder_hidden_size, scope = None):
		# encoder_output : output tensor from passage encoder
		# num_units : size for hidden_tilda

		self.hs = encoder_output
		self.encoder_hidden_size = encoder_hidden_size
		super(LSTMAttnCell, self).__init__(num_units)

		self.hidden_states = None

	def get_hidden_states(self):
		return self.hidden_states

	def clear_hidden_states(self):
		self.hidden_states = None

	def __call__(self, inputs, state, scope = None):
		lstm_out, lstm_state = super(LSTMAttnCell, self).__call__(inputs, state, scope)

		#original_h, original_c = lstm_state
		original_c, original_h = lstm_state

		if self.hidden_states is None:
			self.hidden_states = tf.reshape(original_h, [-1, 1, self._num_units])
		else:
			self.hidden_states = tf.concat(1, [self.hidden_states, tf.reshape(original_h, [-1, 1, self._num_units])])
		print "length of hidden_states", self.hidden_states

		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("Attn"):  # reuse = True???

				# GRAB h_t out of LSTM cell
				
				# h_t : [None x H]		[None x 3H]
				
				h_t = tf.nn.rnn_cell._linear(lstm_out, self.encoder_hidden_size, True, 1.0)
				
				## QUESTION: why self._num_units? vs self.num_units??

				# h_t_expanded : [None x 1 x H]       [None x 1 x 3H]
				h_t = tf.expand_dims(h_t, axis = 1)
				
				# self.hs = [None x max_time x H]       
				scores = tf.reduce_sum(self.hs * h_t, reduction_indices = 2, keep_dims = True) # [None x max_time x 1]
				scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
				scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

				context = tf.reduce_sum(self.hs * scores, reduction_indices = 1) # [None x H]

				with tf.variable_scope("AttnConcat"):
					out = tf.nn.relu(tf.nn.rnn_cell._linear([context, lstm_out], self._num_units, True, 1.0))

		# print "out dimensions", out
		# print "original_h dimensions", original_h
		# h_tilda_and_h = tf.concat(1, [out, original_h])
		# print "original_htidla and h", h_tilda_and_h
		
		output_tuple = tf.nn.rnn_cell.LSTMStateTuple(original_c, out)
		return (out, output_tuple)
