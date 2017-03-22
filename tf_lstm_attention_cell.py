import tensorflow as tf

class LSTMAttnCell(tf.nn.rnn_cell.LSTMCell):
	def __init__(self, num_units, encoder_output, encoder_hidden_size, scope = None, activation=tf.nn.tanh):
		# encoder_output : output tensor from passage encoder
		# num_units : size for hidden_tilda

		self.hs = encoder_output
		self.encoder_hidden_size = encoder_hidden_size
		super(LSTMAttnCell, self).__init__(num_units, activation=activation)

	def __call__(self, inputs, state, scope = None):
		lstm_out, lstm_tuple = super(LSTMAttnCell, self).__call__(inputs, state, scope)

		original_c, original_h = lstm_tuple

#####DEBUGGING:
		temp1 = tf.Print(original_h, [original_h], message = "original_h vector :", summarize = 16 * 5)
		temp1 = temp1 + temp1
		temp2 = tf.Print(lstm_out, [lstm_out], message = "lstm_out vector :", summarize = 16 * 5)
		temp2 = temp2 + temp1
		temp1 = tf.multiply(temp1, temp2)
		

		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("Attn"):  # reuse = True???

				# GRAB h_t out of LSTM cell
				# h_t : [None x H]		[None x 3H]				
				h_t = tf.nn.rnn_cell._linear(lstm_out, self.encoder_hidden_size, True, 1.0)
				
				# h_t_expanded : [None x 1 x H]       [None x 1 x 3H]
				h_t = tf.expand_dims(h_t, axis = 1)
				
				# self.hs = [None x max_time x H]       
				scores = tf.reduce_sum(self.hs * h_t, reduction_indices = 2, keep_dims = True) # [None x max_time x 1]
				scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
				scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))
				context = tf.reduce_sum(self.hs * scores, reduction_indices = 1) # [None x H]

				with tf.variable_scope("AttnConcat"):
					out = tf.nn.relu(tf.nn.rnn_cell._linear([context, lstm_out], self._num_units, True, 1.0))
					# CONFIRM REAL VALUES / DIMS

				temp3 = tf.Print(context, [context], message = "\n\ncontext vector :", summarize = 16 * 5)					
				temp3 = temp3 + tf.concat(2, [temp2, tf.concat(2, [temp1, temp1])])

#####DEBUGGING:
		
		out = tf.Print(out, [out], message = "out vector :", summarize = 16 * 5)			
					
		output_tuple = tf.nn.rnn_cell.LSTMStateTuple(original_c, out)
		return (out, output_tuple)
