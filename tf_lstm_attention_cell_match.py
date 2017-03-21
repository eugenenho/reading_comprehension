import tensorflow as tf

class MatchLSTMAttnCell(tf.nn.rnn_cell.LSTMCell):
	def __init__(self, num_units, encoder_output, encoder_hidden_size, scope = None):

		# encoder_output : output tensor from Question preprocessing encoder
		# num_units : MatchLSTMAttnCell's hidden state size  				[HIDDEN_DIM]
		# encoder_hidden_size: Question proprocessing encoder's hidden size [HIDDEN_DIM]
		# here, H is same as HIDDEN_DIM, and same as self._num_units

		self.hs = encoder_output # [None x QML x H]
		self.encoder_hidden_size = encoder_hidden_size
		super(MatchLSTMAttnCell, self).__init__(num_units)
		

	def __call__(self, inputs, state, scope = None):

		# state: LSTM Tuple passed from previous MatchLSTMAttnCell (t - 1 step)
		# inputs: This time step (t)'s input: in this case, hidden state from Passage preprocessing encoder (time step t)

		previous_c, previous_h = state

		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope("MatchAttn"):  # reuse = True???

				encoder_length = tf.shape(self.hs)[1]

				W_p = tf.get_variable('W_p', shape=(self._num_units, self._num_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
				W_r = tf.get_variable('W_r', shape=(self._num_units, self._num_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            	b_p = tf.get_variable('b_p', shape=(self._num_units, ), dtype=tf.float32, initializer = tf.constant_initializer(0.0))
            	W_q = tf.get_variable('W_q', shape=(self._num_units, self._num_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

            	# inputs: [None x H]
            	z_1 = tf.matmul(inputs, W_p) + tf.matmul(previous_h, W_r) + b_p 	# [None x self._num_units (H)]
            	z_1_reshaped = tf.expand_dims(z_1, axis=1)							# [None x 1 x self._num_units]
            	
            	# self.hs: [None x QML x H]
            	W_q_broadcast = tf.zeros([tf.shape(self.hs)[0], self._num_units, self._num_units], dtype=tf.float32) + W_q
            	z_2 = tf.matmul(self.hs, W_q_broadcast) + z_1_reshaped 				# [None x QML x self._num_units]
            	G = tf.nn.tanh(z_2)													# [None x QML x self._num_units]

            	w_attn = tf.get_variable('w_attn', shape=(self._num_units, 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            	b_attn = tf.get_variable('b_attn', shape=(encoder_length, ), dtype=tf.float32, initializer = tf.constant_initializer(0.0))
            	
            	# w_attn_broadcast: [None x H x 1]
            	# G: [None x QML x H]
            	w_attn_broadcast = tf.zeros([tf.shape(self.hs)[0], self._num_units, 1], dtype=tf.float32) + w_attn 
            	s_1 = tf.matmul(G, w_attn_broadcast)			# [None x QML x 1]
            	s_1_reshaped = tf.reshape(s_1, [-1, encoder_length])			# [None x QML]
            	s_2 = s_1_reshaped + b_attn 
            	attention_mat = tf.nn.softmax(s_2)							# [None x QML]

            	# Attention compotent construction
            	attention_mat = tf.expand_dims(attention_mat, axis=2) 		# [None x QML x 1]
            	transposed_hs = tf.transpose(self.hs, perm=[0, 2, 1])		# [None x H x QML]

            	attention_component = tf.matmul(transposed_hs, attention_mat) # [None x H x 1]
            	attention_component = tf.reshape(attention_component, [-1, self._num_units]) # [None x H]

            	# Final input to the regular LSTM cell
            	concatenated_input = tf.concat(1, [previous_h, attention_component]) # [None x 2H]

		lstm_out, lstm_tuple = super(MatchLSTMAttnCell, self).__call__(concatenated_input, previous_h, scope)
		
		#output_tuple = tf.nn.rnn_cell.LSTMStateTuple(original_c, out)
		return (lstm_out, lstm_tuple)





            
            	"""
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
				"""
		
