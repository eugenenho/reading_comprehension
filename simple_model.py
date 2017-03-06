import os
import sys

import seq2seq
from keras.models import Sequential
from seq2seq.models import Seq2Seq
from keras.layers.embeddings import Embedding
import numpy as np

import simple_configs

# (need to add masking)
def build_model():
	model = Sequential()

	embeddings_index = {} # word -> embedding vector
	f = open(os.path.join(simple_configs.GLOVE_DIR, 'glove.6B.' + str(simple_configs.EMBEDDING_DIM) + 'd.txt'))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	print 'built word map'

	word_index = {}  # dictionary mapping label name to numeric id
	f = open(simple_configs.TEXT_DATA_DIR)
	for i, line in enumerate(f):
		word_index[line] = i
	print 'made vocabulary'

	# prepare embedding matrix
	embedding_matrix = np.zeros((simple_configs.MAX_NB_WORDS, simple_configs.EMBEDDING_DIM))
	for word, i in word_index.iteritems():
	    if i >= simple_configs.MAX_NB_WORDS: continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        embedding_matrix[i] = embedding_vector
	print 'Prepared embedding matrix.'

	# BUILD ACTUAL MODEL
	embedding_layer = Embedding(
			simple_configs.MAX_NB_WORDS,
		    simple_configs.EMBEDDING_DIM,
		    weights=[embedding_matrix],
		    input_length=simple_configs.INPUT_MAX_LENGTH
	    )
	s2s_layer = Seq2Seq(
		batch_input_shape=(None, simple_configs.INPUT_MAX_LENGTH, simple_configs.EMBEDDING_DIM),
		hidden_dim=simple_configs.HIDDEN_DIM, 
		output_length=simple_configs.OUTPUT_MAX_LENGTH, 
		output_dim=simple_configs.MAX_NB_WORDS, 
		depth=simple_configs.DEPTH
	)

	# ADD UP ACTUAL MODEL
	model.add(embedding_layer)
	model.add(s2s_layer)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	print 'Model Built'
	return model
