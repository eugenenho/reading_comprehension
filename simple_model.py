import os
import sys

import seq2seq
from keras.models import Sequential
from seq2seq.models import Seq2Seq
from keras.layers.embeddings import Embedding
import numpy as np


# need to add masking
def build_model():
	GLOVE_DIR = './download/dwr/'
	TEXT_DATA_DIR = './data/marco/vocab.dat'
	INPUT_MAX_LENGTH = 1000
	OUTPUT_MAX_LENGTH = 15
	MAX_NB_WORDS = 200000
	EMBEDDING_DIM = 100

	model = Sequential()

	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	print 'built word map'

	word_index = {}  # dictionary mapping label name to numeric id
	f = open(TEXT_DATA_DIR)
	for i, line in enumerate(f):
		word_index[line] = i
	print 'made vocabulary'

	# prepare embedding matrix
	embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
	for word, i in word_index.iteritems():
	    if i >= MAX_NB_WORDS: continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        embedding_matrix[i] = embedding_vector
	print 'Prepared embedding matrix.'

	embedding_layer = Embedding(
			MAX_NB_WORDS,
		    EMBEDDING_DIM,
		    weights=[embedding_matrix],
		    input_length=INPUT_MAX_LENGTH
	    )
	s2s_layer = Seq2Seq(
		batch_input_shape=(None, INPUT_MAX_LENGTH, EMBEDDING_DIM),
		hidden_dim=50, 
		output_length=OUTPUT_MAX_LENGTH, 
		output_dim=200000, 
		depth=4
	)

	model.add(embedding_layer)
	model.add(s2s_layer)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	print 'Model Built'
	return model
