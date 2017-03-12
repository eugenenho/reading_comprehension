import seq2seq
from keras.models import Sequential
from seq2seq.models import Seq2Seq
from keras.layers.embeddings import Embedding
import numpy as np

from embeddings_handler import EmbeddingHolder
from simple_configs import EMBEDDING_DIM, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

def build_model():
	model = Sequential()

	embedding_matrix = EmbeddingHolder().get_embeddings_mat()
	
	# BUILD ACTUAL MODEL
	embedding_layer = Embedding(
			228999,
		    EMBEDDING_DIM,
		    weights=[embedding_matrix],
		    input_length=INPUT_MAX_LENGTH,
		    mask_zero=True
    )
	s2s_layer = Seq2Seq(
		batch_input_shape=(None, INPUT_MAX_LENGTH, EMBEDDING_DIM),
		hidden_dim=HIDDEN_DIM, 
		output_length=OUTPUT_MAX_LENGTH, 
		output_dim=228999, 
		depth=DEPTH
	)
	
	# ADD UP ACTUAL MODEL
	model.add(embedding_layer)
	model.add(s2s_layer)
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])

	print 'Model Built'
	return model
