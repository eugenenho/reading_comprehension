import os
import numpy as np
from collections import defaultdict

from simple_configs import EMBEDDING_DIM, VOCAB_SIZE, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

class EmbeddingHolder(object):

	def __init__(self):
		self.embedding_matrix = None
		try:
			self.embedding_matrix = np.load(EMBEDDING_MAT_DIR)
		except Exception as e:
			print 'Could not find premade embeddings matrix. creating it...'
		if self.embedding_matrix is None: 
			self.build_embeddings_mat()

	def build_embeddings_mat(self):
		embeddings_index = defaultdict(lambda: None) # word -> embedding vector
		f = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'))
		for line in f:
			values = line.strip().split(' ')
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()


		f = open(TEXT_DATA_DIR)
		word_list = [line.strip() for line in f][:VOCAB_SIZE]
		f.close()

		# prepare embedding matrix
		self.embedding_matrix = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM)
		for i, word in enumerate(word_list):
			if i == 0:
				self.embedding_matrix[i] = np.zeros((1, EMBEDDING_DIM))
				continue
			embedding_vector = embeddings_index[word.lower()]
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector
		print 'Prepared embedding matrix.'

		np.save(EMBEDDING_MAT_DIR, self.embedding_matrix)

	def get_embeddings_mat(self):
		return self.embedding_matrix.astype(np.float32)
		

if __name__ == "__main__":
	embeddings = EmbeddingHolder().get_embeddings_mat()
	print 'Lets check out data set'
	








