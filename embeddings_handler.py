import os
import numpy as np
from collections import defaultdict

from simple_configs import EMBEDDING_DIM, MAX_NB_WORDS, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

class EmbeddingHolder(object):

	def __init__(self):
		self.embedding_matrix = None
		try:
			self.embedding_matrix = np.load(EMBEDDING_MAT_DIR)
		except Exception as e:
			print 'could not find premade embeddings matrix. creating it...'
		if self.embedding_matrix is None: 
			self.build_embeddings_mat()

	def build_embeddings_mat(self):
		embeddings_index = defaultdict(lambda: None) # word -> embedding vector
		f = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'))
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()

		word_index = {}  # dictionary mapping label name to numeric id
		f = open(TEXT_DATA_DIR)
		for i, line in enumerate(f):
			word = line.lower().strip()
			word_index[word] = i if word not in word_index else word_index[word]
		size_of_mat = i + 1

		# prepare embedding matrix
		self.embedding_matrix = np.zeros((size_of_mat, EMBEDDING_DIM))
		for word, i in word_index.iteritems():
			embedding_vector = embeddings_index[word]
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector

		print 'the check embedding:', self.embedding_matrix[4]
		print 'was check embedding:', self.embedding_matrix[41]
		print 'zeros check:', self.embedding_matrix[0]

		print 'Prepared embedding matrix.'

		np.save(EMBEDDING_MAT_DIR, self.embedding_matrix)

	def get_embeddings_mat(self):
		return self.embedding_matrix
		












