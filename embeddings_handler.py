import os
import numpy as np

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
		embeddings_index = {} # word -> embedding vector
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
		self.embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
		for word, i in word_index.iteritems():
			if i >= MAX_NB_WORDS: continue
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector
		print 'Prepared embedding matrix.'
		np.save(EMBEDDING_MAT_DIR, self.embedding_matrix)

	def get_embeddings_mat(self):
		return self.embedding_matrix
		












