import numpy as np
import cPickle

from sklearn.preprocessing import OneHotEncoder
from simple_configs import TRAIN_BATCH_SIZE, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH

class TFDataHolder:

	# pass in data_set ('train', 'val', 'dev')
	def __init__(self, DATA_SET):
		print 'initializing tf Data Holder'
		self.data_set = str(DATA_SET).lower()

		found = False
		try:
			self.Q_data = np.load("./data/marco/" + self.data_set + ".data.q_data.npy")
			self.P_data = np.load("./data/marco/" + self.data_set + ".data.p_data.npy")
			self.A_data = np.load("./data/marco/" + self.data_set + ".data.a_data.npy")
			found = True
		except Exception as e:
			print 'could not find premade tf matrix. creating it...'

		if not found:
			self.Q_data = self.build_Q_data()
			self.P_data = self.build_P_data()
			self.A_data = self.build_A_data() 

		self.data_size = self.Q_data.shape[0]
		self.Q_data = self.Q_data[:self.data_size]
		self.P_data = self.P_data[:self.data_size]
		self.A_data = self.A_data[:self.data_size]
		self.start_token = self.build_start_token()

		self.start_iter = 0


	# This constructs the data from the pickled objects
	def build_Q_data(self):
		questions_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.question.pkl","rb"))
		self.data_size = len(questions_list)
		Q_data = np.zeros((self.data_size, QUESTION_MAX_LENGTH))
		for i, question in enumerate(questions_list):
			# padding
			if len(question) < QUESTION_MAX_LENGTH:
				question.extend([MAX_NB_WORDS - 1] + [0] * (QUESTION_MAX_LENGTH - len(question) - 1) )

			Q_data[i] = np.array(question[:QUESTION_MAX_LENGTH])
		np.save("./data/marco/" + self.data_set + ".data.q_data", Q_data)

		print 'built q data'
		return Q_data

	# This constructs the data from the pickled objects
	def build_P_data(self):
		passages_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.passage.pkl","rb"))
		P_data = np.zeros((self.data_size, PASSAGE_MAX_LENGTH))
		for i, p_l in enumerate(passages_list):
			for passage in p_l:
				# padding
				if len(passage) < PASSAGE_MAX_LENGTH:
					passage.extend([0] * (PASSAGE_MAX_LENGTH - len(passage)) )

				P_data[i] = np.array(passage[:PASSAGE_MAX_LENGTH])

		np.save("./data/marco/" + self.data_set + ".data.p_data", P_data)

		print 'built p data'
		return P_data


	def build_A_data(self):
		answers_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.answer.pkl","rb" ))
		A_data_indexes = np.zeros((self.data_size, OUTPUT_MAX_LENGTH))
		for i, ans in enumerate(answers_list):
			# weird thing here, the answer is stores as a list of lists
			ans = ans[0] if len(ans) >= 1 else []
			# pad / truncate values
			if len(ans) < OUTPUT_MAX_LENGTH: ans.extend( [0] * (OUTPUT_MAX_LENGTH - len(ans)) )
			# add to matrix
			A_data_indexes[i] = np.array(ans[:OUTPUT_MAX_LENGTH])

		A_data_indexes = A_data_indexes.astype(int)
		np.save("./data/marco/" + self.data_set + ".data.a_data", A_data_indexes)

		print 'built y data'
		return A_data_indexes


	def build_start_token(self):
		token_mat = np.zeros((TRAIN_BATCH_SIZE, MAX_NB_WORDS))
		for row in token_mat:
			row[1] = 1
		return token_mat

	def get_full_data(self):
		print 'building full Y data'
		return self.Q_data, self.P_data, self.A_data, self.start_token

	def get_batch(self):
		if self.start_iter >= self.data_size:
			self.start_iter = 0
			return None
		end = min(self.data_size, self.start_iter + TRAIN_BATCH_SIZE)
		batch_size = end - self.start_iter
		to_return = (
				self.Q_data[self.start_iter:end], 
				self.P_data[self.start_iter:end], 
				self.A_data[self.start_iter:end], 
				self.start_token[:batch_size]
				)
		self.start_iter += TRAIN_BATCH_SIZE
		return to_return

if __name__ == "__main__":
	print 'Lets check out data set'
	data_module = TFDataHolder('train')
	print 'Length of Q_data:', data_module.Q_data.shape
	print 'Length of P_data', data_module.P_data.shape
	print 'Length of A_data', data_module.A_data.shape
	print 'Length of Start Token', data_module.start_token.shape
	print 'Data Size', data_module.data_size

	print 'Making batch'
	batch = data_module.get_batch()
	print 'Batch Q', batch[0].shape
	print 'Batch P', batch[1].shape
	print 'Batch A', batch[2].shape
	print 'Batch ST', batch[3].shape














