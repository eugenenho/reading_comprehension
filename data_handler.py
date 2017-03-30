import os
import numpy as np
import cPickle
import h5py

from simple_configs import TRAIN_BATCH_SIZE, EMBEDDING_DIM, VOCAB_SIZE, OUTPUT_MAX_LENGTH, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, MAX_NUM_PASSAGES, MAX_DATA_SIZE, DROPOUT, SMALL_DATA_SET

PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

class DataHolder(object):

	# pass in data_set ('train', 'val', 'dev')
	def __init__(self, DATA_SET):
		print '\nInitializing updated Data Holder'
		self.data_set = str(DATA_SET).lower()
		DATA_FILE = './data/marco/h5py-' +  self.data_set

		if not os.path.isfile(DATA_FILE):
			h5f = h5py.File(DATA_FILE, 'w')
			h5f.create_dataset('q_data', data=self.build_q_data())
			one_row_p = np.load('./data/marco/train.data.one_row_p.npy')
			print one_row_p.shape
			h5f.create_dataset('one_row_p', data=one_row_p)


			p_data, selected_passage = self.build_p_data()
			h5f.create_dataset('p_data', data=p_data)
			h5f.create_dataset('selected_passage', data=selected_passage)

			h5f.create_dataset('a_data', data=self.build_a_data())
			h5f.close()

		h5f = h5py.File(DATA_FILE,'r')
		self.q_data = h5f['q_data']
		self.p_data = h5f['p_data']
		self.a_data = h5f['a_data']
		self.selected_passage = h5f['selected_passage']
		self.one_row_p_data = h5f['one_row_p']
		self.data_size = self.q_data.shape[0] if MAX_DATA_SIZE == -1 else min(MAX_DATA_SIZE, self.q_data.shape[0])

		self.start_token = self.build_start_token()

		self.start_iter = 0

	# This constructs the data from the pickled objects
	def build_q_data(self):
		questions_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.question.pkl","rb"))
		self.data_size = len(questions_list)
		q_data = np.zeros((self.data_size, QUESTION_MAX_LENGTH))
		for i, question in enumerate(questions_list):
			# padding
			if len(question) < QUESTION_MAX_LENGTH:
				pad = [PAD_ID] * (QUESTION_MAX_LENGTH - len(question))
				question.extend(pad)

			q_data[i] = np.array(question[:QUESTION_MAX_LENGTH])
		# Replace vals > VOCAB_SIZE with unknown
		q_data = np.where(q_data < VOCAB_SIZE, q_data, UNK_ID)

		print 'built q data'
		return q_data

	# This constructs the data from the pickled objects
	def build_p_data(self):
		# doens't contain 'is selected'
		passages_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.passage.pkl","rb"))

		# Used only for is_selected
		passage_words_list = cPickle.load(open("./data/marco/" + self.data_set + ".passage.pkl", "rb"))

		p_data = np.zeros((self.data_size, MAX_NUM_PASSAGES, PASSAGE_MAX_LENGTH)) # [None x 10 x passage_max_length]
		selected_passage = np.zeros((self.data_size))							  # [None]

		for entry_num, p_l in enumerate(passages_list):
			for passage_num, passage in enumerate(p_l):
				if passage_num >= MAX_NUM_PASSAGES: break
				# padding
				if len(passage) < PASSAGE_MAX_LENGTH and passage_num < MAX_NUM_PASSAGES:
					pad = [PAD_ID] * (PASSAGE_MAX_LENGTH - len(passage))
					passage.extend(pad)

				# save which passsage is selected
				if passage_num < len(passage_words_list[entry_num]):
					is_selected = passage_words_list[entry_num][passage_num]
					if is_selected > 0:
						selected_passage[entry_num] = passage_num

				p_data[entry_num][passage_num] = np.array(passage[:PASSAGE_MAX_LENGTH])

		# Replace vals > VOCAB_SIZE with unknown
		p_data = np.where(p_data < VOCAB_SIZE, p_data, UNK_ID)

		print 'built p data'
		return p_data, selected_passage

	# This constructs the data from the pickled objects
	def build_one_row_p_data(self):
		# doens't contain 'is selected'
		passages_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.passage.pkl","rb"))

		# Used only for is_selected
		passage_words_list = cPickle.load(open("./data/marco/" + self.data_set + ".passage.pkl", "rb"))

		p_data = np.zeros((self.data_size, PASSAGE_MAX_LENGTH * MAX_NUM_PASSAGES)) # [None x 10 x passage_max_length]
		selected_passage = np.zeros((self.data_size))							  # [None]

		for entry_num, p_l in enumerate(passages_list):
			passages = []
			for passage_num, p in enumerate(p_l):
				if passage_num >= MAX_NUM_PASSAGES: break
				passages.extend(p)
				passages.append(END_ID)

				# save which passsage is selected
				if passage_num < len(passage_words_list[entry_num]):
					is_selected = passage_words_list[entry_num][passage_num]
					if is_selected > 0:
						selected_passage[entry_num] = passage_num

			passages.extend( [PAD_ID] * (MAX_NUM_PASSAGES * PASSAGE_MAX_LENGTH - len(passages)) )
			passages_arr = np.array(passages[:PASSAGE_MAX_LENGTH * MAX_NUM_PASSAGES])
			print passages_arr.shape, p_data.shape
			p_data[entry_num] = passages_arr

		# Replace vals > VOCAB_SIZE with unknown
		p_data = np.where(p_data < VOCAB_SIZE, p_data, UNK_ID)
		np.save("./data/marco/" + self.data_set + ".data.one_row_p", p_data)

		print 'built one row p data'
		return p_data, selected_passage


	def build_a_data(self):
		answers_list = cPickle.load(open("./data/marco/" + self.data_set + ".ids.answer.pkl","rb" ))
		a_data = np.zeros((self.data_size, OUTPUT_MAX_LENGTH))
		for i, ans in enumerate(answers_list):
			# weird thing here, the answer is stores as a list of lists
			ans = ans[0] if len(ans) >= 1 else []
			# pad
			if len(ans) < OUTPUT_MAX_LENGTH: 
				pad = [END_ID] + [PAD_ID] * (OUTPUT_MAX_LENGTH - len(ans) - 1)
				ans.extend(pad)
			# add to matrix
			a_data[i] = np.array(ans[:OUTPUT_MAX_LENGTH])

		# Replace vals > VOCAB_SIZE with unknown
		a_data = np.where(a_data < VOCAB_SIZE, a_data, UNK_ID)

		print 'built y data'
		return a_data

	def build_start_token(self):
		return np.zeros((TRAIN_BATCH_SIZE)) + STR_ID

	def get_batch(self, predicting=False):
		if self.start_iter >= self.data_size:
			self.start_iter = 0
			return None

		start = self.start_iter
		end = min(self.data_size, self.start_iter + TRAIN_BATCH_SIZE)
		batch_size = end - self.start_iter

		self.start_iter += batch_size

		# tf.cast( tf.sequence_mask(self.seq_length(self.answers_placeholder), OUTPUT_MAX_LENGTH), tf.float32)
		elements_as_ones = np.sign(np.abs(self.a_data[start:end]))
		seq_length = np.sum(elements_as_ones, axis=1)
		answer_mask = list()
		for i in seq_length:
			curr_mask = [True] * int(i) + [False] * (OUTPUT_MAX_LENGTH - int(i))
			answer_mask.append(curr_mask)
		answer_mask = np.array(answer_mask)
		
		if SMALL_DATA_SET:
			return {
				'question' : self.q_data[:batch_size], 
				'passage' : self.p_data[:batch_size], 
				'selected_passage' : self.selected_passage[:batch_size],
				'answer' : self.a_data[:batch_size], 
				'answer_mask' : answer_mask,
				'start_token' : self.start_token[:batch_size],
				'dropout' : 1
				}

		return {
				'question' : self.q_data[start:end], 
				'passage' : self.p_data[start:end], 
				'selected_passage' : self.selected_passage[start:end],
				'answer' : self.a_data[start:end], 
				'answer_mask' : answer_mask,
				'start_token' : self.start_token[:batch_size],
				'dropout' : DROPOUT if not predicting else 1
				}

	def get_selected_passage_batch(self, predicting=False):
		full_batch = self.get_batch(predicting)
		if full_batch is None: return None

		all_passages = full_batch['passage']
		selected_passage = full_batch['selected_passage']

		batch_size = all_passages.shape[0]
		passages_mat = np.zeros((batch_size, PASSAGE_MAX_LENGTH)) # [None x passage_max]

		
		for i in range(batch_size):
			sel = int(selected_passage[i])
			passages_mat[i] = all_passages[i][sel]
		full_batch['passage'] = passages_mat
		full_batch['selected_passage'] = None
		return full_batch

	def get_classifier_batch(self, predicting=False):
		if self.start_iter >= self.data_size:
			self.start_iter = 0
			return None

		start = self.start_iter
		end = min(self.data_size, self.start_iter + TRAIN_BATCH_SIZE)
		batch_size = end - self.start_iter
		
		self.start_iter += batch_size

		return {
				'question' : self.q_data[start:end], 
				'passage' : self.one_row_p_data[start:end], 
				'answer' : self.selected_passage[start:end], 
				'dropout' : DROPOUT if not predicting else 1
				}

	def reset_iter(self):
		self.start_iter = 0

	def get_full_selected(self):
		return self.selected_passage[:self.data_size]


if __name__ == "__main__":
	data_module = DataHolder('val')
	print 'Lets check out data set'
	print 'Length of Q_data: ', data_module.q_data.shape
	print 'Length of P_data ', data_module.p_data.shape
	print 'Length of A_data ', data_module.a_data.shape
	print 'Length of Start Token ', data_module.start_token.shape
	print 'Data Size ', data_module.data_size

	print 'Making batch predicting'
	print len(data_module.get_batch(True))
	print 'Making batch Training'
	print len(data_module.get_batch(False))








