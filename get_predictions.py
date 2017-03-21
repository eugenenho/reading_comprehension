import json
import time
import tensorflow as tf

from embeddings_handler import EmbeddingHolder
from data_handler import DataHolder
from embeddings_handler import EmbeddingHolder

import l2_attn
import l2_2attn
import l2_2attn_dbl

import ce_attn
import ce_2attn

from simple_configs import TEXT_DATA_DIR, VOCAB_SIZE, SAVE_MODEL_DIR


PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

DATA_SET = 'train'
OUTPUT_FILE_NAME = './data/last_run_preds.json'

class AnswerPredictor():

	def get_index_word_dict(self):
		index_word = dict()
		f = open(TEXT_DATA_DIR)
		for i, line in enumerate(f):
			if i > VOCAB_SIZE: break
			word = line.strip()
			index_word[i] = word
		f.close()
		return index_word

	def get_model_and_preds(self):
		with tf.Graph().as_default():
			start = time.time()

			if self.model_type == 'l2_attn': model = l2_attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
			elif self.model_type == 'l2_2attn': model = l2_2attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
			elif self.model_type == 'l2_2attn_dbl': model = l2_2attn_dbl.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
			elif self.model_type == 'ce_attn': model = ce_attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
			elif self.model_type == 'ce_2attn': model = ce_2attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)

			print "\nRebuild graph took " + str(time.time() - start) + " seconds"

			saver = tf.train.Saver()

			with tf.Session() as session:
				self.session = session
				saver.restore(session, SAVE_MODEL_DIR)
				print 'Restored model'
				print 'Predicting....'
				self.preds = model.predict(self.session, self.data)
		print 'Done Predicting'


	def build_predictions(self):
		print 'Starting predictions'
		self.preds = self.model.predict(self.session, self.data)
		print 'finished predicting'

	# Takes in a LIST of NUMPY MATRIXS [BATCH, OUTPUT_MAX_LEN]
	# Replaces indexes with words
	'''RETURNS: [str(answer), ... , str(answer)]'''
	def sub_in_words(self, preds=None):
		print 'Substituting Words...'
		if preds is None: preds = self.preds

		word_preds = list()
		for batch in preds:
			for row in batch:
				ans = list()
				for i in row:
					ans.append(self.index_word[i])
					if i == END_ID: break #break on end tag
				ans = ' '.join(ans)
				word_preds.append(ans)
		print 'Done substituting'
		return word_preds

	def build_json_file(self, word_ans, file_name=None):
		if file_name is None: file_name = self.output_file_name

		f = open(file_name, 'w')
		for i, p in enumerate(word_ans):
			curr_ans = dict()
			curr_ans["answers"] = [p]
			curr_ans["query_id"] = i
			f.write( json.dumps(curr_ans) )
			f.write('\n')
		f.close()

	def get_ground_truth(self, output_file_name = './data/last_run_ground_truth.json'):
		self.data.reset_iter()

		A_data = list()
		batch = self.data.get_selected_passage_batch()
		while batch is not None:
			A_batch = batch['answer']
			A_data.append(A_batch)
			batch = self.data.get_selected_passage_batch()

		word_truths = self.sub_in_words(A_data)
		self.build_json_file(word_truths, output_file_name)

	def get_preds(self, model_path = SAVE_MODEL_DIR, model_type = None, model = None, session = None, output_file_name = None):
		if output_file_name is not None:
		 	self.output_file_name = output_file_name
		
		self.data.reset_iter()
		
		self.model_path = model_path
		self.model_type = model_type

		# must pass in a model
		if model is not None: 
			assert session is not None
			self.model = model
			self.session = session
			self.build_predictions()
		else:
			assert model_type is not None
			self.model = self.get_model_and_preds()
	
		word_preds = self.sub_in_words()
		self.build_json_file(word_preds)

	def __init__(self, data_set, output_file_name = OUTPUT_FILE_NAME):
		self.data_set = data_set
		self.output_file_name = output_file_name
		self.data = DataHolder(data_set)
		self.index_word = self.get_index_word_dict()




if __name__ == "__main__":
	predictor = AnswerPredictor('train') 
	predictor.get_preds(model_path = './data/model.weights', model_type = 'l2_attn', model = None, session = None, output_file_name = './data/train_preds.json')
	predictor.get_ground_truth()










