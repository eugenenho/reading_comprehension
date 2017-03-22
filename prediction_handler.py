import json
import time
import tensorflow as tf

from embeddings_handler import EmbeddingHolder
from data_handler import DataHolder
from embeddings_handler import EmbeddingHolder

# import l2_attn
# import l2_2attn
# import l2_2attn_dbl
# import ce_attn
# import ce_2attn

from simple_configs import TEXT_DATA_DIR, VOCAB_SIZE, SAVE_MODEL_DIR


PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

DATA_SET = 'train'
OUTPUT_FILE_NAME = './data/last_run_preds.json'

class PredictionHandler():

	def get_index_word_dict(self):
		index_word = dict()
		f = open(TEXT_DATA_DIR)
		for i, line in enumerate(f):
			if i > VOCAB_SIZE: break
			word = line.strip()
			index_word[i] = word
		f.close()
		return index_word

	# Takes in a LIST of NUMPY MATRIXS [BATCH, OUTPUT_MAX_LEN]
	# Replaces indexes with words
	'''RETURNS: [str(answer), ... , str(answer)]'''
	def sub_in_words(self, preds):
		word_preds = list()
		for batch in preds:
			for row in batch:
				ans = list()
				for i in row:
					if i == END_ID: break #break on end tag
					ans.append(self.index_word[i])
				ans = ' '.join(ans)
				word_preds.append(ans)
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

	def write_preds(self, preds, output_file_name = None):
		word_preds = self.sub_in_words(preds)
		self.build_json_file(word_preds, output_file_name)

	def __init__(self, data_set=None, output_file_name = OUTPUT_FILE_NAME, build_ground_truth=False):
		# To build ground truth, uncomment these two lines and run the comamand:
		#  python prediction_handler.py
		if build_ground_truth:
			self.data_set = data_set
			self.data = DataHolder(data_set)

		self.output_file_name = output_file_name
		self.index_word = self.get_index_word_dict()




if __name__ == "__main__":
	predictor = PredictionHandler('train', OUTPUT_FILE_NAME, True) 
	predictor.get_ground_truth('./data/train_ground_truth.json')
	predictor = PredictionHandler('val', OUTPUT_FILE_NAME, True)
	predictor.get_ground_truth('./data/val_ground_truth.json')
	# predictor.get_preds(model = None, session = None, output_file_name = './data/train_preds.json')











