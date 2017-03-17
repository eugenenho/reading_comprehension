import json
import time
import tensorflow as tf
from tensorflow_model_attn import TFAttnModel
from tensorflow_model import TFModel

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder

from simple_configs import LOG_FILE_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, PRED_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, MAX_NB_WORDS, LEARNING_RATE, DEPTH, HIDDEN_DIM, GLOVE_DIR, TEXT_DATA_DIR, EMBEDDING_MAT_DIR

DATA_SET = 'train'
MODEL_PATH = './data/model.weights'

def get_predictions(data, embeddings):
	# saver = tf.train.import_meta_graph('data/Models/model.weights.meta')
	with tf.Graph().as_default():
			start = time.time()
			# model = TFModel(embeddings)
			model = TFAttnModel(embeddings)
			print "\nRebuild graph took " + str(time.time() - start) + " seconds"
			init = tf.global_variables_initializer()
			with tf.Session() as session:
				session.run(init)
				saver = tf.train.import_meta_graph(MODEL_PATH + ".meta")
				saver.restore(session, MODEL_PATH)
				predictions = model.predict(session, saver, data)
	return predictions

def get_ground_truth(data, index_word):
	A_data = list()
	batch = data.get_batch(batch_size=PRED_BATCH_SIZE)
	while batch is not None:
		A_batch = batch[2]
		A_data.append(A_batch)
		batch = data.get_batch(batch_size=PRED_BATCH_SIZE)

	word_truths = sub_in_word(A_data, index_word)
	build_json_file(word_truths, DATA_SET + '_ground_truth_json.json')


def get_index_word_dict():
	index_word = dict()
	f = open(TEXT_DATA_DIR)
	for i, line in enumerate(f):
		word = line.lower().strip()
		index_word[i] = word
	return index_word

def sub_in_word(preds, index_word):
	word_preds = list()

	for batch in preds:
		for row in batch:
			ans = list()
			for i in row:
				ans.append(index_word[i])
				if i == 1: break #break on end tag
			ans = ' '.join(ans)
			word_preds.append(ans)

	return word_preds

def build_json_file(preds, file_name):
	f = open(file_name, 'w')
	for i, p in enumerate(preds):
		curr_ans = dict()
		curr_ans["answers"] = [p]
		curr_ans["query_id"] = i
		f.write( json.dumps(curr_ans) )
		f.write('\n')
	f.close()

if __name__ == "__main__":
	embeddings = EmbeddingHolder().get_embeddings_mat()
	data = TFDataHolder(DATA_SET, PRED_BATCH_SIZE)
	
	index_word = get_index_word_dict()

	preds = get_predictions(data, embeddings)
	preds = sub_in_word(preds, index_word)
	build_json_file(preds, DATA_SET + '_preds_json.json')

	get_ground_truth(data, index_word)




























