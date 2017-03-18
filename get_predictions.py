import json
import time
import tensorflow as tf

from embeddings_handler import EmbeddingHolder
from tf_data_handler import TFDataHolder
from embeddings_handler import EmbeddingHolder

DATA_SET = 'train'
MODEL_PATH = './data/Models/1direction_attn_lstm_with_embeddings_passed_in/model.weights'

# ~~~ DOESN'T WORK ~~~
# def get_preds(data, embeddings):
# 	# saver = tf.train.import_meta_graph('data/Models/model.weights.meta')
# 	with tf.Graph().as_default():
# 		start = time.time()
# 		model = TFModel(embeddings, True)
# 		print "\nRebuild graph took " + str(time.time() - start) + " seconds"
# 		init = tf.global_variables_initializer()
# 		with tf.Session() as session:
# 			session.run(init)
# 			saver = tf.train.import_meta_graph(MODEL_PATH + ".meta")
# 			saver.restore(session, MODEL_PATH)
# 			predictions = model.predict(session, saver, data)
# 	print 'predictions', predictions
# 	return predictions

def get_ground_truth(data, index_word):
	A_data = list()
	batch = data.get_batch()
	while batch is not None:
		A_batch = batch[2]
		A_data.append(A_batch)
		batch = data.get_batch()

	word_truths = sub_in_word(A_data, index_word)
	build_json_file(word_truths, DATA_SET + '_ground_truth.json')


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
'''
if __name__ == "__main__":
	embeddings = EmbeddingHolder().get_embeddings_mat()
	data = TFDataHolder(DATA_SET, TRAIN_BATCH_SIZE)
	
	index_word = get_index_word_dict()

	preds = get_preds(data, embeddings)
	preds = sub_in_word(preds, index_word)
	build_json_file(preds, DATA_SET + '_preds_json.json')

	get_ground_truth(data, index_word)
'''



























