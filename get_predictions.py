import json
import time
import tensorflow as tf

from embeddings_handler import EmbeddingHolder
from data_handler import DataHolder
from embeddings_handler import EmbeddingHolder

import l2_2attn_dbl

from simple_configs import TEXT_DATA_DIR, VOCAB_SIZE

DATA_SET = 'train'

PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

# ~~~ DOESN'T WORK ~~~
MODEL_PATH = './data/model.weights'
def get_preds(data, embeddings):
	with tf.Graph().as_default():
		start = time.time()
		model = l2_2attn_dbl.TFModel(embeddings, True)
		print "\nRebuild graph took " + str(time.time() - start) + " seconds"
		saver = tf.train.Saver()
		with tf.Session() as session:
			saver.restore(session, MODEL_PATH)
			print 'STARTING TO MAKE PREDICTIONS'
			predictions = model.predict(session, saver, data)
	print 'predictions', predictions
	return predictions

def get_ground_truth(data, index_word):
	A_data = list()
	batch = data.get_selected_passage_batch()
	while batch is not None:
		A_batch = batch['answer']
		A_data.append(A_batch)
		batch = data.get_batch()

	word_truths = sub_in_word(A_data, index_word)
	build_json_file(word_truths, './data/' + DATA_SET + '_ground_truth.json')


def get_index_word_dict():
	index_word = dict()
	f = open(TEXT_DATA_DIR)
	for i, line in enumerate(f):
		if i > VOCAB_SIZE: break
		word = line.strip()
		index_word[i] = word
	f.close()
	return index_word

def sub_in_word(preds, index_word):
	word_preds = list()

	for batch in preds:
		for row in batch:
			ans = list()
			for i in row:
				ans.append(index_word[i])
				if i == END_ID: break #break on end tag
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
	data = DataHolder(DATA_SET)
	
	index_word = get_index_word_dict()

	preds = get_preds(data, embeddings)
	preds = sub_in_word(preds, index_word)
	build_json_file(preds, DATA_SET + '_preds.json')

	# get_ground_truth(data, index_word)
'''
if __name__ == "__main__":
	data = DataHolder('train')
	get_ground_truth(data, get_index_word_dict())

'''



# def do_shell(args):
#     # config = Config(args)
#     helper = ModelHelper.load(args.model_path)
#     embeddings = load_embeddings(args, helper)
#     config.embed_size = embeddings.shape[1]

#     with tf.Graph().as_default():
#         logger.info("Building model...",)
#         start = time.time()
#         model = RNNModel(helper, config, embeddings)
#         logger.info("took %.2f seconds", time.time() - start)

#         init = tf.global_variables_initializer()
#         saver = tf.train.Saver()

#         with tf.Session() as session:
#             session.run(init)
#             saver.restore(session, model.config.model_output)







#     def load(cls, path):
#         # Make sure the directory exists.
#         assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
#         # Save the tok2id map.
#         with open(os.path.join(path, "features.pkl")) as f:
#             tok2id, max_length = pickle.load(f)
#         return cls(tok2id, max_length)



# def load_embeddings(args, helper):
#     embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, EMBED_SIZE), dtype=np.float32)
#     embeddings[0] = 0.
#     for word, vec in load_word_vector_mapping(args.vocab, args.vectors).items():
#         word = normalize(word)
#         if word in helper.tok2id:
#             embeddings[helper.tok2id[word]] = vec
#     logger.info("Initialized embeddings.")

#     return embeddings









