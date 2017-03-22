import time
import tensorflow as tf
import numpy as np

import l2_attn
import l2_2attn
import l2_2attn_dbl
import ce_attn
import ce_2attn

from simple_configs import SAVE_MODEL_DIR
from prediction_handler import PredictionHandler
from data_handler import DataHolder
from embeddings_handler import EmbeddingHolder
from passage_classifier_eval import classifier_eval
from passage_classifier import PassClassifier


def get_preds(data_set, model_type = None, model_path = SAVE_MODEL_DIR, output_file_name='./data/last_preds.json'):
	pred_handler = PredictionHandler() 
	data = DataHolder(data_set)

	with tf.Graph().as_default():
		start = time.time()

		if model_type == 'l2_attn': model = l2_attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
		elif model_type == 'l2_2attn': model = l2_2attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
		elif model_type == 'l2_2attn_dbl': model = l2_2attn_dbl.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
		elif model_type == 'ce_attn': model = ce_attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
		elif model_type == 'ce_2attn': model = ce_2attn.TFModel(EmbeddingHolder().get_embeddings_mat(), True)
		else: return

		print "\nRebuild graph took " + str(time.time() - start) + " seconds"

		saver = tf.train.Saver()

		with tf.Session() as session:
			session = session
			saver.restore(session, model_path)
			print 'Restored model. Predicting....'
			preds = model.predict(session, data)
			pred_handler.write_preds(preds, output_file_name)
	print 'Done Predicting'


def get_classifier_results(data_set, model_path = SAVE_MODEL_DIR):
	data = DataHolder(data_set)

	with tf.Graph().as_default():
		start = time.time()

		model = PassClassifier(EmbeddingHolder().get_embeddings_mat(), True)

		print "\nRebuild graph took " + str(time.time() - start) + " seconds"

		saver = tf.train.Saver()

		with tf.Session() as session:
			session = session
			saver.restore(session, model_path)
			print 'Restored model. Predicting....'
			preds = model.predict(session, data)
			list_preds = list()
			for batch in preds:
				for row in batch:
					list_preds.append(row)
			preds = np.asarray(list_preds)
			y = data.get_full_selected()
			classifier_eval(preds, y)
		print 'Done Predicting'

if __name__ == "__main__":
	# get_preds('val', model_type='l2_2attn', output_file_name='./data/doublecheck_val_preds.json')
	get_classifier_results('val')
	print 'done'






