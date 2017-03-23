import time
import tensorflow as tf
import numpy as np

import l2_double_2attn

from simple_configs import SAVE_MODEL_DIR
from prediction_handler import PredictionHandler
from data_handler import DataHolder
from embeddings_handler import EmbeddingHolder


def keep_training(model_path = SAVE_MODEL_DIR):
	pred_handler = PredictionHandler() 
	data = DataHolder('train')

	with tf.Graph().as_default():
		start = time.time()

		model = l2_double_2attn.TFModel(EmbeddingHolder().get_embeddings_mat())

		print "\nRebuild graph took " + str(time.time() - start) + " seconds"

		saver = tf.train.Saver()

		with tf.Session() as session:
			session = session
			saver.restore(session, model_path)
			print 'Restored model. Predicting....'
            merged = tf.summary.merge_all()
            losses = model.fit(session, saver, merged, data)

	model.train_writer.close()      
	model.test_writer.close()
	model.log.close()
	print 'Done continueing'

if __name__ == "__main__":
	keep_training()







