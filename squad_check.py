import time
import tensorflow as tf
import numpy as np

from model import Model
from progbar import Progbar

from embeddings_handler import EmbeddingHolder
from data_handler import DataHolder
from embeddings_handler import EmbeddingHolder
from tf_lstm_attention_cell import LSTMAttnCell
import get_predictions

from simple_configs import LOG_FILE_DIR, SAVE_MODEL_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE, EMBEDDING_DIM, QUESTION_MAX_LENGTH, PASSAGE_MAX_LENGTH, OUTPUT_MAX_LENGTH, VOCAB_SIZE, LEARNING_RATE, HIDDEN_DIM, MAX_NUM_PASSAGES

PAD_ID = 0
STR_ID = 1
END_ID = 2
SOS_ID = 3
UNK_ID = 4

class SquadCheck(Model):
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, QUESTION_MAX_LENGTH), name="questions")
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, PASSAGE_MAX_LENGTH), name="passages")
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="answers")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, questions_batch, passages_batch, dropout=0.5, answers_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.questions_placeholder : questions_batch,
            self.passages_placeholder : passages_batch,
            self.dropout_placeholder : dropout
        }
        if answers_batch is not None: feed_dict[self.answers_placeholder] = answers_batch
        return feed_dict

    def add_embedding(self, placeholder):  
        embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, placeholder)
        return embeddings

    def seq_length(self, sequence):
        # used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        used = tf.sign(sequence)
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def encode_w_attn(self, inputs, mask, prev_states, scope="", reuse=False):
        
        with tf.variable_scope(scope, reuse):
            attn_cell = LSTMAttnCell(HIDDEN_DIM, prev_states)
            o, final_state = tf.nn.dynamic_rnn(attn_cell, inputs, dtype=tf.float32, sequence_length=mask)
        return (o, final_state)

    def add_prediction_op(self): 
        questions = self.add_embedding(self.questions_placeholder)
        passages = self.add_embedding(self.passages_placeholder)

        # Question encoder
        with tf.variable_scope("question"): 
            q_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            q_outputs, _ = tf.nn.dynamic_rnn(q_cell, questions, dtype=tf.float32, sequence_length=self.seq_length(self.questions_placeholder))

        # Passage encoder
        p_outputs, _ = self.encode_w_attn(passages, self.seq_length(self.passages_placeholder), q_outputs, scope = "passage_attn")
 
        # Attention state encoder
        with tf.variable_scope("attention"): 
            a_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
            a_outputs, _ = tf.nn.dynamic_rnn(a_cell, p_outputs, dtype=tf.float32, sequence_length=self.seq_length(self.passages_placeholder))

        q_last = tf.slice(q_outputs, [0, QUESTION_MAX_LENGTH - 1, 0], [-1, 1, -1])
        p_last = tf.slice(p_outputs, [0, PASSAGE_MAX_LENGTH - 1, 0], [-1, 1, -1])
        a_last = tf.slice(a_outputs, [0, PASSAGE_MAX_LENGTH - 1, 0], [-1, 1, -1])
        encoded_info = tf.concat(2, [q_last, p_last, a_last]) # SHAPE: [BATCH, 1, 3*HIDDEN_DIM]
       
        preds = list()
        input_dim = 3*HIDDEN_DIM
        encoded_info = tf.reshape(encoded_info, [-1, input_dim])

        # start token
        with tf.variable_scope("start_token"):
	        W = tf.get_variable('W', shape=(input_dim, input_dim), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
	        b1 = tf.get_variable('b1', shape=(input_dim, ), dtype=tf.float32)

	        h = tf.nn.relu(tf.matmul(encoded_info, W) + b1) # SHAPE: [BATCH, MAX_NUM_PASSAGE]

	        U = tf.get_variable('U', shape=(input_dim, PASSAGE_MAX_LENGTH), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
	        b2 = tf.get_variable('b2', shape=(PASSAGE_MAX_LENGTH, ), dtype=tf.float32)

	        start_token_logit = tf.matmul(h, U) + b2

	    # end token
        with tf.variable_scope("end_token"):
	        W = tf.get_variable('W', shape=(input_dim, input_dim), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
	        b1 = tf.get_variable('b1', shape=(input_dim, ), dtype=tf.float32)

	        h = tf.nn.relu(tf.matmul(encoded_info, W) + b1) # SHAPE: [BATCH, MAX_NUM_PASSAGE]

	        U = tf.get_variable('U', shape=(input_dim, PASSAGE_MAX_LENGTH), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
	        b2 = tf.get_variable('b2', shape=(PASSAGE_MAX_LENGTH, ), dtype=tf.float32)

	        end_token_logit = tf.matmul(h, U) + b2

        return (start_token_logit, end_token_logit)


    def add_loss_op(self, preds):
    	start_token_logit = preds[0]
    	end_token_logit = preds[1]

        start_loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(start_token_logit, self.start_answers_placeholder)
        end_loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(end_token_logit, self.end_answers_placeholder)

        loss_mat = start_loss_mat + end_loss_mat
        return tf.reduce_mean(loss_mat)

    def add_training_op(self, loss):        
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        grad_var_pairs = optimizer.compute_gradients(loss)
        grads = [g[0] for g in grad_var_pairs]
        grad_norm = tf.global_norm(grads)
        tf.summary.scalar('Global Gradient Norm', grad_norm)

        return optimizer.apply_gradients(grad_var_pairs)

    def train_on_batch(self, sess, merged, questions_batch, passages_batch, dropout, answers_batch):
        """Perform one step of gradient descent on the provided batch of data."""
        feed = self.create_feed_dict(questions_batch, passages_batch, dropout, answers_batch=answers_batch)
        #_, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        summary, _, loss = sess.run([merged, self.train_op, self.loss], feed_dict=feed)
        self.train_writer.add_summary(summary, self.step)
        self.step += 1
        return loss

    def run_epoch(self, sess, merged, data):
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        losses = list()
        i = 0
        batch = data.get_batch()
        while batch is not None:
            q_batch = batch['question']
            p_batch = batch['passage']
            a_batch = batch['selected_passage']
            dropout = batch['dropout']

            loss = self.train_on_batch(sess, merged, q_batch, p_batch, dropout, a_batch)
            tf.summary.scalar('Classifier Loss per Batch', loss)
            losses.append(loss)

            prog.update(i + 1, [("train loss", loss)])

            batch = data.get_batch()
            i += 1

        return losses

    def predict(self, sess, saver, data):
        self.testing = False
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        preds = list()
        i = 0
        
        batch = data.get_batch(predicting=True)
        print 'batch', batch
        while batch is not None:
            q_batch = batch['question']
            p_batch = batch['passage']
            dropout = batch['dropout']

            prediction = self.predict_on_batch(sess, q_batch, p_batch, dropout)
            preds.append(prediction)

            prog.update(i + 1, [("Predictions going...", 1)])

            batch = data.get_batch(predicting=True)
            i += 1

        return preds

    def predict_on_batch(self, sess, questions_batch, passages_batch, dropout):
        feed = self.create_feed_dict(questions_batch, passages_batch, dropout)
        predictions = sess.run(tf.nn.softmax(self.pred), feed_dict=feed)
        print 'preds', predictions
        predictions = np.argmax(predictions, axis=1)
        print 'argmax:', predictions
        return predictions

    def __init__(self, embeddings, predicting=False):
        self.predicting = predicting
        self.pretrained_embeddings = tf.Variable(embeddings)
        self.log = open(LOG_FILE_DIR, "a")
        self.build()

if __name__ == "__main__":
    print 'Starting Classifier, and now printing to log.txt'
    data = DataHolder('train')
    embeddings = EmbeddingHolder().get_embeddings_mat()
    with tf.Graph().as_default():
        start = time.time()
        model = PassClassifier(embeddings)
        model.log.write("\nBuild Classifier graph took " + str(time.time() - start) + " seconds")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        model.log.write('\ninitialzed classifier variables')
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=config) as session:
            merged = tf.summary.merge_all()
            session.run(init)
            model.log.write('\nran init, fitting classifier.....')
            losses = model.fit(session, saver, merged, data)

            model.log.write("starting predictions now.....")
            preds = model.predict(session, saver, data)
            print preds


    model.log.close()
















