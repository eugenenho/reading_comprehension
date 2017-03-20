from progbar import Progbar
import tensorflow as tf
import numpy as np
from simple_configs import SAVE_MODEL_DIR, NUM_EPOCS, TRAIN_BATCH_SIZE

class Model(object):    
    def add_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, questions_batch, passages_batch, start_token_batch, dropout=0.5, answers_batch=None):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        raise NotImplementedError("Each Model must re-implement this method.")

    def predict(self, sess, saver, data):
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_epoch(self, sess, data):
        raise NotImplementedError("Each Model must re-implement this method.")

    def variable_summaries(var):       
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""        
        with tf.name_scope('summaries'):        
            mean = tf.reduce_mean(var)      
            tf.summary.scalar('mean', mean)     
            with tf.name_scope('stddev'):       
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))     
                tf.summary.scalar('stddev', stddev)     
                tf.summary.scalar('max', tf.reduce_max(var))        
                tf.summary.scalar('min', tf.reduce_min(var))        
                tf.summary.histogram('histogram', var)

    def train_on_batch(self, sess, merged, questions_batch, passages_batch, start_token_batch, dropout, answers_batch):
        """Perform one step of gradient descent on the provided batch of data."""
        feed = self.create_feed_dict(questions_batch, passages_batch, start_token_batch, dropout, answers_batch=answers_batch)
        #_, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        summary, _, loss, self.last_preds = sess.run([merged, self.train_op, self.loss, self.pred], feed_dict=feed)
        self.train_writer.add_summary(summary, self.step)
        self.step += 1
        return loss

    def fit(self, sess, saver, merged, data):
        losses = []
        for epoch in range(NUM_EPOCS):
            self.log.write("\nEpoch: " + str(epoch + 1) + " out of " + str(NUM_EPOCS))
            loss = self.run_epoch(sess, merged, data)
            losses.append(loss)
            saver.save(sess, SAVE_MODEL_DIR)
        return losses

    def build(self):
        self.step = 0
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.train_writer = tf.summary.FileWriter('tsboard/' + '/train')
        self.test_writer = tf.summary.FileWriter('tsboard/' + '/test')
