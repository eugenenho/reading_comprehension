from progbar import Progbar

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

    def train_on_batch(self, sess, questions_batch, passages_batch, start_token_batch, dropout, answers_batch):
        """Perform one step of gradient descent on the provided batch of data."""
        feed = self.create_feed_dict(questions_batch, passages_batch, start_token_batch, dropout, answers_batch=answers_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, data):
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        losses = list()
        i = 0
        batch = data.get_selected_passage_batch()
        while batch is not None:
            q_batch = batch['question']
            p_batch = batch['passage']
            a_batch = batch['answer']
            s_t_batch = batch['start_token']
            droput = batch['dropout']

            loss = self.train_on_batch(sess, q_batch, p_batch, s_t_batch, dropout, a_batch)
            losses.append(loss)

            prog.update(i + 1, [("train loss", loss)])

            batch = data.get_selected_passage_batch()
            if i % 1200 == 0 and i > 0:
                self.log.write('\nNow saving file...')
                saver.save(sess, SAVE_MODEL_DIR)
                self.log.write('\nSaved...')
            i += 1
        return losses

    def fit(self, sess, saver, data):
        losses = []
        for epoch in range(NUM_EPOCS):
            self.log.write("\nEpoch: " + str(epoch + 1) + " out of " + str(NUM_EPOCS))
            loss = self.run_epoch(sess, data)
            losses.append(loss)
            saver.save(sess, SAVE_MODEL_DIR)
        return losses

    def predict_on_batch(self, sess, questions_batch, passages_batch, start_token_batch, dropout):
        feed = self.create_feed_dict(questions_batch, passages_batch, start_token_batch, dropout)
        print 'feed', feed
        predictions = sess.run(tf.nn.softmax(self.pred), feed_dict=feed)
        predictions = np.argmax(predictions, axis=2)
        return predictions

    def predict(self, sess, saver, data):
        self.testing = False
        prog = Progbar(target=1 + int(data.data_size / TRAIN_BATCH_SIZE), file_given=self.log)
        
        preds = list()
        i = 0
        
        batch = data.get_selected_passage_batch(predicting=True)
        print 'batch', batch
        while batch is not None:
            q_batch = batch['question']
            p_batch = batch['passage']
            s_t_batch = batch['start_token']
            droput = batch['dropout']

            prediction = self.predict_on_batch(sess, q_batch, p_batch, s_t_batch, dropout)
            preds.append(prediction)

            prog.update(i + 1, [("Predictions going...", 1)])

            batch = data.get_selected_passage_batch(predicting=True)
            i += 1

        return preds

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
