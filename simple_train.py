from tqdm import tqdm 
import time

from simple_model import build_model
from simple_configs import FULL_PASS_ITERS, TRAIN_BATCH_SIZE, SAVE_PREDICTIONS_FREQUENCY, SAVE_MODEL_DIR
from data_handler import DataHolder

def save_model(nn_model):
    nn_model.save_weights(SAVE_MODEL_DIR, overwrite=True)

# This is the main function for the model right now.
# Builds model from the other file, gets data, and runs a fit
# Then saves model weights
def train_model():
    nn_model = build_model()

    # get data
    data = DataHolder('train')

    start_time = time.time()
    sents_batch_iteration = 1
    for iter_num in range(1, FULL_PASS_ITERS + 1):
        print '\n\nFull-data-pass iteration num: ' + str(iter_num)

        # FIT
        for i in tqdm(range(1, data.get_num_iterations())):

            X_batch, Y_batch = data.get_batch_data()

            if X_batch is None or Y_batch is None: break

            nn_model.fit(X_batch, Y_batch, batch_size=TRAIN_BATCH_SIZE, nb_epoch=1, verbose=1)

            if sents_batch_iteration % SAVE_PREDICTIONS_FREQUENCY == 0:
                save_model(nn_model)

            sents_batch_iteration += 1

        print 'Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / iter_num)

    save_model(nn_model)

if __name__ == "__main__":
    train_model()























