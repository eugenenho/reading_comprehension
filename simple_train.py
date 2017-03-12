from tqdm import tqdm 

from simple_model import build_model
from simple_configs import TRAIN_BATCH_SIZE, SAVE_PREDICTIONS_FREQUENCY, SAVE_MODEL_DIR, NUM_EPOCS
from data_handler import DataHolder

def save_model(nn_model):
    nn_model.save_weights(SAVE_MODEL_DIR, overwrite=True)

# This is the main function for the model right now.
# Builds model from the other file, gets data, and runs a fit
# Then saves model weights
def train_model(nn_model):
    data = DataHolder('train')
    
    print 'Data retrieved'

    for epoc in range(NUM_EPOCS):
        for i in tqdm(range(1, data.get_num_iterations())):
            X_batch, Y_batch = data.get_batch_data()
            if X_batch is None or Y_batch is None: break
            curr_loss = nn_model.train_on_batch(X_batch, Y_batch)
            print 'LOSS:', curr_loss
            if i % SAVE_PREDICTIONS_FREQUENCY == 0: 
                save_model(nn_model)
        print 'finished EPOC: ', epoc
    save_model(nn_model)

def test_model(nn_model):
    data = DataHolder('dev')
    X_dev, Y_dev = data.get_full_data()
    loss_and_metrics = model.evaluate(X_dev, Y_dev, batch_size=TRAIN_BATCH_SIZE)
    print 'Evals:', loss_and_metrics

if __name__ == "__main__":
    nn_model = build_model()
    train_model(nn_model)
    test_model(nn_model)























