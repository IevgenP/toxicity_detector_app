import numpy as np
from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import BdRNN_Attention

# load data
x_tr_tokenized = np.load(ROOT_DIR + '/prep_data/x_tr_tokenized.npy')
y_tr = np.load(ROOT_DIR + '/prep_data/y_tr.npy')
x_val_tokenized = np.load(ROOT_DIR + '/prep_data/x_val_tokenized.npy')
y_val = np.load(ROOT_DIR + '/prep_data/y_val.npy')

print(x_tr_tokenized.shape, y_tr.shape, x_val_tokenized.shape, y_val.shape)

# load nerual network model
model = BdRNN_Attention()

if __name__ == "__main__":
    model.fit(x=x_tr_tokenized, 
            y=y_tr, 
            batch_size=1024,
            epochs=10, 
            validation_data=(x_val_tokenized, y_val),
            use_multiprocessing=False)