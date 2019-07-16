import pickle
import tensorflow as tf
import numpy as np
from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import BdRNN_Attention
from src.neural_networks.custom_callbacks import RocAucEvaluation
from src.preprocessing.emb_loader import PreTrainedEmbLoader
import matplotlib.pyplot as plt


# load data
x_tr_tokenized = np.load(ROOT_DIR + '/prep_data/x_tr_tokenized.npy')
y_tr = np.load(ROOT_DIR + '/prep_data/y_tr.npy')
x_val_tokenized = np.load(ROOT_DIR + '/prep_data/x_val_tokenized.npy')
y_val = np.load(ROOT_DIR + '/prep_data/y_val.npy')

print("Loaded data dimensions:")
print("x_tr_tokenized: ", x_tr_tokenized.shape)
print("y_tr:", y_tr.shape)
print("x_val_tokenized:", x_val_tokenized.shape)
print("y_val:", y_val.shape)

# load word index from Tokenizer
with open(ROOT_DIR + '/pickled/tokenizer.pickle', 'rb') as file:
    tk = pickle.load(file)
w_index = tk.word_index

# set variables
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBEDDING_DIM = 300
ATT_UNITS = 30

# load matrix with pretrained embeddings
emb_loader = PreTrainedEmbLoader(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM)
emb_loader.load_pre_trained('/home/ievgen/projects/pretr_emb/wiki-news-300d-1M.vec')
emb_matrix = emb_loader.prepare_embedding_matrix(word_index=w_index)
emb_matrix = emb_matrix[1:]
print("Embedding matrix shape is: ", emb_matrix.shape)

# make callbacks
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath=ROOT_DIR + '/pickled/checkpoints/weigths.hdf5', 
    verbose=1,
    save_best_only=True
)

earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.00001,
    patience=2,
    mode='min',
    restore_best_weights=True
)

roc_auc_eval = RocAucEvaluation(
    validation_data=(x_val_tokenized, y_val),
     interval=1
)


if __name__ == "__main__":
   
   # load nerual network model
    model = BdRNN_Attention(dropout=0.2, 
                            num_words=VOCAB_SIZE, 
                            emb_dim=EMBEDDING_DIM, 
                            max_len=MAX_LEN,
                            att_units=ATT_UNITS,
                            emb_matrix=[emb_matrix],
                            trainable_flag=True)
   
    # pring model summary
    model.summary()

    # fit model
    history = model.fit(
        x=x_tr_tokenized, 
        y=y_tr, 
        batch_size=2048,
        epochs=50, 
        validation_data=(x_val_tokenized, y_val),
        use_multiprocessing=False,
        callbacks=[earlystopper]
    )
    
    # save model
    model.save(ROOT_DIR + '/pickled/bd_self_att_gl300.h5')

    # plot history for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    
    fig.savefig(ROOT_DIR + '/val_acc_loss_charts/val_loss.png')
    