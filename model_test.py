import numpy as np
import tensorflow as tf
from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import SelfAttentionLayer, CustomReduceSumLayer
from sklearn.metrics import roc_auc_score

# load data
x_test_tokenized = np.load(ROOT_DIR + '/prep_data/x_test_tokenized.npy')
y_test = np.load(ROOT_DIR + '/prep_data/y_test.npy')

# load model
model = tf.keras.models.load_model(
    ROOT_DIR + '/pickled/bd_self_att_gl300.h5',
    custom_objects={
        'SelfAttentionLayer': SelfAttentionLayer,
        'CustomReduceSumLayer': CustomReduceSumLayer
    }
)

# make predictions
pred = model.predict(x_test_tokenized)

# get evaluation metrics
eval = roc_auc_score(y_test, pred, average='macro')
print(eval)