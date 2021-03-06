import numpy as np
import tensorflow as tf
from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import SelfAttentionLayer
from sklearn.metrics import roc_auc_score, log_loss

# load data
x_test_tokenized = np.load(ROOT_DIR + '/prep_data/x_test_tokenized.npy')
y_test = np.load(ROOT_DIR + '/prep_data/y_test.npy')

# load model
model = tf.keras.models.load_model(
    ROOT_DIR + '/pickled/bd_self_att_gl300.h5',
    custom_objects={'SelfAttentionLayer': SelfAttentionLayer}
)

# make predictions
pred = model.predict(x_test_tokenized, batch_size=256)

# get evaluation metrics
print('Ground truth shape: {}, Predicitons shape: {}'.format(y_test.shape, pred.shape))

eval = roc_auc_score(y_test, pred)
print("Mean ROC AUC for all classes: {}".format(eval))