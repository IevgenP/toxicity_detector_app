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
    custom_objects={
        'SelfAttentionLayer': SelfAttentionLayer
    }
)

# make predictions
pred = model.predict(x_test_tokenized)

# get evaluation metrics
print('Ground truth shape: {}, Predicitons shape: {}'.format(y_test.shape, pred.shape))

# define custom log loss function
def calculate_cross_entropy(y_true, y_pred):
    N = y_pred.shape[0]
    ce = -np.sum(y_true*np.log(y_pred))/N
    return ce

#eval = calculate_cross_entropy(y_test, pred[:])

eval = roc_auc_score(y_test, pred)

print(eval)