import numpy as np
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# sess = tf.keras.backend.set_session(tf.Session(config=config))


from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import SelfAttentionLayer, penalize_loss
from sklearn.metrics import roc_auc_score, log_loss

# load data
x_test_tokenized = np.load(ROOT_DIR + '/prep_data/x_test_tokenized.npy')
y_test = np.load(ROOT_DIR + '/prep_data/y_test.npy')

# make random initializations 
# (required only to load the model, during validation these random values 
# are going to be substituted with values calculated by the model)
batch_size = 100
att_weights = tf.keras.backend.variable(np.array([[1, 2], [1, 2]]))

# load model
model = tf.keras.models.load_model(
    ROOT_DIR + '/pickled/bd_self_att_gl300.h5', # baseline_gl300
    custom_objects={
        'SelfAttentionLayer': SelfAttentionLayer,
        'penalization': penalize_loss(att_weights, batch_size)
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