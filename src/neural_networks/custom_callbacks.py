import tensorflow as tf
from sklearn.metrics import roc_auc_score

class RocAucEvaluation(tf.keras.callbacks.Callback):
    
    def __init__(self, validation_data=(), interval=1, **kwargs):
        self.interval = interval
        self.X_val, self.y_val = validation_data
        super(RocAucEvaluation, self).__init__(**kwargs)
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred, average='macro')
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))