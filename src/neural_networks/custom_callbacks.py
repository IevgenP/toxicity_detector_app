import tensorflow as tf
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(tf.keras.callbacks.Callback):
    """Custom class for ROC AUC evaluation at the end of an epoch during training"""
    
    def __init__(self, validation_data=(), interval=1, **kwargs):
        """Initialize RocAucEvaluation class
        
        :param validation_data: validation data and labels, defaults to ()
        :type validation_data: tuple on numpy arrays
        :param interval: interval between epoch for evaluation, defaults to 1
        :type interval: int
        """
        
        self.interval = interval
        self.X_val, self.y_val = validation_data
        super(RocAucEvaluation, self).__init__(**kwargs)
    
    def on_epoch_end(self, epoch, logs={}):
        """Calculation of ROC AUC score at the end of an epoch
        
        :param epoch: epoch number
        :type epoch: int
        :param logs: data to be saved in logs, defaults to {}
        :type logs: dict
        """

        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred, average='macro')
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))