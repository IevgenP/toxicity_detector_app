import flask
import flask_restful
import nltk
import pickle
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# sess = tf.keras.backend.set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import SelfAttentionLayer, penalize_loss
from src.preprocessing.text_utils import tokenize_by_sentences, fit_tokenizer, tokenize_text_with_sentences

app = flask.Flask(__name__)
api = flask_restful.Api(app)

# load preprocessing pipeline
with open(ROOT_DIR + '/pickled/prep_pipe.pickle', 'rb') as file:
    pipe = pickle.load(file)

# loaded_model = tf.keras.models.load_model(
#     ROOT_DIR + '/pickled/bd_self_att_gl300.h5', # baseline_gl300 
#     custom_objects={
#         'SelfAttentionLayer': SelfAttentionLayer
#     }
# )
# loaded_model.summary()

class ToxicityClassifier(flask_restful.Resource):
    """Class for giving prediction on specified endpoint"""

    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.prep_pipe = pipe
        # make random initializations 
        # (required only to load the model, during validation these random values 
        # are going to be substituted with values calculated by the model)
        self.batch_size = 100
        self.att_weights = tf.keras.backend.variable(np.array([[1, 2], [1, 2]]))
        loaded_model = tf.keras.models.load_model(
            ROOT_DIR + '/pickled/bd_self_att_gl300.h5', # baseline_gl300 
            custom_objects={
                'SelfAttentionLayer': SelfAttentionLayer,
                'penalization': penalize_loss(self.att_weights, self.batch_size)
            }
        )
        loaded_model.summary
        self.model = tf.keras.models.Model(inputs=loaded_model.input,
                                           outputs=[loaded_model.output, loaded_model.get_layer('self_attention_layer_1').output[1]])


    def post(self):
        """Returns predictions as a response to post resuest
        
        :return: predictions made by loaded neural network
        :rtype: string
        """

        # load data
        dataload = flask.request.get_json()
        raw_data = pd.DataFrame.from_dict(dataload['data'], orient='columns')

        # make preprocessing
        raw_data['prep_text'] = raw_data['text'].copy()
        prep_data = self.prep_pipe.transform(raw_data)
        print(prep_data)
        print('-'*10)

        # tokenize text
        # transform text data into 3D vector (sample, sentences, tokens_in_sentence)
        prep_data_sent = tokenize_by_sentences(df=prep_data, column='prep_text')
        print(prep_data_sent)

        # tokenize words in 3D vector
        with open(ROOT_DIR + '/pickled/tokenizer.pickle', 'rb') as file:
            tokenizer = pickle.load(file)

        MAX_SENT_LENGTH = 100
        MAX_SENTS = 15
        MAX_NB_WORDS = 20000

        print('tokenizing new data...')
        tokenized_text = tokenize_text_with_sentences(
            text_3d_vector=prep_data_sent, 
            loaded_tokenizer=tokenizer, 
            max_sentences=MAX_SENTS, 
            max_sentence_length=MAX_SENT_LENGTH, 
            max_num_words=MAX_NB_WORDS
        )

        # make predictions
        predictions, weights = self.model.predict(tokenized_text)

        # for i in range(prep_data.shape[0]):
        #     print('Sample number: {}'.format(i))
        #     col = prep_data['prep_text'].values[i]
        #     data = weights[i,-len(col):,:]
        #     data = np.transpose(data)
        #     with pd.option_context('display.max_columns', 50):
        #         print(pd.DataFrame(columns=col, data=data))
        #     print('--------------')

        # jsonify results
        pred_dict = {}
        for i in range(raw_data.shape[0]):
            pred_row = predictions[i]
            pred_dict[i+1] = str({
                'toxic': np.round(pred_row[0], 4),
                'obscene': np.round(pred_row[1], 4),
                'threat': np.round(pred_row[2], 4),
                'insult': np.round(pred_row[3], 4),
                'identity_hate': np.round(pred_row[4], 4)
            })
        return flask.jsonify(pred_dict)

api.add_resource(ToxicityClassifier, '/toxicity_clf')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=2006, debug=True)