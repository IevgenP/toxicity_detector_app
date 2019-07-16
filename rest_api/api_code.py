import flask
import flask_restful
import nltk
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from definitions_toxicity import ROOT_DIR
from src.neural_networks.nn import SelfAttentionLayer
from src.preprocessing import text_to_seq_utils as tts

app = flask.Flask(__name__)
api = flask_restful.Api(app)

# load preprocessing pipeline
with open(ROOT_DIR + '/pickled/prep_pipe.pickle', 'rb') as file:
    pipe = pickle.load(file)

# load stop words list
# eng_stop_words = nltk.corpus.stopwords.words('english') 

class ToxicityClassifier(flask_restful.Resource):

    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.prep_pipe = pipe
        self.model = tf.keras.models.load_model(
            ROOT_DIR + '/pickled/bd_self_att_gl300.h5', 
            custom_objects={'SelfAttentionLayer': SelfAttentionLayer}
        )

    def post(self):

        # load data
        dataload = flask.request.get_json()
        raw_data = pd.DataFrame.from_dict(dataload['data'], orient='columns')

        # make preprocessing
        raw_data['prep_text'] = raw_data['text'].copy()
        prep_data = self.prep_pipe.transform(raw_data)

        # tokenize text
        tokenized_text = tts.tokenize_text(prep_data,
                                           'prep_text',
                                            path_to_tokenizer='/pickled/tokenizer.pickle',
                                            max_len=200, 
                                            padding_mode='pre', 
                                            truncating_mode='pre')

        print(prep_data)

        # make predictions
        predictions = self.model.predict(tokenized_text)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2006, debug=True)