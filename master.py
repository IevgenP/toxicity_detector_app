from definitions_toxicity import ROOT_DIR
import pandas as pd 
from src.preprocessing import custom_transformers as ct  
from sklearn.pipeline import Pipeline
import nltk
import pickle
from src.preprocessing import text_to_seq_utils as tts
import numpy as np
from sklearn.model_selection import train_test_split

# load dataset
print('loading dataset...')
j_df = pd.read_csv(ROOT_DIR + '/raw_data/jigsaw_dataset.csv')
j_df['prep_text'] = j_df['comment_text'].copy()

# split on train and validation sets
print('splitting on train and validation sets...')
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
x_tr, x_val, y_tr, y_val = train_test_split(j_df[['comment_text', 'prep_text']],
                                              j_df[labels], 
                                              test_size=0.15,
                                              random_state=167)

eng_stop_words = nltk.corpus.stopwords.words('english') 

# build pipeline from prepared blocks
pipe = Pipeline(
    steps = [
        ('ip_remover', ct.IpRemover(['prep_text'])),
        ('http_remover', ct.HTTPremover(['prep_text'])),
        ('new_line_rem', ct.NewLineRemover(['prep_text'])),
        ('user_name_rem', ct.UserNameRemover(['prep_text'])),
        ('lower', ct.IntoLowerCase(['prep_text'])),
        ('short_to_long', ct.ShortToLong(['prep_text'])),
        ('symbols_remover', ct.SymbolsRemover(['prep_text'])),
        ('word_tokenizer', ct.WordsTokenizerNLTK(['prep_text'])),
        ('punctuation_rem', ct.PunctuationRemover(['prep_text'])),
        ('stop_words_rem', ct.StopWordsRemover(['prep_text'], eng_stop_words)),
        ('pos', ct.PosTaggerNLTK(['prep_text'])),
        ('lemmatizer', ct.WordLemmatizerNLTK(['prep_text'])),
    ]
)

# use pipeline for preprocessing
print('fitting pipeline and transforming datasets...')
x_tr_prep = pipe.fit_transform(x_tr)
x_val_prep = pipe.transform(x_val)

# save pipeline
print('saving pipeline...')
with open(ROOT_DIR + '/pickled/prep_pipe.pickle', 'wb') as file:
    pickle.dump(pipe, file, protocol=pickle.HIGHEST_PROTOCOL)

# save preprocessed train and validation sets
x_tr_prep.to_csv(ROOT_DIR + '/temp_data/x_tr_prep.csv', index=False)
x_val_prep.to_csv(ROOT_DIR + '/temp_data/x_val_prep.csv', index = False)

# tokenize train and validation sets
print('fitting and saving tokenizer...')
tts.fit_tokenizer(x_tr_prep, 
                  'prep_text', 
                  vocab_size=20000,
                  save=True, 
                  path_to_tokenizer=ROOT_DIR + '/pickled/tokenizer.pickle')

print('tokenizing train set...')
x_tr_tokenized = tts.tokenize_text(x_tr_prep,
                                   'prep_text',
                                   path_to_tokenizer=ROOT_DIR + '/pickled/tokenizer.pickle',
                                   max_len=100, 
                                   padding_mode='pre', 
                                   truncating_mode='pre')

# save training set
print('saving training set and labels...')
np.save(ROOT_DIR + '/prep_data/x_tr_tokenized.npy', x_tr_tokenized)
np.save(ROOT_DIR + '/prep_data/y_tr_tokenized.npy', y_tr.values)


# tokenizing validation set
print('saving validation set and labels...')
x_val_tokenized = tts.tokenize_text(x_val_prep,
                                    'prep_text',
                                    path_to_tokenizer=ROOT_DIR + '/pickled/tokenizer.pickle',
                                    max_len=100, 
                                    padding_mode='pre', 
                                    truncating_mode='pre')
np.save(ROOT_DIR + '/prep_data/x_val_tokenized.npy', x_val_tokenized)
np.save(ROOT_DIR + '/prep_data/y_val_tokenized.npy', y_val.values)

print(x_tr_tokenized.shape, y_tr.shape, x_val_tokenized.shape, y_val.shape)