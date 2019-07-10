import time
from definitions_toxicity import ROOT_DIR
import pandas as pd 
from src.preprocessing import custom_transformers as ct  
from sklearn.pipeline import Pipeline
import nltk
import pickle
from src.preprocessing import text_to_seq_utils as tts
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split


def load_text_data(data_filepath, text_column, copy_text_column=False, copy_col_name=None):
    # load dataset
    df = pd.read_csv(ROOT_DIR + data_filepath)
    if copy_text_column:
        df[copy_col_name] = df[text_column].copy()
    return df

def split_dataset(df, columns, labels, split='train_test', test_size=0.3, random_state=111, stratify=False, multiclass=False): 
    # split on train and validation sets
    assert split == 'train_test' or split == 'train_val_test', "Split attribute accepts only 'train_test' or 'train_val_test'"
    strat = None
    if stratify:
        strat = df[labels]
    if not multiclass:
        x_tr, x_test, y_tr, y_test = train_test_split(df[columns],
                                                        df[labels], 
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=strat)
    else:
        x_tr, y_tr, x_test, y_test = iterative_train_test_split(df[columns].values,
                                                                df[labels].values, 
                                                                test_size=test_size)
        x_tr = pd.DataFrame(x_tr, columns=columns)
        y_tr = pd.DataFrame(y_tr, columns=labels)
        x_test = pd.DataFrame(x_test, columns=columns)
        y_test = pd.DataFrame(y_test, columns=labels)

    if split == 'train_test':
        return x_tr, x_test, y_tr, y_test
    elif split == 'train_val_test':
        if stratify:
            strat = y_test
        if not multiclass:
            x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                            y_test, 
                                                            test_size=0.5,
                                                            random_state=random_state,
                                                            stratify=strat)
        else:
            x_val, y_val, x_test, y_test = iterative_train_test_split(x_test.values,
                                                                      y_test.values, 
                                                                      test_size=0.5)
            x_val = pd.DataFrame(x_val, columns=columns)
            y_val = pd.DataFrame(y_val, columns=labels)
            x_test = pd.DataFrame(x_test, columns=columns)
            y_test = pd.DataFrame(y_test, columns=labels)

        return x_tr, x_val, x_test, y_tr, y_val, y_test


def apply_pipeline(x_tr,
                   x_test,
                   steps_list,
                   x_val = None,
                   save_pipeline=False, 
                   save_pipeline_filepath=None, 
                   save_preprocessed=False, 
                   prep_x_tr_filepath=None,
                   prep_x_val_filepath=None,
                   prep_x_test_filepath=None):

    # build pipeline from prepared blocks
    pipe = Pipeline(steps=steps_list)

    # use pipeline for preprocessing
    x_tr_prep = pipe.fit_transform(x_tr)
    x_test_prep = pipe.transform(x_test)
    
    # save transformed sets
    if save_preprocessed:
        x_tr_prep.to_csv(ROOT_DIR + prep_x_tr_filepath, index=False)
        x_test_prep.to_csv(ROOT_DIR + prep_x_test_filepath, index = False)

    # save pipeline
    if save_pipeline:
        with open(ROOT_DIR + save_pipeline_filepath, 'wb') as file:
            pickle.dump(pipe, file, protocol=pickle.HIGHEST_PROTOCOL)

    # return result and optionally include validation set
    if x_val is not None:
        x_val_prep = pipe.transform(x_val)
        if save_preprocessed:
            x_val_prep.to_csv(ROOT_DIR + prep_x_val_filepath, index = False)
        return x_tr_prep, x_val_prep, x_test_prep
    else: 
        return x_tr_prep, x_test_prep


if __name__ == '__main__':

    # load the data
    text_df = load_text_data(
        data_filepath='/raw_data/jigsaw_dataset.csv',
        text_column='comment_text', 
        copy_text_column=True, 
        copy_col_name='prep_text'
    )
    tstep = time.time()

    # spilit on train, validation and test sets
    x_tr, x_val, x_test, y_tr, y_val, y_test = split_dataset(
        df = text_df, 
        columns=['comment_text', 'prep_text'],
        labels=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], 
        split='train_val_test', 
        test_size=0.2, 
        random_state=31, 
        stratify=False,
        multiclass=False
    )

    # check that split contains all classes
    print('y_tr', y_tr.sum(axis=0))
    print('y_val', y_val.sum(axis=0))
    print('y_test', y_test.sum(axis=0))

    # save labels
    np.save(ROOT_DIR + '/prep_data/y_tr.npy', y_tr.values)
    np.save(ROOT_DIR + '/prep_data/y_val', y_val.values)
    np.save(ROOT_DIR + '/prep_data/y_test', y_test.values)

    # get stop-words list and define pipeline steps
    eng_stop_words = nltk.corpus.stopwords.words('english') 
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

    # apply pipeline
    start = time.time()

    x_tr_prep, x_val_prep, x_test_prep = apply_pipeline(
        x_tr=x_tr,
        x_test=x_test,
        x_val = x_val,
        steps_list=steps, 
        save_pipeline=True, 
        save_pipeline_filepath='/pickled/prep_pipe.pickle', 
        save_preprocessed=True, 
        prep_x_tr_filepath='/temp_data/x_tr_prep.csv',
        prep_x_val_filepath='/temp_data/x_val_prep.csv',
        prep_x_test_filepath='/temp_data/x_test_prep.csv'
    )

    tstep = time.time()
    print('Pipeline is trained and applied to datasets. Time required: {}'.format(tstep-start))


    # tokenize train and validation sets
    tart = time.time()


    print('fitting and saving tokenizer...')
    tts.fit_tokenizer(x_tr_prep, 
                     'prep_text', 
                     vocab_size=20000,
                     save=True, 
                     path_to_tokenizer='/pickled/tokenizer.pickle')

    print('tokenizing train, test and validation sets...')

    for dset, path in [(x_tr_prep, '/prep_data/x_tr_tokenized.npy'),
                       (x_val_prep, '/prep_data/x_val_tokenized.npy'),
                       (x_test_prep, '/prep_data/x_test_tokenized.npy')]:
        
        tts.tokenize_text(dset,
                         'prep_text',
                         path_to_tokenizer='/pickled/tokenizer.pickle',
                         max_len=200, 
                         padding_mode='pre', 
                         truncating_mode='pre',
                         save=True,
                         save_path=path)

    tstep = time.time()
    print('Tokenizer is trained and applied to datasets. Time required: {}'.format(tstep-start))

    