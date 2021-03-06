import time
from definitions_toxicity import ROOT_DIR
import pandas as pd 
from src.preprocessing import custom_transformers as ct  
from sklearn.pipeline import Pipeline
import nltk
import pickle
from src.preprocessing.text_utils import tokenize_by_sentences, fit_tokenizer, tokenize_text_with_sentences
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split


def load_text_data(data_filepath, text_column, copy_text_column=False, copy_col_name=None):
    """Data loader
    
    :param data_filepath: file path to data
    :type data_filepath: string
    :param text_column: name of column with text
    :type text_column: string
    :param copy_text_column: whether to copy column with text into new one, defaults to False
    :type copy_text_column: bool, optional
    :param copy_col_name: name of new column for copying original one, defaults to None
    :type copy_col_name: boolean, optional
    :return: dataframe with text data
    :rtype: Pandas dataframe
    """
    df = pd.read_csv(ROOT_DIR + data_filepath)
    if copy_text_column:
        df[copy_col_name] = df[text_column].copy()
    return df

def split_dataset(df, columns, labels, split='train_test', test_size=0.3, random_state=111, stratify=False, multiclass=False):
    """Data split function that can use both sklearn train_test_split and skmultilearn iterative_train_test_split for cases with imbalanced multilabel data. 
    
    :param df: dataframe that requires splitting into train, test and possibly validation sets
    :type df: Pandas dataframe
    :param columns: names of columns that will be left in train/test/valiation sets
    :type columns: list
    :param labels: name of columns that represent labels
    :type labels: list
    :param split: selection of how to split dataset: train_test or train_val_test, defaults to 'train_test'
    :type split: str, optional
    :param test_size: fraction of whole dataset to be used as test set, defaults to 0.3
    :type test_size: float, optional
    :param random_state: random state for splitting, defaults to 111
    :type random_state: int, optional
    :param stratify: whether to stratify the data, defaults to False
    :type stratify: bool, optional
    :param multiclass: whether dataset has multiclass labels, defaults to False
    :type multiclass: bool, optional
    :return: train, test and optionally validation sets
    :rtype: Pandas dataframe
    """
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
    """Function for applying skleran pipeline to train and test/validation datasets
    
    :param x_tr: training data
    :type x_tr: Pandas dataframe
    :param x_test: test data
    :type x_test: Pandas dataframe
    :param steps_list: list of steps, where each element is a transformer acceptable by skleran Pipeline
    :type steps_list: list
    :param x_val: validation data, defaults to None
    :type x_val: Pandas dataframe, optional
    :param save_pipeline: whether to save fitted pipeline, defaults to False
    :type save_pipeline: bool, optional
    :param save_pipeline_filepath: file path for saving pipeline, defaults to None
    :type save_pipeline_filepath: string, optional
    :param save_preprocessed: whether to save preprocessed data, defaults to False
    :type save_preprocessed: bool, optional
    :param prep_x_tr_filepath: file path for saving preprocessed train data, defaults to None
    :type prep_x_tr_filepath: string, optional
    :param prep_x_val_filepath: file path for saving preprocessed validation data, defaults to None
    :type prep_x_val_filepath: string, optional
    :param prep_x_test_filepath: file path for saving preprocessed test data, defaults to None
    :type prep_x_test_filepath: string, optional
    """

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
        data_filepath='/raw_data/jigsaw_dataset_2.csv',
        text_column='comment_text', 
        copy_text_column=True, 
        copy_col_name='prep_text'
    )
    
    # rename columns according to first jigsaw table
    text_df = text_df.rename(columns={
            'target': 'toxic', 
            'identity_attack': 'identity_hate'
        }
    )

    # replace fraction of human raters who consider comment to be toxic in any way
    # to binary label of 0 or 1
    for col in ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']: #severe_toxicity excluded as it has only few positive values
        array = text_df[col].values
        array[array > 0.5] = 1
        array[array <= 0.5] = 0
        text_df[col] = array

    print("Replacement of fraction of human raters into binary values is completed.")

    tstep = time.time()

    # spilit on train, validation and test sets
    x_tr, x_val, x_test, y_tr, y_val, y_test = split_dataset(
        df = text_df, 
        columns=['comment_text', 'prep_text'],
        labels=['toxic', 'obscene', 'threat', 'insult', 'identity_hate'], 
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

    # fit word-level tokenizer
    MAX_NB_WORDS = 20000
    start = time.time()
    print('fitting and saving tokenizer...')
    fit_tokenizer(x_tr_prep, 
                  'prep_text', 
                  vocab_size=MAX_NB_WORDS,
                  save=True, 
                  path_to_tokenizer='/pickled/tokenizer.pickle')

    
    # transform text data into 3D vector (sample, sentences, tokens_in_sentence)
    x_tr_sent = tokenize_by_sentences(df=x_tr, column='prep_text')
    x_test_sent = tokenize_by_sentences(df=x_test, column='prep_text')
    x_val_sent = tokenize_by_sentences(df=x_val, column='prep_text')

    # tokenize words in 3D vector
    with open(ROOT_DIR + '/pickled/tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)

    MAX_SENT_LENGTH = 100
    MAX_SENTS = 15

    print('tokenizing train, test and validation sets...')
    for dset, path in [(x_tr_sent, '/prep_data/x_tr_tokenized.npy'),
                       (x_test_sent, '/prep_data/x_test_tokenized.npy'),
                       (x_val_sent, '/prep_data/x_val_tokenized.npy')]:
        tokenize_text_with_sentences(text_samples_list=dset, 
                                    loaded_tokenizer=tokenizer, 
                                    max_sentences=MAX_SENTS, 
                                    max_sentence_length=MAX_SENT_LENGTH, 
                                    max_num_words=MAX_NB_WORDS, 
                                    save=True, 
                                    save_path=path)

    tstep = time.time()
    print('Tokenizer is trained and applied to datasets. Time required: {}'.format(tstep-start))

    