import pickle
import nltk
import numpy as np
import tensorflow as tf
from definitions_toxicity import ROOT_DIR

    
def fit_tokenizer(df, column, vocab_size=20000, save=False, path_to_tokenizer=None):
    """Function that wraps Keras Tokenizer fit on text and its saving for further use
    
    :param df: dataframe with column that contains text to be fitted on
    :type df: Pandas dataframe
    :param column: name of the column with text to be fitted on
    :type column: string
    :param vocab_size: size of vocabulary, defaults to 20000
    :type vocab_size: int, optional
    :param save: whether to save fitted tokenizer, defaults to False
    :type save: bool, optional
    :param path_to_tokenizer: file path for saving trained tokenizer, defaults to None
    :type path_to_tokenizer: string, optional
    :return tokenizer: fitted tokenizer
    """
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df[column].values.tolist())

    if save:
        with open(ROOT_DIR + path_to_tokenizer, 'wb') as file:
            pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return tokenizer

def tokenize_by_sentences(df, column):
    """Function that presents column with text samples as list of samples,
    where each sample is a list of sentences
    
    :param df: dataframe with column that contains text samples
    :type df: Pandas dataframe
    :param column: name of column that contains text samples
    :type column: string
    :return: list of text samples, where each sample contains of list of sentences
    :rtype: list
    """

    tokenized_corpus = []
    text_array = df[column].values
    for idx in range(df.shape[0]):
        text = ' '.join(text_array[idx])
        sentences = nltk.tokenize.sent_tokenize(text)
        tokenized_corpus.append(sentences)
    return tokenized_corpus


def tokenize_text_with_sentences(text_samples_list, loaded_tokenizer, max_sentences, max_sentence_length, max_num_words, save=False, save_path=None):
    """Function that tokenize each sentence using prepared tokenizer
    
    :param text_samples_list: list of text samples, where each sample contains of list of sentences
    :type text_samples_list: list
    :param loaded_tokenizer: tokenizer fitted on training data
    :type loaded_tokenizer: Keras Tokenizer
    :param max_sentences: max number of sentences in each text sample
    :type max_sentences: int
    :param max_sentence_length: max number of tokens/words in each sentence
    :type max_sentence_length: int
    :param max_num_words: max number of words in vocabulary
    :type max_num_words: int
    :param save: whether to save array with tokenized sentences, defaults to False
    :type save: bool, optional
    :param save_path: path for saving array with tokenized sentences, defaults to None
    :type save_path: string, optional
    :return: 3D array of [samples, sentences in each sample, tokenized words in each sentence]
    :rtype: numpy array
    """
    tok_sent_words = np.zeros((len(text_samples_list), max_sentences, max_sentence_length), dtype='int32')
    for i, sentences in enumerate(text_samples_list):
        for j, sent in enumerate(sentences):
            if j < max_sentences:
                wordTokens = tf.keras.preprocessing.text.text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_sentence_length and loaded_tokenizer.word_index[word] < max_num_words:
                            tok_sent_words[i, j, k] = loaded_tokenizer.word_index[word]
                            k = k + 1
                    except: KeyError
    if save:
        np.save(ROOT_DIR + save_path, tok_sent_words)

    return tok_sent_words