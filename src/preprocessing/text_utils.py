import pickle
import nltk
import numpy as np
import tensorflow as tf
from definitions_toxicity import ROOT_DIR

    
def fit_tokenizer(df, column, vocab_size=20000, save=False, path_to_tokenizer=None):
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df[column].values.tolist())

    if save:
        with open(ROOT_DIR + path_to_tokenizer, 'wb') as file:
            pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)


def tokenize_text(df, column, path_to_tokenizer, max_len, padding_mode, truncating_mode, save=False, save_path=None):

    with open(ROOT_DIR + path_to_tokenizer, 'rb') as file:
        tokenizer = pickle.load(file)

    sequences = tokenizer.texts_to_sequences(list(df[column].values))

    tok_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_len, 
        padding=padding_mode,
        truncating = truncating_mode
    )
    
    if save:
        np.save(ROOT_DIR + save_path, tok_sequence)
    
    return tok_sequence


def tokenize_by_sentences(df, column):
    tokenized_corpus = []
    text_array = df[column].values
    for idx in range(df.shape[0]):
        text = ' '.join(text_array[idx])
        sentences = nltk.tokenize.sent_tokenize(text)
        tokenized_corpus.append(sentences)
    return tokenized_corpus


def tokenize_text_with_sentences(text_3d_vector, loaded_tokenizer, max_sentences, max_sentence_length, max_num_words, save=False, save_path=None):
    tok_sent_words = np.zeros((len(text_3d_vector), max_sentences, max_sentence_length), dtype='int32')
    for i, sentences in enumerate(text_3d_vector):
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