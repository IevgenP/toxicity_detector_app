import pickle
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

    tok_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                 maxlen=max_len, 
                                                                 padding=padding_mode,
                                                                 truncating = truncating_mode)
    
    if save:
        np.save(ROOT_DIR + save_path, tok_sequence)

    return tok_sequence


    

