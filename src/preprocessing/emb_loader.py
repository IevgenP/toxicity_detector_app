import tensorflow as tf 
import pandas as pd
import numpy as np


class PreTrainedEmbLoader():

    def __init__(self, VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM):
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_LEN = MAX_LEN
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def load_pre_trained(self, filepath):
        vocab_keys = []
        self.embedding_dict = {}
        with open(filepath, 'r', encoding='UTF-8') as file:
            for line in file.readlines():
                row = line.strip().split(' ')
                vocab_word = row[0]
                vocab_keys.append(vocab_word)
                embed_vector = [float(i) for i in row[1:]]
                self.embedding_dict[vocab_word] = embed_vector
        return self

    def prepare_embedding_matrix(self, word_index):
        num_words = min(self.VOCAB_SIZE, len(word_index))+1
        embedding_matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > self.VOCAB_SIZE:
                continue
            embedding_vector = self.embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.uniform(low=-0.2, high=0.2, size=self.EMBEDDING_DIM)

        return embedding_matrix