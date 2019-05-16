import nltk
import collections
import warnings
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import seaborn as sns
from definitions import ROOT_DIR


def get_stats(df, colum_content, column):
    """
    Gives simple statistics for selected column of a dataframe.
    
    :param df: a DataFrame
    :type df: Pandas DataFrame
    :param colum_content: content for which statistics is required
    :type colum_content: string
    :param column: column that requre description
    :type column: int/float
    :returns: distribution plot
    """
    print("Min number of {}: {}".format(colum_content, df[column].min()))
    print("Max number of {}: {}".format(colum_content, df[column].max()))
    print("Mean number of {}: {}".format(colum_content, df[column].mean()))
    print("Median number of {}: {}".format(colum_content, df[column].median()))

    return sns.distplot(df[column])


def plot_counter(counter_output, top_n, n_gram, text):
    """Function for getting the most common words for both target values"""
    count_df = pd.DataFrame.from_dict(counter_output, orient='index').reset_index()
    count_df = count_df.rename(columns={'index': 'word', 0: 'TF-IDF count'})
    count_df = count_df.sort_values(by='TF-IDF count', axis=0, ascending=False)
    sns.barplot(x="TF-IDF count", y="word", data=count_df.head(top_n), palette="Blues_d").set_title('{} IF-IDF count{}.'.format(n_gram.capitalize(), text))


def count_words(df, column, n_gram, top_n, text=None):
    """Function for counting most common words words"""

    count = collections.Counter()
    tf_idf_dict = {}
    for row in df[column]:
        words = nltk.word_tokenize(row)
        if n_gram == 'unigrams':
            
            # compute raw term frequency of each words in dataframe column
            count.update(words)
            
            # compute soft inverse document frequency
            for word in count:
                idf = np.log(float(df.shape[0])/(len([1 for row in df[column] if word in row])+0.001) + 0.001)
                tf_idf_dict[word] = count[word] * idf
        
        elif n_gram == 'bigrams':
            
            # compute raw term frequency of each words in dataframe column
            count.update(nltk.bigrams(words))
            
            # compute soft inverse document frequency
            for ngram in count:
                comb = " ".join(list(ngram))
                idf = np.log(float(df.shape[0])/(len([1 for row in df[column] if comb in row])+0.001) + 0.001)
                tf_idf_dict[ngram] = count[ngram] * idf
        
        elif n_gram == 'trigrams':

            # compute raw term frequency of each words in dataframe column
            count.update(nltk.trigrams(words))

            # compute soft inverse document frequency
            for ngram in count:
                comb = " ".join(list(ngram))
                idf = np.log(float(df.shape[0])/(len([1 for row in df[column] if comb in row])+0.001) + 0.001)
                tf_idf_dict[ngram] = count[ngram] * idf
        
        else:
            warnings.warn("Not viable value for n_gram is provided. Please select one of the following: 'unigrams', 'bigrams', 'trigrams'.")
    
    
    
    plot_counter(tf_idf_dict, top_n, n_gram, text)