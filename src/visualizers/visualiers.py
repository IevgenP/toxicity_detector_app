import nltk
import collections
import warnings
from sklearn.pipeline import Pipeline
import pandas as pd
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
    count_df = count_df.rename(columns={'index': 'word', 0: 'count'})
    count_df = count_df.sort_values(by='count', axis=0, ascending=False)
    sns.barplot(x="count", y="word", data=count_df.head(top_n), palette="Blues_d").set_title('{} frequency count{}.'.format(n_gram.capitalize(), text))


def count_words(df, column, n_gram, top_n, text=None):
    """Function for counting most common words words"""

    count = collections.Counter()
    if n_gram == 'unigrams':
        for row in df[column]:
            words = nltk.word_tokenize(row)
            count.update(words)
    elif n_gram == 'bigrams':
        for row in df[column]:
            words = nltk.word_tokenize(row)
            count.update(nltk.bigrams(words))
    elif n_gram == 'trigrams':
        for row in df[column]:
            words = nltk.word_tokenize(row)
            count.update(nltk.trigrams(words))
    else:
        warnings.warn("Not viable value for n_gram is provided. Please select one of the following: 'unigrams', 'bigrams', 'trigrams'.")

    plot_counter(count, top_n, n_gram, text)