import nltk
import collections
import warnings
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import seaborn as sns
from definitions import ROOT_DIR
from sklearn.feature_extraction.text import TfidfVectorizer


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


# def plot_counter(count_df, top_n, n_gram, text):
#     """Function for getting the most common words for both target values"""
#     count_df = count_df.rename(columns={'index': 'word', 0: 'TF-IDF count'})
#     count_df = count_df.sort_values(by='TF-IDF count', axis=0, ascending=False)
#     sns.barplot(x="TF-IDF count", y="word", data=count_df.head(top_n), palette="Blues_d").set_title('{} IF-IDF count{}.'.format(n_gram.capitalize(), text))


def get_tf_idf_scores(df, column, n_gram):
    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(n_gram,n_gram),
                                 max_df=100000,
                                 min_df=1,
                                 norm=None)
    tf_idf_matrix = vectorizer.fit_transform(df[column].values)
    feature_names = vectorizer.get_feature_names()
    tf_idf_df = pd.DataFrame({"tf_idf_score": tf_idf_matrix.max(axis=0).toarray()[0],
                              "words": feature_names})
    tf_idf_df = tf_idf_df.sort_values('tf_idf_score', axis=0, ascending=False)
    return tf_idf_df
