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

def get_tf_idf_scores(df, column, n_gram):
    """Function for plotting tf-idf scores
    
    :param df: dataframe with prepared text colum
    :type df: Pandas dataframe
    :param column: name of column with prepared text data
    :type column: string
    :param n_gram: quantity of words in each n_gram
    :type n_gram: int
    :return: dataframe with tf-idf scores of n_grams
    :rtype: Pandas dataframe
    """
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
