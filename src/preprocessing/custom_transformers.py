import string
from sklearn.base import BaseEstimator, TransformerMixin

class PunctuationRemover(BaseEstimator, TransformerMixin):
    """Class for punctuation removal from Pandas Dataframe columns"""

    def __init__(self):
        self
    
    def fit(self, df, columns):
        """
        Method that take input data for further transform
        
        :param df: dataset with columns that require punctuation removal
        :type df: pandas DataFrame
        :param column: columns name
        :type column: list
        """
        self.df = df
        self.columns = columns

    def transform(self):
        """
        Method that transforms input data
        
        :return: dataset where punctuation is removed from selected columns
        :rtype: pandas DataFrame
        """
        
        for column in self.columns:
            self.df[column] = self.df[column].apply(
                lambda x: "".join([char for char in x if char not in string.punctuation])
            )
        return self.df


class StopWordsRemover(BaseEstimator, TransformerMixin):
    """Class for removing stop words from Pandas Dataframe columns"""

    def __init__(self):
        self
    
    def fit(self, df, columns, stopwords_list):
        self.df = df
        self.columns = columns
        self.stopwords_list = stopwords_list
    
    def transform(self):
        for column in self.columns:
            self.df[column] = self.df[column].apply(
                lambda x: [word for word in x.split(" ") if word not in self.stopwords_list]
            )
        return self.df