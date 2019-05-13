import string
from sklearn.base import BaseEstimator, TransformerMixin

class PunctuationRemover(BaseEstimator, TransformerMixin):
    """Class for punctuation removal from Pandas Dataframe columns"""

    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, df):
        """
        Method that take input data for further transform
        
        :param df: dataset with columns that require punctuation removal
        :type df: pandas DataFrame
        :param column: columns name
        :type column: list
        """
        return self

    def transform(self, df):
        """
        Method that transforms input data
        
        :return: dataset where punctuation is removed from selected columns
        :rtype: pandas DataFrame
        """
        
        for column in self.columns:
            df[column] = df[column].apply(
                lambda x: "".join([char for char in x if char not in string.punctuation])
            )
        return df


class StopWordsRemover(BaseEstimator, TransformerMixin):
    """Class for removing stop words from Pandas Dataframe columns"""

    def __init__(self, columns, stopwords_list):
        self.columns = columns
        self.stopwords_list = stopwords_list
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        for column in self.columns:
            df[column] = df[column].apply(
                lambda x: " ".join([word for word in x.split(" ") if word not in self.stopwords_list])
            )
        return df


class IntoLowerCase(BaseEstimator, TransformerMixin):
    """Class for transforming all words to lower case"""

    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        for column in self. columns:
            df[column] = df[column].astype(str).str.lower()
        return df