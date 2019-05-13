
#%%
import nltk
import collections
import warnings
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
from definitions import ROOT_DIR
nltk.download('punkt')

import string
from src.preprocessing.custom_transformers import PunctuationRemover, StopWordsRemover, IntoLowerCase


#%%
# Download Quora database
q_df = pd.read_csv(ROOT_DIR+'/raw_data/quora_dataset.csv')
print(q_df.loc[q_df['target']==1].head(3))


#%%
# Target value distribution
sns.set(style="darkgrid")
sns.countplot(x='target', data=q_df)


#%%
# Function for printing min, max, mean and median
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


#%%
# Add column with number of words
q_df['num_words'] = q_df['question_text'].apply(
    lambda x: len([token for token in x.split(" ") if token != ""])
)


#%%
# Print stats for number of words
print("Stats for whole dataset:")
get_stats(q_df, 'words', 'num_words')


#%%
# One word as a question? It is interesting to look into.
q_df.loc[q_df['num_words']==1, ]
# It seems that all one question are considered as trolling.
# This is a questionable approach of those who labeled the data.
# But for this specific case I will agree with it.


#%%
# Stats for number of words if target value is 0

q_df_0 = q_df.copy().loc[q_df['target']==0, ]
print("Stats for 0 target values:")
get_stats(q_df_0, 'words', 'num_words')


#%%
# Stats for number of words if target value is 1

q_df_1 = q_df.copy().loc[q_df['target']==1, ]
print("Stats for 0 target values:")
get_stats(q_df_1, 'words', 'num_words')


#%%
# It seems like authors of troll questions are too busy to type more than 64 words.
# Also distribution of number of words shows spikes with some ordered pattern.
# My suggestion is that trolling questions were automatically created (either for pupose of this
# competition or by Quora's trolls, when they posted questions).
# The conclusion from this section of analysis: max of 40 words is enough for padding.


#%%
# Get the most common words for both target values
# q_df['tokens'] = q_df['question_text'].apply(nltk.tokenize.word_tokenize)
# print(q_df.head(5))


#%%
# Download stop words from nltk library
nltk.download('stopwords')
eng_stop_words = nltk.corpus.stopwords.words('english') 


#%%
# Add column for text transformations
q_df['prep_text'] = q_df['question_text'].copy()

# create pipeline for transformation
pipeline = Pipeline(
    steps=[
        ('punctuation', PunctuationRemover(['prep_text'])),
        ('lowercase', IntoLowerCase(['prep_text'])),
        ('stopwords', StopWordsRemover(['prep_text'], eng_stop_words))
    ]
)

pipeline.fit_transform(q_df)
q_df.to_csv(ROOT_DIR + '/temp_data/quora_df_no_stop_words.csv', index=False)


#%%
# Get the most common words for both target values
def plot_counter(counter_output, top_n, n_gram, text):
    count_df = pd.DataFrame.from_dict(counter_output, orient='index').reset_index()
    count_df = count_df.rename(columns={'index': 'word', 0: 'count'})
    count_df = count_df.sort_values(by='count', axis=0, ascending=False)
    sns.barplot(x="count", y="word", data=count_df.head(top_n), palette="Blues_d").set_title('{} frequency count{}.'.format(n_gram.capitalize(), text))

def count_words(df, column, n_gram, top_n, text=None):
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
    
    # plot results
    plot_counter(count, top_n, n_gram, text)


#%%
count_words(q_df.head(100), 'prep_text', 'unigrams', 15, ' for whole dataset')



#%%
j_df = pd.read_csv(ROOT_DIR+'/raw_data/jigsaw_dataset.csv')
print(j_df)


# EDA for q_df
#%%
