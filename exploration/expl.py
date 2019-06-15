
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
from src.preprocessing.custom_transformers import PunctuationRemover, StopWordsRemover, IntoLowerCase, ShortToLong
from src.visualizers.visualiers import get_stats, plot_counter, count_words


#%%
# Download Quora database
q_df = pd.read_csv(ROOT_DIR+'/raw_data/quora_dataset.csv')
print(q_df.loc[q_df['target']==1].head(3))

#%%
# Replace dataset specific symbols to common ones
q_df['question_text'] = q_df['question_text'].str.replace("’", "'")

#%%
# Target value distribution
sns.set(style="darkgrid")
sns.countplot(x='target', data=q_df)

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
# Download stop words from nltk library
nltk.download('stopwords')
eng_stop_words = nltk.corpus.stopwords.words('english') 


#%%
# Add column for text transformations
q_df['prep_text'] = q_df['question_text'].copy()

# Create pipeline for transformation
pipeline = Pipeline(
    steps=[
        ('contracted', ShortToLong(['prep_text'])),
        ('punctuation', PunctuationRemover(['prep_text'])),
        ('lowercase', IntoLowerCase(['prep_text'])),
        ('stopwords', StopWordsRemover(['prep_text'], eng_stop_words))
    ]
)

#%%
# Transform the dataset using created pipeline
pipeline.fit_transform(q_df)
q_df.to_csv(ROOT_DIR + '/temp_data/quora_df_no_stop_words.csv', index=False)

#%%
# Count unigrams frequency for whole dataset
count_words(q_df, 'prep_text', 'unigrams', 15, ' for whole dataset')

#%%
# Count bigrams frequency for whole dataset
count_words(q_df, 'prep_text', 'bigrams', 15, ' for whole dataset')


#%%
# Count words frequency for sinsere questions only
count_words(q_df.loc[q_df['target'] == 0, ], 'prep_text', 'unigrams', 15, ' for sinsere questions only')

#%%
# Count words frequency for sinsere questions only
count_words(q_df.loc[q_df['target'] == 0, ], 'prep_text', 'bigrams', 15, ' for sinsere questions only')

#%%
# Count words frequency for sinsere questions only
count_words(q_df.loc[q_df['target'] == 0, ], 'prep_text', 'trigrams', 15, ' for sinsere questions only')


#%%
# Count words frequency for insinsere questions only
count_words(q_df.loc[q_df['target'] == 1, ], 'prep_text', 'unigrams', 15, ' for insinsere questions only')

#%%
# Count words frequency for insinsere questions only
count_words(q_df.loc[q_df['target'] == 1, ], 'prep_text', 'bigrams', 15, ' for insinsere questions only')

#%%
# Count words frequency for insinsere questions only
count_words(q_df.loc[q_df['target'] == 1, ], 'prep_text', 'trigrams', 15, ' for insinsere questions only')











#%% CHECKS ---------------------------------------------------------------------------------------- #
dff = pd.DataFrame({'prep_text': ["I've selected these! It's a right choice! I'm sure of it"]})

# temp_dict = {
#             "'ve": " have",
#             "'d": " had",
#             "'s": " is",
#             "'ll": " will",
#             "'re": " are",
#             "can't": "can not",
#             "couldn't": "could not",
#             "doesn't": "does not",
#             "shouldn't": "sould not",
#             "won't": "will not",
#             "let's": "let us",
#         }

# for column in ['prep_text']:
#     for i,j in temp_dict.items():
#         dff[column] = dff[column].str.replace(i, j)
# print(dff)

sl = ShortToLong(['prep_text'])
sl.fit_transform(dff)

#pipeline.fit_transform(dff)





#%%
j_df = pd.read_csv(ROOT_DIR+'/raw_data/jigsaw_dataset.csv')
print(j_df)


# EDA for q_df
#%%
’
