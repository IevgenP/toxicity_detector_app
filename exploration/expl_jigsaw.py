
#%%
import nltk
import collections
import warnings
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from definitions import ROOT_DIR
nltk.download('punkt')

import string
from src.preprocessing.custom_transformers import PunctuationRemover, StopWordsRemover, IntoLowerCase, ShortToLong
from src.visualizers.visualiers import get_stats, plot_counter, count_words
sns.set(style="darkgrid")

#%%
# Download Jigsaw database
j_df = pd.read_csv(ROOT_DIR+'/raw_data/jigsaw_dataset.csv')


#%%
# Replace dataset specific symbols to common ones
# j_df['question_text'] = j_df['question_text'].str.replace(PUT SPECIAL SYMBOL HERE, "'")

#%%
j_df.head(3)

#%%
# Target value distribution
for column in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    plt.figure()
    sns.countplot(x=column, data=j_df).set_title("Distribution of {} comments.".format(column))

#%%
# Add column with number of words
j_df['num_words'] = j_df['comment_text'].apply(
    lambda x: len([token for token in x.split(" ") if token != ""])
)

#%%
# Print stats for number of words
print("Stats for whole dataset:")
get_stats(j_df, 'words', 'num_words')


#%%
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10,10))
axes = axes.flatten()
number = 0
for column in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    for i in [0,1]:
        sns.distplot(j_df.loc[j_df[column]==i, 'num_words'], ax=axes[number]).set_title(column+" "+str(i))
        axes[number].set_xlim(0,300)
        number += 1
plt.tight_layout()
plt.show()

#%%
# Each plot shows that toxic comments tend to be short.
