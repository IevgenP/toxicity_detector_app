
#%%
import nltk
import numpy as np
import collections
import warnings
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from definitions_toxicity import ROOT_DIR
nltk.download('punkt')

import string
#from src.preprocessing.custom_transformers import PunctuationRemover, StopWordsRemover, IntoLowerCase, ShortToLong
#from src.visualizers.visualiers import get_stats, get_tf_idf_scores
sns.set(style="darkgrid")

#%%
# Download Jigsaw database
j_df = pd.read_csv(ROOT_DIR+'/raw_data/jigsaw_dataset.csv')


#%%
# see cases with threats
print(j_df.loc[j_df['toxic']==1, ].shape)

tox_only = j_df_2.loc[
    (j_df_2['target']==1) &
    (j_df_2['severe_toxicity']==0) &
    (j_df_2['obscene']==0) &
    (j_df_2['threat']==0) &
    (j_df_2['insult']==0) & 
    (j_df_2['identity_attack']==0),
]
tox_only.shape

#%%
j_df_2[['target', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']].sum()


#%%
# Combine datasets from bowth Jigsaw competitions
j_df_2 = pd.read_csv(ROOT_DIR+'/raw_data/jigsaw_dataset_2.csv')
print(j_df.head(5))
print(j_df_2.head(5))


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
print(j_df.head(3))

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

#%%
# Download stop words from nltk library
nltk.download('stopwords')
eng_stop_words = nltk.corpus.stopwords.words('english') 


#%%
# Add column for text transformations
j_df['prep_text'] = j_df['comment_text'].copy()

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
#j_df_transf = pipeline.fit_transform(j_df)
#j_df_transf.to_csv(ROOT_DIR + '/temp_data/jigsaw_df_transformed.csv', index=False)
j_df_transf = pd.read_csv(ROOT_DIR + '/temp_data/jigsaw_df_transformed.csv')


#%%
# The following analysis must use tf-idf otherwise there are cases where comment 
# consists of one word only repeated many times (example: "fucksex").


#%%
# Count words frequency for toxic comments only
df_tox_uni = get_tf_idf_scores(j_df_transf.loc[j_df_transf['toxic'] == 1, ],
                               column='prep_text', 
                               n_gram=1)
print(df_tox_uni.head(10))

# plot top n words
sns.barplot(
    x="tf_idf_score",
    y="words", 
    data=df_tox_uni.head(15), 
    palette="Blues_d"
).set_title('Toxic comments unigrams by IF-IDF')

#%%
df_tox_bi = get_tf_idf_scores(j_df_transf.loc[j_df_transf['toxic'] == 1, ],
                              'prep_text',
                              2)

# plot top n words
sns.barplot(
    x="tf_idf_score",
    y="words", 
    data=df_tox_bi.head(15), 
    palette="Blues_d"
).set_title('Toxic comments bigrams by IF-IDF')

#%%
# Many bigrams looks like repetition of the same word. It deserves additional investigation.

#%%
print(j_df_transf.shape)
j_df_transf = j_df_transf.dropna()
print(j_df_transf.shape)

#%%
print(j_df_transf.loc[j_df_transf['prep_text'].str.contains('faggot faggot'), 'prep_text'].values)
print("")
print(j_df_transf.loc[j_df_transf['prep_text'].str.contains('shit shit'), 'prep_text'].values)
print("")
print(j_df_transf.loc[j_df_transf['prep_text'].str.contains('ass ass'), 'prep_text'].values)

#%%
#Some cases are trully repetition, but not all of them. 
#Let's try to exclude most obvious with help of regex.

#%%
import re 
def duplicates_cutter(df, column, rep_text_list):
    df_c = df.copy()
    for rep_text in rep_text_list:
    #     df_c[column] = df_c[column].replace(
    #         "(({})[\s,.?\w]*)".format(rep_text), rep_text, regex=True
    # )
        df_c[column] = df_c[column].str.replace(
            r"(({})[\s,.?\w]*)".format(rep_text), rep_text
    )
    return df_c

#%%
j_df_transf_2 = duplicates_cutter(j_df_transf, 'prep_text', df_tox_bi['words'].tolist()[:1000])

#%%
df_tox_bi_2 = get_tf_idf_scores(j_df_transf_2.loc[j_df_transf_2['toxic'] == 1, ],
                              'prep_text',
                              2)

# plot top n words
sns.barplot(
    x="tf_idf_score",
    y="words", 
    data=df_tox_bi_2.head(15), 
    palette="Blues_d"
).set_title('Toxic comments bigrams by IF-IDF')


#%%
print(j_df_transf_2.loc[j_df_transf_2['prep_text'].str.contains('faggot faggot'), 'prep_text'].values)
print("")
print(j_df_transf_2.loc[j_df_transf_2['prep_text'].str.contains('shit shit'), 'prep_text'].values)
print("")
print(j_df_transf_2.loc[j_df_transf_2['prep_text'].str.contains('ass ass'), 'prep_text'].values)

#%%
df_tox_bi_2.head(10)



#%%
print(j_df_transf_2.loc[j_df_transf_2['prep_text'].str.contains('cuntbag'), 'prep_text'].values)


#%%
