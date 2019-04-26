
#%%
import pandas as pd
from definitions import ROOT_DIR

#%%
q_df = pd.read_csv(ROOT_DIR+'/raw_data/quora_dataset.csv')
print(q_df.loc[q_df['target']==1].head(3))

#%%
j_df = pd.read_csv(ROOT_DIR+'/raw_data/jigsaw_dataset.csv')
print(j_df)