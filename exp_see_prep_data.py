import numpy as np 
import pandas as pd
from definitions_toxicity import ROOT_DIR

check = np.load(ROOT_DIR + '/prep_data/x_tr_tokenized.npy')
print(check.shape)
print(check[2,0,:])
print(check[2,1,:])
print(check[2,3,:])

check_text = pd.read_csv(ROOT_DIR + '/temp_data/x_tr_prep.csv')
print(check_text.iloc[2].values)