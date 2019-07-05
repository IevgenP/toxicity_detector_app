#%%
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
from definitions_toxicity import ROOT_DIR

y_tr = np.load(ROOT_DIR + '/prep_data/y_tr.npy')
print('y_tr', y_tr.sum(axis=0))

y_val = np.load(ROOT_DIR + '/prep_data/y_val.npy')
print('y_val', y_val.sum(axis=0))

y_test = np.load(ROOT_DIR + '/prep_data/y_test.npy')
print('y_test', y_test.sum(axis=0))

#%%
