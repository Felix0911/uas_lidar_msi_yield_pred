# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:30:20 2022

@author: Fei

Description:
    Calculation correlation matrix from a pandas dataframe.
"""


import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {'A': [45,37,42,35,39],
        'B': [38,31,26,28,33],
        'C': [10,15,17,21,12]
        }

df = pd.DataFrame(data,columns=['A','B','C'])
corrMatrix = df.corr()
mask = np.zeros_like(corrMatrix)
mask[np.triu_indices_from(mask)] = True
sn.heatmap(corrMatrix, mask=mask, annot=True)
plt.show()