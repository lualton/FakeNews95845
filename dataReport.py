import pandas as pd
import numpy as np

data = pd.read_csv('data_Wang_expandedfeatures.csv')

data['dataset'].value_counts()

df1 = data[data['dataset'] == 'Rosas-Celebrity'].reset_index(drop=True)

df1[['text', 'label']].tail()

df1['text'][498]
