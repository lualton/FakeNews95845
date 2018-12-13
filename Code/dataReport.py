import pandas as pd
import numpy as np

data = pd.read_csv('data_Wang_expandedfeatures.csv')

data[data['dataset'] == 'Rosas-News']['label'].value_counts()


data['dataset'].value_counts()
data.columns
# Data, Counts, Average Words, Average Sentences

data['totalwords'] = data['text'].str.split().str.len()

data.groupby('dataset')['totalwords'].min()




(df.groupby(['cluster', 'org'], as_index=False).mean()
            .groupby('cluster')['time'].mean())


df1 = data[data['dataset'] == 'Rosas-Celebrity'].reset_index(drop=True)

df1[['text', 'label']].tail()

df1['text'][498]
