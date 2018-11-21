### This will join data from our three datasets


import pandas as pd
import numpy as np
import re


buzz = pd.read_csv("buzzfeedcorpus.csv")
kaggle = pd.read_csv("fake_or_real_news.csv")
news = pd.read_csv('news.csv')
celebrity = pd.read_csv("celebrity.csv")

buzz.shape
kaggle.shape
news.shape
celebrity.shape

buzz.columns
kaggle.columns
news.columns
celebrity.columns

buzz['veracity'].value_counts()
kaggle['label'].value_counts()
news['class'].value_counts()
celebrity['class'].value_counts()

## We want similar columns to rbind
## Title, author, text, source, label

kaggle['author'], kaggle['url'] = np.nan, np.nan
kaggle = kaggle.drop('Unnamed: 0', axis=1)
kaggle = kaggle[['title', 'author', 'text', 'url', 'label']]

buzz.rename(columns = {'veracity': 'label'}, inplace=True)

news.rename(columns = {'content': 'text', 'class': 'label', 'file_name':'url'}, inplace=True)
celebrity.rename(columns = {'content': 'text', 'class': 'label', 'file_name':'url'}, inplace=True)
news['title'], news['author'], celebrity['title'], celebrity['author'] = np.nan, np.nan, np.nan, np.nan
news, celebrity = news[['title', 'author', 'text', 'url', 'label']], celebrity[['title', 'author', 'text', 'url', 'label']]


data = pd.concat([buzz, kaggle, news, celebrity])
data.head()

data['label'].value_counts()
type(data['label'][0])
len(data)

data = data.replace(('mostly true', 'legit', 'REAL'), 'real')
data = data.replace(('FAKE', 'fake', 'mostly false', 'no factual content'), 'fake')

data['label'].value_counts()

data.to_csv('combinedData.csv', index=False)
