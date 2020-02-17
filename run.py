
## packages to be imported:
import os
os.chdir('C:\\Users\\Jack\\Desktop\\scraps\\bida')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# modules from packages
import sklearn.linear_model as lm
from sklearn.neural_network import MLPClassifier

# own modules
from functions.splitter import *


## import dataset using pandas 

infile = '.\\glass.csv'
df = pd.read_csv(infile, sep=',')

df['Type']=(df['Type']+64).apply(chr)


## analyse by type
df_by_type = df.groupby('Type')
df_by_type.describe()
df_means = df_by_type.mean()

dft = df_means.transpose()
dft.plot(kind='bar')
plt.show()

## correlation

## PCA
x_df = df.drop(['Type'], axis=1)
x_df_norm = (x_df-x_df.mean()) / x_df.std()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x_df_norm)
x_pca = pca.transform(x_df_norm)

tmp=pd.DataFrame()
tmp['Type'] = df['Type']
tmp['1st PC'] = x_pca[:,0]
tmp['2nd PC'] = x_pca[:,1]

sns.set()
ax = sns.scatterplot(x='1st PC', y='2nd PC',  hue='Type', data=tmp)
ax.set(xlabel='1st PC', ylabel='2nd PC')
plt.show()

sns.set()
map= pd.DataFrame(pca.components_,columns=list(x_df.columns))
sns.heatmap(map,cmap='RdBu')
plt.show()


dft.transpose().plot.bar(subplots= True, figsize=(7,10), legend=False)
plt.show()


## clean data (normalize)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


## test and train (+ noise, + normalize)
df
df_norm = x_df_norm
df_noise = df
df_norm['Type'] = df['Type']

train, test = split_data(df, predictand='Type', test_share=0.20)
y_train = train.Type
y_test = test.Type
x_train = train[train.columns.difference(['Type'])]
x_test = test[test.columns.difference(['Type'])]
    
    ## split data
    


## create multinomial reg model
## with differebnt ditributions
lr = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')
lr.fit(x_train, y_train)

res = lr.predict(x_test)


## create ann
## https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

## eval model

    ## numbers
    ## graphs


eval = pd.DataFrame({'obs': y_test, 'pred': res})

eval['truth'] = eval.obs == eval.pred
eval['measured'] = True

eval_cat = eval.groupby(['obs']).sum()
eval_cat['acc'] = eval_cat['truth']/eval_cat['measured']
eval_cat['obs'] = eval_cat.index

sns.set()
sns.barplot(x='obs',y='acc', data=eval_cat)
plt.show()