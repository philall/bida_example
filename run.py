
## packages to be imported:
import os
os.chdir('C:\\Users\\Jack\\source\\repos\\bida_example')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# modules from packages
import sklearn.linear_model as lm
from sklearn.neural_network import MLPClassifier

# own modules
from functions.splitter import *
from functions.math import *


## import dataset using pandas 

infile = '.\\data\\glass.csv'
df = pd.read_csv(infile, sep=',')


## analyse by type
df['Type'].unique()

df['Type']=(df['Type']+64).apply(chr)

## clean data (normalize)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)



df_by_type = df.groupby('Type')
df_by_type.count()
df_by_type.describe()
df_means = df_by_type.mean()

dft = df_means.transpose()
dft.plot(kind='bar')
plt.show()

## normalize
x_df = df.drop(['Type'], axis=1)
x_df_norm = (x_df-x_df.mean()) / x_df.std()
##df_norm = normalize(df, 'Type')
df_norm_by_type = df_norm.groupby('Type')
df_norm_means = df_norm_by_type.mean()
df_norm_means.plot(kind='bar')
plt.legend(loc='upper left', ncol=3)
plt.show()

## correlation
pd.plotting.scatter_matrix(x_df_norm)
plt.show()
x_df_corr = x_df_norm.corr()

x_tmp = x_df_corr.replace(1, 0)
x_tmp.max()


## PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x_df_norm)
x_pca = pca.transform(x_df_norm)

tmp=pd.DataFrame()
tmp['Type'] = df_norm['Type']
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

## test and train (+ noise, + normalize)
df_norm = x_df_norm
df_noise = noise(df, 'Type', mu=0, sigma=0.0001)

df_norm['Type'] = df['Type']

train, test = split_data(df, predictand='Type', test_share=0.20)
y_train = train.Type
y_test = test.Type
x_train = train[train.columns.difference(['Type'])]
x_test = test[test.columns.difference(['Type'])]



## add noise
df_noise = noise(df, 'Type', mu=0, sigma=0.0001)

train, test = split_data(df_noise, predictand='Type', test_share=0.20)
y_train = train.Type
y_test = test.Type
x_train = train[train.columns.difference(['Type'])]
x_test = test[test.columns.difference(['Type'])]


## create multinomial reg model
## with differebnt ditributions
lr_noise = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')
lr_noise.fit(x_train, y_train)

res = lr_noise.predict(x_test)


## create ann
## https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

y = list(df_norm['Type'])
X = [list(n) for index,n in x_df_norm.iterrows()]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 2), random_state=1)
clf.fit(X, y)

X_test = [list(n) for index,n in x_test.iterrows()]

res = clf.predict(X_test)


## eval model

    ## numbers
    ## graphs


eval = pd.DataFrame({'obs': y_test, 'pred': res})

eval['truth'] = eval.obs == eval.pred
eval['measured'] = True
sum(eval['truth'])/sum(eval['measured'])

eval_cat = eval.groupby(['obs']).sum()
eval_cat['acc'] = eval_cat['truth']/eval_cat['measured']
eval_cat['obs'] = eval_cat.index

sns.set()
sns.barplot(x='obs',y='acc', data=eval_cat)
plt.show()
