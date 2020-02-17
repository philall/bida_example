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
df_norm = normalize(df, 'Type')
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

results = pd.DataFrame()
runs = 20
for run in range(1, runs):
    ## prepare test and train
    train, test = split_data_random(df_norm, predictand='Type', test_share=0.20)
    y_train = train.Type
    y_test = test.Type
    x_train = train[train.columns.difference(['Type'])]
    x_test = test[test.columns.difference(['Type'])]
    
    ## create multinomial reg model
    lr = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    lr.fit(x_train, y_train)
    res = lr.predict(x_test)


    ## add noise
    train_noise = noise(train, 'Type', mu=0, sigma=0.1)
    x_train_noise = train_noise[train_noise.columns.difference(['Type'])]
    
    lr_noise = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    lr_noise.fit(x_train_noise, y_train)
    res_noise = lr_noise.predict(x_test)


    ## create ann
    ## https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
    y = list(y_train)
    X = [list(n) for index,n in x_train.iterrows()]
    clf = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-5, verbose=10, hidden_layer_sizes=(20, 3), random_state=1)
    clf.fit(X, y)
    X_test = [list(n) for index,n in x_test.iterrows()]
    res_clf = clf.predict(X_test)


    run_eval = pd.DataFrame({'LR': sum(y_test == res)/y_test.shape[0], 'LR_ns': sum(y_test == res_noise)/y_test.shape[0], 'CLF': sum(y_test == res_clf)/y_test.shape[0]}, index=[run])

    results = results.append([run_eval])



results_upv = pd.melt(results, value_vars=['CLF', 'LR', 'LR_ns'])

medians = results_upv.groupby(['variable'])['value'].median()
vertical_offset = 0.01


sns.set()
box_plot = sns.boxplot(x="variable", y="value", data=results_upv)

for xtick in box_plot.get_xticks():
    box_plot.text(xtick, round(medians[xtick],3),round(medians[xtick],3), 
            horizontalalignment='center',color='w')
    
plt.ylim(0, 1)
plt.ylabel('Score')
plt.xlabel('Model')
plt.show()
