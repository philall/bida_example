## packages to be imported:

import os # os interface
os.chdir('C:\\Users\\Jack\\source\\repos\\bida_example')    # strings notation either '' or ""

import pandas as pd                                         # 'as' defines an alias for a package
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                                       # visualisation building on matplotlib

# modules from packages
import sklearn.linear_model as lm                           # with a '.' submodules can be selected
from sklearn.neural_network import MLPClassifier            # 'from ... import' imports a submodule directly
from sklearn.decomposition import PCA

## own modules
from functions.splitter import *                            # with '*' all modules are directly imported
from functions.math import *



## import dataset using pandas 

infile = '.\\data\\glass.csv'                               # variables are implicitly allocated!
df = pd.read_csv(infile, sep=',')                           # 'pd' is the alias for pandas. / pandas allows many easy possibilities of importing various data sources
df                                                          # df is of type dataframe (similar to R)
df.head()                                                   # '.function()' implies a funciton working on the object itself
df.shape                                                    # '.variable' implies a variable in the 

## analyse by type
df['Type'].unique()                                         # '["Type"]' defines a certain column

df['Type']=(df['Type']+64).apply(chr)                       # 'apply(function)' applies a function for each row element 


## clean data (normalize)
df.dropna(inplace=True)                                     # functions can take arguments (inside the brackets)
df.drop_duplicates(inplace=True)



df_by_type = df.groupby('Type')                             # if in defined order functions take arguments in location (no 'varname=' needed)
df_by_type.describe()                                       # pandas way of summarize()/R ... easy dataset insights
df_means = df_by_type.mean()

dft = df_means.transpose()
dft.plot(kind='bar')                                        # pandas is also able to create graphical objects which can be passed to visual packages
plt.show()                                                  # 'plt' is the alias for matplotlib

## normalize
x_df = df.drop(['Type'], axis=1)
df_norm = normalize(df, 'Type')                             # this is a self-made function
df_norm_by_type = df_norm.groupby('Type')
df_norm_means = df_norm_by_type.mean()
df_norm_means.plot(kind='bar')
plt.legend(loc='upper left', ncol=3)                        # matplotlib is sort of the equivalent to ggplot, maybe not quite as strong
plt.show()                                                  # that's why we'll ad some seaborn ;)

## correlation
pd.plotting.scatter_matrix(x_df_norm)                       # again some vary easy, but helpful analysis with pandas
plt.show()
x_df_corr = x_df_norm.corr()

x_tmp = x_df_corr.replace(1, 0)
x_tmp.max()


## PCA - let's get deep!                                    
## https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

pca = PCA(n_components=2)                                   # only choose 2 main components -> easy visualization, most variance
pca.fit(x_df_norm)

pca.explained_variance_ratio_                               # so how much variancle is actually explained

x_pca = pca.transform(x_df_norm)


tmp=pd.DataFrame()                                          # quick transformation to dataframe for easier handling
tmp['Type'] = df_norm['Type']
tmp['1st PC'] = x_pca[:,0]
tmp['2nd PC'] = x_pca[:,1]


sns.set()                                                               # set default aesthetic parameters in one step
ax = sns.scatterplot(x='1st PC', y='2nd PC',  hue='Type', data=tmp)     # allocate the scatterplot with default aesthetic to ax
ax.set(xlabel='1st PC', ylabel='2nd PC')                                # add aesthetic
plt.show()

sns.set()
map= pd.DataFrame(pca.components_,columns=list(x_df.columns))
sns.heatmap(map,cmap='RdBu')
plt.show()


#results = pd.DataFrame()                                               # initiate dataframe for data collection (append data)
#runs = 20
#for run in range(1, runs):                                             # intendation is required!... https://www.azquotes.com/quote/765138
    
## prepare test and train
train, test = split_data_random(df_norm, predictand='Type', test_share=0.20)            # self made function, but sklearn would have also provided a solution in itself
y_train = train.Type                                                                    # pandas provides another easy way to access columns
y_test = test.Type
x_train = train[train.columns.difference(['Type'])]
x_test = test[test.columns.difference(['Type'])]
    
## create multinomial logistic regression
## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

mlr = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')              # set up logistic regression object 
mlr.fit(x_train, y_train)                                                               # train the model
res = mlr.predict(x_test)                                                               # test it

sum(y_test == res)/y_test.shape[0]                                                      # 'true' (boolean) is summed as 1.0, 'false' as 0.0


## add noise
train_noise = noise(train, 'Type', mu=0, sigma=0.1)                                     # self made function (see .\functions\math.py)
x_train_noise = train_noise[train_noise.columns.difference(['Type'])]
    
mlr_noise = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mlr_noise.fit(x_train_noise, y_train)
res_noise = mlr_noise.predict(x_test)


## create ann
## https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

y = list(y_train)
X = [list(n) for index,n in x_train.iterrows()]                                         # one row for-loop !!very nice python feature
clf = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-5, verbose=10, hidden_layer_sizes=(20, 3), random_state=1)
clf.fit(X, y)
X_test = [list(n) for index,n in x_test.iterrows()]
res_clf = clf.predict(X_test)


### set temporary results row
#run_eval = pd.DataFrame({'LR': sum(y_test == res)/y_test.shape[0],
#                         'LR_ns': sum(y_test == res_noise)/y_test.shape[0],
#                         'CLF': sum(y_test == res_clf)/y_test.shape[0]},
#                        index=[run])

### append to main results dataframe
#results = results.append([run_eval])


## visualize results
#results_upv = pd.melt(results, value_vars=['CLF', 'LR', 'LR_ns'])                       # again data transformation made easy by pandas

#sns.set()                                                               
#ax = sns.scatterplot(x='variable', y='value',  hue='variable', data=results_upv) 
#ax.set(xlabel='1st PC', ylabel='2nd PC')                                
#plt.show()


## make it nicer and stick into a boxplot
#medians = results_upv.groupby(['variable'])['value'].median()
#vertical_offset = 0.01

#sns.set()
#box_plot = sns.boxplot(x="variable", y="value", data=results_upv)

#for xtick in box_plot.get_xticks():
#    box_plot.text(xtick, 
#                  round(medians[xtick],3),
#                  round(medians[xtick],3),
#                  horizontalalignment='center',
#                  color='w')
    
#plt.ylim(0, 1)
#plt.ylabel('Score')
#plt.xlabel('Model')
#plt.show()
