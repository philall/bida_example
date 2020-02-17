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



def lr_mod(x_train, y_train, x_test, y_test):
    lr = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    lr.fit(x_train, y_train)
    res = lr.predict(x_test)

    return res



def clf_mod(x_train, y_train, x_test, y_test):
    y = list(y_train)
    X = [list(n) for index,n in x_train.iterrows()]
    clf = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-5, verbose=10, hidden_layer_sizes=(20, 3), random_state=1)
    clf.fit(X, y)
    X_test = [list(n) for index,n in x_test.iterrows()]
    res_clf = clf.predict(X_test)

    return res_clf






if __name__ == '__main__':
    infile = '.\\data\\glass.csv'
    df = pd.read_csv(infile, sep=',')

    df['Type']=(df['Type']+64).apply(chr)

    ## clean data (normalize)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df_norm = normalize(df, 'Type')

    results = pd.DataFrame()
    runs = 20
    for run in range(1, runs):
        ## prepare test and train
        train, test = split_data_random(df_norm, predictand='Type', test_share=0.20)
        y_train = train.Type
        y_test = test.Type
        x_train = train[train.columns.difference(['Type'])]
        x_test = test[test.columns.difference(['Type'])]
    
        train_noise = noise(train, 'Type', mu=0, sigma=0.1)
        x_train_noise = train_noise[train_noise.columns.difference(['Type'])]

        res = lr_mod(x_train, y_train, x_test, y_test)
        res_noise = lr_mod(x_train_noise, y_train, x_test, y_test)
        res_clf = clf_mod(x_train, y_train, x_test, y_test)


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

