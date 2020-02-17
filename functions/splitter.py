import pandas as pd
import numpy as np

def split_data(df, predictand='Type', test_share=0.10):

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for group in df[predictand].unique():
        df_group = df[df[predictand]==group]
            
        n_items = df_group.shape[0]
        x_len = round(n_items * test_share)
            
        rand_ind = [np.random.rand() for n in range(1,n_items)]
            
        for n in range(1, x_len):
            max_rand = max(rand_ind) 
            test_df = test_df.append(df_group.iloc[rand_ind.index(max_rand)])
            rand_ind.remove(max_rand)

        train_ind = [n-1 for n in df.index if n not in test_df.index]
        train_df = df.iloc[train_ind]




def split_data_random(df, predictand='Type', test_share=0.10):

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    n_items = df.shape[0]
    x_len = round(n_items * test_share)
            
    rand_ind = [np.random.rand() for n in range(1,n_items)]
            
    for n in range(1, x_len):
        max_rand = max(rand_ind) 
        test_df = test_df.append(df.iloc[rand_ind.index(max_rand)])
        rand_ind.remove(max_rand)

    train_ind = [n-1 for n in df.index if n not in test_df.index]
    train_df = df.iloc[train_ind]


    return train_df, test_df
