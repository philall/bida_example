import pandas as pd
import numpy as np 

def normalize(df, predictand):

    x_df = df.drop([predictand], axis=1)
    x_df_norm = (x_df-x_df.mean()) / x_df.std()
    x_df_norm[predictand]=df[predictand]

    return x_df_norm



def noise(df, predictand, mu=0, sigma=0.0001):

    x_df = df.drop([predictand], axis=1)
    for column in x_df.columns:
        noise = np.random.normal(mu, sigma, x_df[column].size) 
        x_df[column]=x_df[column]+noise
    
    x_df[predictand]=df[predictand]

    return x_df
