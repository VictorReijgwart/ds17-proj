# Ingestion and lagging of the input data #

# Import libraries
import os.path
import pandas as pd
import numpy as np
import code

# Settings
featureFile = 'clean_data/GDELT_I/ire_esp_prt_cyp_ita_brussels_strassburg_frankfurt.csv'
targetFile = 'clean_data/From2007.csv'

lags = 5

# dstFile = 'features/'+activeFile

# Code
feature_df = pd.read_csv(featureFile)
target_df = pd.read_csv(targetFile, usecols=['Date','S_Log_Exchange'])

feature_df['c_norm'] = feature_df['c_yes']/feature_df['c']
feature_df['c_norm'] = feature_df['c_norm']/max(feature_df['c_norm'])

target_df['US_ex'] = np.exp(target_df['S_Log_Exchange'])

for lag in range(1,lags+1):
    feature_df['c_norm_lag'+str(lag)] = feature_df['c_norm'].shift(lag)
# print(feature_df[:lags])
feature_df.drop(feature_df.index[:lags], inplace=True)

print(feature_df)

feature_df.to_csv(dstFile, index=False)

print('Success')