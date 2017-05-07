# Ingestion and lagging of the input data #

# Import libraries
import csv
import os.path
import pandas as pd
import numpy as np
import code

# Settings
activeFile = 'GDELT_I/ire_esp_prt_cyp_ita_brussels_strassburg_frankfurt.csv'
srcFile = 'clean_data/'+activeFile
usecols = ['SQLDATE', 'c_yes', 'c'] # columns to import from .csv file

lags = 5

dstFile = 'features/'+activeFile

# Code
if os.path.isfile(srcFile):
    data_frame = pd.read_csv(srcFile, usecols=usecols)
else:
    print('Source file not found')
    raise SystemExit

data_frame['c_norm'] = data_frame['c_yes']/data_frame['c']
data_frame['c_norm'] = data_frame['c_norm']/max(data_frame['c_norm'])

for lag in range(1,lags+1):
    data_frame['c_norm_lag'+str(lag)] = data_frame['c_norm'].shift(lag)
# print(data_frame[:lags])
data_frame.drop(data_frame.index[:lags], inplace=True)

data_frame.to_csv(dstFile, index=False)

print('Success')