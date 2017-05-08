# Import libraries
import os.path
import pandas as pd
import numpy as np
import code
import datetime


merged_df = pd.read_csv('features/merged.csv')


merged_df.to_csv('features/targets.csv', index=False, columns=['US_ex'])
# merged_df.drop('US_ex', axis=1)
# print(merged_df)

merged_df.to_csv('features/features.csv', index=False)
