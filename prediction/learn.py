# Ingestion and lagging of the input data #

# Import libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import code

# Settings
merged_file = 'clean_data/From2007.csv'
lags = 120

param_GS = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 1],
    'gamma': [0.001, 0.01, 0.1],
    'verbose': [0]
    }

# Code
merged_df = pd.read_csv(merged_file)#, usecols=['Date','S_Log_Exchange'])

merged_df.drop(['Date'], axis=1, inplace=True)
merged_df.rename(columns={'S_Log_All_Crisis_Yes': 'ft', 'S_Log_Exchange': 'target'}, inplace=True)
merged_df = merged_df[['target', 'ft']]

for lag in range(1,lags+1):
    merged_df['ft_lag'+str(lag)] = merged_df['ft'].shift(lag)
merged_df.drop(merged_df.index[:lags], inplace=True)

n_train = int(np.floor(merged_df.shape[0]*0.94))

y_train = merged_df['target'].values[:n_train]
X_train = merged_df.drop(['target'], axis=1).values[:n_train]
y_test = merged_df['target'].values[n_train:]
X_test = merged_df.drop(['target'], axis=1).values[n_train:]

model = GridSearchCV(
    svm.SVR(),
    scoring='neg_mean_squared_error',
    param_grid=param_GS,
    cv = 10,
    n_jobs = 3,
)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

plt.plot(range(n_train), y_train_pred, label='y_pred_train')
plt.plot(range(n_train, merged_df.shape[0]), y_test_pred, label='y_pred_test')
plt.plot(np.append(y_train, y_test), label='y_test')
plt.legend()
plt.show()
# code.interact(local=locals())
