# Ingestion and lagging of the input data #

# Import libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import neural_network
from matplotlib import pyplot as plt
import code

# Settings
merged_file = 'clean_data/From2007_merged.csv'
lags = 100
target_lags = 1

param_GS = {
    # 'hidden_layer_sizes': [(10,6)],
    # 'kernel': ['rbf'],
    # 'C': [100],
    # 'epsilon': [0.1],
    # 'gamma': [0.001],
    # 'verbose': [1]
}

# Code
merged_df = pd.read_csv(merged_file)#, usecols=['Date','S_Log_Exchange'])

# print merged_df
# raise SystemExit

merged_df.drop(['Date'], axis=1, inplace=True)
merged_df.rename(columns={'S_Log_All_Crisis_Yes': 'news', 'greece': 'gtrend', 'S_Log_Exchange': 'target'}, inplace=True)
merged_df.drop(['gtrend'], axis=1, inplace=True)

for lag in range(1,target_lags+1):
    merged_df['target_lag'+str(lag)] = merged_df['target'].shift(lag)
cols = list(merged_df)
cols.insert(-1, cols.pop(cols.index('news')))

for lag in range(1,lags+1):
    merged_df['news_lag' + str(lag)] = merged_df['news'].shift(lag)
    # merged_df['gtrend_lag' + str(lag)] = merged_df['gtrend'].shift(lag)
    # merged_df['target_lag' + str(lag)] = merged_df['target'].shift(lag)
merged_df.drop(merged_df.index[:max([lags, target_lags])], inplace=True)

# print merged_df
# raise SystemExit
# print merged_df

n_train = int(np.floor(merged_df.shape[0]*0.70))
n_test = int(np.floor(merged_df.shape[0]-n_train))

y_train = merged_df['target'].values[:n_train]
X_train = merged_df.drop(['target'], axis=1).values[:n_train]
y_test = merged_df['target'].values[n_train:]

cols = [c for c in merged_df.columns if c[:6] == 'target']
X_test = merged_df.drop(cols, axis=1).values[n_train:]
# y = merged_df['target'].values
# X = merged_df.drop(['target'], axis=1).values

model = GridSearchCV(
    svm.SVR(),
    # linear_model.RidgeCV(),
    # linear_model.LinearRegression(),
    # neural_network.MLPRegressor(),
    scoring='neg_mean_squared_error',
    param_grid=param_GS,
    cv = 10,
    n_jobs = 4,
    verbose=3,
)

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)

y_pred_test = y_pred_train[-target_lags-1:-1]
# print X_train
# print X_test
for i in range(n_test-1):
    print 'Recurrent prediction step '+str(i)
    current_sample = np.append(y_pred_test[i:i+target_lags], X_test[i]).reshape(1,-1)
    y_pred_test = np.append(y_pred_test, model.predict(current_sample))
    # raise SystemExit
y_pred_test = y_pred_test[target_lags-1:,]
# code.interact(local=locals())

print('--- best params ---')
print(model.best_params_)
print('--- best score according to the Algorithm ---')
scoreEstimate = -model.best_score_
print(scoreEstimate)

dtype = [('index','int32'), ('y','float32'), ('y_pred_train','float32'), ('y_pred_test','float32')]
values = np.zeros([merged_df.shape[0]], dtype=dtype)
index = range(merged_df.shape[0])
results_df = pd.DataFrame(values, index=index)
results_df['y'] = np.append(y_train, y_test)
code.interact(local=locals())
results_df[range(n_train), 'y_pred_train'] = y_pred_train
results_df[range(n_train, merged_df.shape[0]), 'y_pred_test'] = y_pred_test

results_df.write_csv('prediction/results_svr.csv')

plt.plot(range(n_train), y_pred_train, label='y_pred_train')
plt.plot(range(n_train, merged_df.shape[0]), y_pred_test, label='y_pred_test')
plt.plot(np.append(y_train, y_test), label='y_test')
plt.title('Recurrent SVR with 30% test')

# plt.plot(y_pred, label='y_pred')
# plt.plot(y, label='y')
plt.legend()
plt.show()
# code.interact(local=locals())
