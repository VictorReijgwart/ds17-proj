
# coding: utf-8

# In[315]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[294]:

from pybrain.datasets import SequentialDataSet

def create_data(dataframe, inputs, targets):
    data = SequentialDataSet(inputs, targets)
    for i in range(0, dataframe.shape[0]-1):
        row = dataframe.iloc[i]
        data.newSequence()
        ins = row.values
        target =dataframe.iloc[i + 1].values[0]
        data.appendLinked(ins, target)
    return data


# In[316]:

import matplotlib
import pandas as pan
import datahandler as dh
import nethandler as nh
from matplotlib import pyplot as pp
import numpy as np
#Input Data
TRAINING_PERCENT = 0.50
LAG_DAYS = 1
startdate = '20000101'  # YYYYMMDD
indices = ["SQLDATE", "c_norm", "c_norm_lag1"]

#Neural Network
INPUT = len(indices) * (LAG_DAYS)
HIDDEN = 12
OUTPUT = 1

#Training
ITERATIONS = 20
LRATE = 0.4
MOMENTUM = 0.6



# filename_rates = '../clean_data/euro_exchange.csv'
filename_rates = '../clean_data/log_log_exchange_rate_all_crisis.csv'
data_euro_rate = pan.read_csv(filename_rates,usecols=["SQLDATE","S_Log_Exchange"])
# data_euro_rate['Date'] =  pan.to_datetime(data_euro_rate['Date']).dt.strftime('%Y%m%d')
data_euro_rate = data_euro_rate.set_index('SQLDATE')

filename_greece = '../features/GDELT_I/greece.csv'
data_greece = pan.read_csv(filename_greece,usecols=indices,index_col='SQLDATE')
data_greece =  data_greece.loc[ data_euro_rate.iloc[0].name: data_euro_rate.iloc[-1].name]


data_euro_rate['S_Log_Exchange'] =  np.exp(data_euro_rate['S_Log_Exchange'])

data_euro_rate.index =  data_greece.index

data =  pan.concat([data_euro_rate, data_greece],axis = 1)

data_full = create_data(data,INPUT,OUTPUT)

print data_full
train, test =  data_full.splitWithProportion(0.1)
# data_euro_rate.loc[mask]
# print data_euro_rate.head
# data.create_data(INPUT, OUTPUT)
# train, test = data.get_datasets(TRAINING_PERCENT)
# print "Training:", len(train), "Testing:", len(test)


# In[302]:


sp_net = nh.NetHandler(INPUT, HIDDEN, OUTPUT, data_full)
train_errors, val_errors = sp_net.train(data_full, LRATE, MOMENTUM, ITERATIONS)

# out_ser = sp_net.get_output(test, TRAINING_PERCENT)
# print "Net Topology: %d-%d-%d" % (INPUT, HIDDEN, OUTPUT)
# print sp_net.change_tomorrow()

# correct = 0
# total = 0
# misses = 0
# for index, row in out_ser.iteritems():
#     try:
#         actual = data.dataframe.ix[:, 0][index]
#         total += 1
#         if row > 0 and actual > 0:
#             correct += 1
#         elif row < 0 and actual < 0:
#             correct += 1
#     except KeyError:
#         misses += 1
# print "%.3f%% Directional Accuracy" % (float(correct) / float(total) * 100)
# print "(%d misses)" % misses



# In[ ]:

pp.figure(0)
data.dataframe.ix[:, 0].plot(style='bo-', alpha=0.8)
data.dataframe.ix[:, (LAG_DAYS+1) * 1].plot(style='g-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 2].plot(style='y-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 3].plot(style='m-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 4].plot(style='c-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 5].plot(style='-', color='0.75', alpha=0.5)
out_ser.plot(style='ro-')
pp.axhline(0, color='black')

pp.figure(1)
pp.plot(train_errors)
pp.plot(val_errors)
pp.show()

