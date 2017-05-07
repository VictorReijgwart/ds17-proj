import os.path
import pandas as pan
# import pandas.io.data as web
import numpy as np
import datetime
from pandas.tools.merge import merge
from pybrain.datasets import SequentialDataSet
import pandas_datareader as web

class DataHandler():
    #Financial Data
    sp = ''
    filename = ''
    tickers = ''
    startdate = ''
    enddate = ''
    dataframe = pan.DataFrame()
    data = ''

    #fetch financial data from file or yahoo API
    def load_indices(self, tickers, startdate, lags,filename):
        self.tickers = tickers
        self.filename = filename
        self.startdate = startdate
        self.enddate = datetime.date.today().strftime("%Y%m%d")
        self.dataframe = pan.read_csv(self.filename,usecols=tickers,index_col='SQLDATE')
        

    def create_data(self, inputs, targets):
        data = SequentialDataSet(inputs, targets)
		for i in range(0, self.dataframe.shape[0]-1):
			row = self.dataframe.irow(i)
            data.newSequence()
            ins = row.values
            target = self.dataframe.ix[i + 1].values[0]
            data.appendLinked(ins, target)
        self.data = data

    def get_datasets(self, proportion):
        return self.data.splitWithProportion(proportion)


def preprocess(vals):
    #log transform to reduce dynamic range and outliers
    outs = []
    for val in vals:
        if val >= 0:
            outs.append(np.log(np.abs(val * 100) + 1))
        else:
            outs.append(-np.log(np.abs(val * 100) + 1))

    #scale to {-0.9,0.9}
    vals_max = np.max(outs)
    vals_min = np.min(outs)
    scale = 1.8 / (vals_max - vals_min)
    for i, val in enumerate(outs):
        outs[i] = (scale * (val - vals_min)) - 0.9

    #mean to 0
    mean = np.mean(outs)
    for i, val in enumerate(outs):
        outs[i] = val - mean

    return np.array(outs)