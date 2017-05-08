# Libraries
import pandas as pd
from matplotlib import pyplot as plt

# Code
trends = pd.read_csv('clean_data/google_trends/kw1_cat0_2007-01-01-2017-04-01.csv')
plt.plot(trends['greece'].values)
plt.show()