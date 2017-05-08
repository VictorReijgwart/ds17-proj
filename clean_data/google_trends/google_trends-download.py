### Automatically download and store Google Trends data

## Import libraries
import os
import getpass
from pytrends.request import TrendReq
import numpy as np
import datetime as dt
import pandas as pd
import code


## Settings
# Search terms of interest
keyword_set_id = 1 # Set this manually to easily distinguish saved files
keyword_list = ['greece','greece bailout','greece debt','greece crisis','greece referendum']
# keyword_list = ['bailout','debt','crisis','referendum','euro dollar']
# keyword_list = ['grexit']

# Date range
start_date = dt.date(2007,1,1)
end_date = dt.date(2017,4,1)

# Filter by search type (google product):
google_product='' # options: ['news', 'images', ...]

# Filter by searches originating from geo
geo=''

# Filter by searches with specific category
category=0 # ['Any': 0, 'Finance': 7, 'Business & Industrial': 12, 'Law & Gov': 19, ...]

# Storage directory
dst_dir = os.path.join('features','google_trends')
dst_filename = 'kw{}_cat{}_{}-{}.csv'.format(keyword_set_id, category, start_date, end_date)
dst_file = os.path.join(dst_dir, dst_filename)


## Request Google credentials for use in this session
print('Google authentification is required to download trend data.')
print('Please provide credentials to use in this session (won''t be stored):')
google_username = 'victorreijgwart'#raw_input()
google_password = getpass.getpass()


## Login and init
pytrends = TrendReq(google_username, google_password, hl='en-US', custom_useragent=None)


## Loop to pull over entire range
interest_timeseries = pd.DataFrame()
date = start_date - dt.timedelta(days=1)
for i in range(int(np.ceil((end_date-start_date).days/89)+1)):
    date = date + dt.timedelta(days=89)
    timeframe = '{} {}'.format(date - dt.timedelta(89), date if date < end_date else end_date)
    print('Loading trend data for: '+ str(timeframe))
    pytrends.build_payload(kw_list=keyword_list, timeframe=timeframe)#, cat=category, geo=geo, gprop=google_product)

    # Remove normalization
    if i == 0:
        print i
        # interest_timeseries

    interest_timeseries = pd.concat([interest_timeseries, pytrends.interest_over_time()])

## Save to file
print('Saving data to: '+dst_file)
interest_timeseries.to_csv(dst_file)

# code.interact(local=locals())
