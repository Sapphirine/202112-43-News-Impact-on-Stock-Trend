#!/usr/bin/env python
# coding: utf-8

# In[2]:


# this is the code to calculate accumutive news amount accroding to tickers
# this is used for visualization
import pandas as pd
import datetime
def func(x):
    ymd = x.split(' ')[0]
    l = ymd.split('-')
    return l[0]+'/'+l[1].rjust(2,'0')+'/'+'01'

s_l = ['aapl','amzn','baba','dis','fb','googl','nflx','nvda','se','tal','tsla','uber']
for stock in s_l:
    data = pd.read_csv('./news_data/'+stock+'.csv')
    data['date'] = data['date'].apply(lambda x:func(x))
    data['count_'] = data.groupby('date')['news'].transform('count')
    data['stock'] = stock
    data = data[['date','count_','stock']]
    data = data.sort_values(by ='date')
    data.drop_duplicates(['date'],inplace = True)
    data.set_index('date',inplace = True)
    data['count'] = data['count_'].cumsum()
    data = data[['count','stock']]
    data.to_csv('news_amount.csv',mode = 'a+',header = False)


# In[ ]:




