#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm


# In[2]:


for stock in ['tal','baba','googl','tsla','aapl','dis','fb','nflx','nvda','uber','se','amzn']:
    his = pd.read_csv('./stock_data_new/'+stock+'.csv',index_col='Date')
    news = pd.read_csv('./news_data/'+stock+'.csv',encoding='utf_8_sig')
    news['date'] = news['date'].apply(lambda x:x[:10])
    news = news.sort_values(by ='date',ignore_index=True)
    for i,row in tqdm(news.iterrows()):
        day = row['date']
        try:
            temp = his.loc[day]
        except:
            if i-1<0:
                continue
            news.at[i,'date'] = news.iloc[i-1]['date']
            day = news.iloc[i-1]['date']
            try:
                temp = his.loc[day]
            except:
                continue
        for col in his.columns.values.tolist():
            news.at[i,col]=his.at[day,col]

    news.to_csv('./news_stock/'+stock+'.csv')


# In[ ]:




