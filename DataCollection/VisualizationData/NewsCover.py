#!/usr/bin/env python
# coding: utf-8

# In[9]:


from bing_image_downloader import bing
import os, sys
import shutil
from pathlib import Path
import urllib.request
import urllib
import imghdr
import posixpath
import re
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import datetime
import json


# In[10]:


# get a picture from Bing as the news cover
class Bing_2(bing.Bing):
    def __init__(self, query, limit, output_dir, adult, timeout,  filters='', verbose=True):
        super().__init__(query, limit, output_dir, adult, timeout,  filters='', verbose=True)
    def run(self):
        while self.download_count < self.limit:
            # Parse the page source and download pics
            request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(self.query)                           + '&first=' + str(self.page_counter) + '&count=' + str(self.limit)                           + '&adlt=' + self.adult + '&qft=' + ('' if self.filters is None else str(self.filters))
            request = urllib.request.Request(request_url, None, headers=self.headers)
            response = urllib.request.urlopen(request)
            html = response.read().decode('utf8')
            if html ==  "":
                #print("[%] No more images are available")
                break
            links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
            #if self.verbose:
                #print("[%] Indexed {} Images on Page {}.".format(len(links), self.page_counter + 1))
                #print("\n===============================================\n")
                        
                
            for link in links:
                if self.download_count < self.limit:
                    self.download_count+=1
                    if 'jpg' in link or 'png' in link:
                        return link
            self.page_counter += 1
        return download(query='Nasdaq', limit=1, output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)

def download(query, limit=100, output_dir='dataset', adult_filter_off=True, 
    force_replace=False, timeout=60, verbose=True):

    # engine = 'bing'
    if adult_filter_off:
        adult = 'off'
    else:
        adult = 'on'

    image_dir = Path(output_dir).joinpath(query).absolute()

    #print("[%] Downloading Images to {}".format(str(image_dir.absolute())))
    bing = Bing_2(query, limit, image_dir, adult, timeout, verbose)
    return bing.run()

whole_dict=dict()
#for stock in ['amzn','tal','baba','googl','tsla','aapl','dis','fb','nflx','nvda','uber','se']:

# Bing constrains the frequence of requests so run one by one
for stock in ['amzn']:
    stock_dict = dict()
    stock_data = yf.download(stock,period = 'max',interval ='1d')
    stock_data.sort_index(ascending=True,inplace=True)
    news_data = pd.read_csv('./news_data/'+stock+'.csv')
    news_data['date'] = pd.to_datetime(news_data['date'])
    news_data.sort_values(by = ['date'],ascending=True,inplace=True)
    news_data.reset_index(drop=True,inplace=True)
    earilest_day = news_data['date'][0]        
    
    i = 0
    for rows in tqdm(stock_data.itertuples()):
        if rows[0]<earilest_day:
            continue 
            
        day = rows[0].strftime('%Y-%m-%d')
        stock_dict[day] = []

        
        while i < news_data.shape[0] and rows[0]-news_data.iloc[i]['date']>datetime.timedelta(10):
            i+=1
        
        j = i
        
        while j< news_data.shape[0] and rows[0]>=news_data.iloc[j]['date'] and  rows[0]-news_data.iloc[j]['date']<=datetime.timedelta(10):
            news_dict = dict()
            news_dict['title']=news_data.iloc[j]['title']
            news_dict['url']=news_data.iloc[j]['url']
            if len(stock_dict[day])>=4:
                stock_dict[day].pop(0)
            stock_dict[day].append(news_dict)
            j+=1
        for i in range(len(stock_dict[day])):
            q = stock_dict[day][i]['title']
            if ':' in q:
                q = q.split(':')[1]
            stock_dict[day][i]['image'] = download(query=q, limit=5, output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    
    #whole_dict[stock] = stock_dict


    filename='./picture/'+stock+'.json'
    with open(filename,'w') as file_obj:
        json.dump(stock_dict,file_obj)


# In[ ]:




