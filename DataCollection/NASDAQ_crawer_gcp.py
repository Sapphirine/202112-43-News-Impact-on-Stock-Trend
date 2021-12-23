#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import yfinance as yf
import random
import json
import requests
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import time
import os
import datetime
import argparse 

headers = {
    "Host": "api.nasdaq.com",
    #"Connection": "keep-alive",
    'sec-ch-ua': '"Microsoft Edge";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    "Accept": "application/json, text/plain, */*",
    "sec-ch-ua-mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.44",
    "sec-ch-ua-platform": '"Windows"',
    "Origin": "https://www.nasdaq.com",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://www.nasdaq.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6"
}

headers2 = {
    "Host": "www.nasdaq.com",
    'sec-ch-ua': '"Google Chromium";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    "Accept": "*/*",
    "sec-ch-ua-mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
    "sec-ch-ua-platform": '"Windows"',
    "Origin": "https://www.nasdaq.com",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://www.nasdaq.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9"
}

http_proxy  = "18.224.59.63:3128"

proxyDict = {
              "http"  : http_proxy,
              "https" : http_proxy,
              "ftp"   : http_proxy
            }

stock_col = ["Open","High","Low","Close","Adj Close","Volume"]

TODAY = datetime.datetime.now()

def save_file(data,stock,i):
    if not os.path.exists('gs://xxx/news_data/'):
        os.makedirs('gs://xxx/news_data/')
    print("saving {} news to file...".format(stock))
    file_dir = 'gs://xxx/news_data/'+stock+'_'+str(i)+'.csv'
    data.to_csv(file_dir,index = False, encoding='utf_8_sig', header = True)
    return

def impact_index(factor,d = [1,1,0.2],p = 2):
    res = 0
    for n,i in enumerate(factor[:-1]):
        res += np.sign(i)*(abs(i*100)**p)*d[n]
    decay = (factor[-1]+1)/2
    if decay > 1:
        delta = (decay-1)/10
        decay = 1 + delta
    return res*decay/3

def get_stock_data(data,date):
    #data_np = data.value()
    base_date = business_day_cal(date,-1)
    fut_0 = business_day_cal(date,0)
    fut_5 = business_day_cal(date,4)
    fut_30 = business_day_cal(date,29)

    if base_date not in data.index:
        return None
    if fut_0 not in data.index:
        return None
    if fut_5 not in data.index:
        return None
    if fut_30 not in data.index:
        return None

    res = []
    # 1,5,30 percentage

    res.append((data.loc[fut_0,'Close']-data.loc[base_date,'Close'])/data.loc[base_date,'Close'])
    res.append((data.loc[fut_5,'Close']-data.loc[base_date,'Close'])/data.loc[base_date,'Close'])
    res.append((data.loc[fut_30,'Close']-data.loc[base_date,'Close'])/data.loc[base_date,'Close']    )


    # volumn 5 day percnetage
    curr_r = data.index.get_loc(fut_0)

    vol_pas_5 = data.iloc[curr_r-5:curr_r].sum(axis =0)["Volume"]
    vol_fut_5 = data.iloc[curr_r:curr_r+5].sum(axis =0)["Volume"]
    res.append((vol_fut_5 - vol_pas_5)/vol_pas_5)

    res.append(impact_index(res))
    return res


def business_day_cal(date,x):
    US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    if x == 0:
        return ((date - 1 * US_BUSINESS_DAY) + 1* US_BUSINESS_DAY ).normalize()
    return (date + x * US_BUSINESS_DAY).normalize()

def get_info(url,stock_data):
    response = requests.get(url,headers = headers2)

    #parse
    soup = BeautifulSoup(response.text,'lxml')
    #get imformation
    try:
        news_date = datetime.datetime.strptime(soup.find('time',attrs = {'class':'timestamp__date'}).text[:-4],"%b %d, %Y %H:%M%p")
    except:
        return None

    #earilest = business_day_cal(TODAY, -30)
    #if (earilest - news_date).days <= 0:
    #    return None

    news_text = soup.find('div',attrs = {'class': 'body__content'}).text.replace('\n','')
    news_title = soup.find(attrs = {'name': 'title'})['content']
    related_stocks = soup.find(attrs = {'name':'com.nasdaq.cms.taxonomy.quoteSymbol'})['content']
    if news_title.startswith("Pre-Market Most Active"):
        return None

    # get stock data for training
    #stock_factor = get_stock_data(stock_data,news_date)
    
    # in the original setting, this function supposes to get the future performance of stocks at the same time 
    # but now we deal with the stock data saperately.

    #combine all info into a list of columns
    columns = [news_text, news_title, news_date, related_stocks, url]
    #give columns names
    column_names = ['news','title','date','related','url']
    return dict(zip(column_names,columns))

def get_news(stock_list,amount = 10000):
    info_allstock = {}
    for stock in stock_list:
        links = []

        stock_data = yf.download(stock, period = "max", interval = "1d")
        stock_data = stock_data.dropna()

        url = "https://api.nasdaq.com/api/news/topic/articlebysymbol?q={}|stocks&offset=0&limit={}&fallback=false".format(stock,amount)

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text,'lxml')
        data = json.loads(soup.p.text)

        for row in data['data']['rows']:
            links.append(r'https://www.nasdaq.com/'+row['url'])

        batch = 100
        num_batches = len(links) // batch + 1

        for i in tqdm(range(num_batches)):
            info = []
            for u in links[i*batch:(i+1)*batch]:
                scraped = get_info(u,stock_data)
                if scraped:
                    info.append(scraped)
                time.sleep(1)
            if info:
                info = pd.DataFrame(info)
                save_file(info,stock,i)
    return
parser = argparse.ArgumentParser(description='Scraped stock news on GCP.')
parser.add_argument('--tickers_list', type=str, required = True, help='stock tickers you wish to download in lower case, e.g. aapl,tsla,amzn')
args = parser.parse_args()
stock_list = args.tickers_list.split(',')
get_news(stock_list)
print("finished!")

