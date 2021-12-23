#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm
import time
import numpy as np
import datetime
import json
import collections


# In[ ]:


# download stock data and calculater factors for training
def cal_mean(data,r,period,fact):
    return data.iloc[r-period+1:r+1][fact].mean()

def cal_std(data,r,period,fact):
    return data.iloc[r-period+1:r+1][fact].std()

def cal_RVI(data,r,std_length = 21):
    lstd,hstd =[0]*std_length,[0]*std_length
    for i in range(std_length-1,-1,-1):
        if data.iloc[r-i]['day_change']<0:
            lstd.append(data.iloc[r-i]['10_day_Close_std'])
        elif data.iloc[r-i]['day_change']>0:
            hstd.append(data.iloc[r-i]['10_day_Close_std'])
    lsum = sum(lstd)/len(lstd)
    hsum = sum(hstd)/len(hstd)
    return 100 * hsum/(hsum+lsum)

def UpOrDown(data,r,period = 21,avg_line = 21):
    if avg_line == 1:
        try:
            return data.iloc[r+period]['Close'] - data.iloc[r]['Close']
        except:
            return None
    else:
        try:
            return data.iloc[r+period][str(avg_line)+'_day_Close_mean']-data.iloc[r][str(avg_line)+'_day_Close_mean']
        except:
            return None

def avg_value(data,r,period = 21, avg_line = 21):
    if avg_line == 1:
        try:
            return data.iloc[r+period]['Close']
        except:
            return None
    else:
        try:
            return data.iloc[r+period][str(avg_line)+'_day_Close_mean']
        except:
            return None

def avg_value_percentage(data,r,period = 21, avg_line = 21):
    if avg_line == 1:
        try:
            return (data.iloc[r+period]['Close'] - data.iloc[r]['Close'])/data.iloc[r]['Close']
        except:
            return None
    else:
        try:
            return (data.iloc[r+period][str(avg_line)+'_day_Close_mean']-data.iloc[r][str(avg_line)+'_day_Close_mean'])/data.iloc[r][str(avg_line)+'_day_Close_mean']
        except:
            return None

for stock in tqdm(["tal","aapl","amzn","baba","dis","fb","googl","nflx","nvda","se","tsla","uber"]):
    data = yf.download(stock,period = 'max',interval ='1d')
    data.to_csv('test_'+stock+'.csv',header = True)
    data = pd.read_csv('test_'+stock+'.csv')

    for i,row in data.iterrows():
        data.at[i,'inday_change'] = (data.iloc[i]['Close']-data.iloc[i]['Open'])/data.iloc[i]['Open']
        data.at[i,'day_change'] = (data.iloc[i]['Close']-data.iloc[i-1]['Close'])/data.iloc[i-1]['Close']
        data.at[i,'data_amplitude'] = (data.iloc[i]['High']-data.iloc[i-1]['Low'])/data.iloc[i-1]['Close']
        for f in ['High','Low','Close','Open','Adj Close','Volume']:
            data.at[i,'5_day_{}_mean'.format(f)] = cal_mean(data,i,5,f)
            data.at[i,'10_day_{}_mean'.format(f)] = cal_mean(data,i,10,f)
            data.at[i,'21_day_{}_mean'.format(f)] = cal_mean(data,i,21,f)
            data.at[i,'5_day_{}_std'.format(f)] = cal_std(data,i,5,f)
            data.at[i,'10_day_{}_std'.format(f)] = cal_std(data,i,10,f)
            data.at[i,'21_day_{}_std'.format(f)] = cal_std(data,i,21,f)
        data.at[i,'RVI'] = cal_RVI(data,i)
    for i,row in data.iterrows():
        data.at[i,'score_g'] = 0
        data.at[i,'score_n'] = 0
        data.at[i,'score_p'] = 0
        data.at[i,'score_ralative'] = 0

        for avg_line in [1,5,10,21]:
            for period in [1,5,10,21]:
                data.at[i,'after_'+str(period)+'_day_'+str(avg_line)+'_avgline'] = avg_value(data,i,period=period,avg_line=avg_line)
                data.at[i,str(period)+'_day_UpDown_'+str(avg_line)+'_avgline'] = UpOrDown(data,i,period=period,avg_line=avg_line)
                data.at[i,str(period)+'_day_UpDown_percentage_'+str(avg_line)+'_avgline'] = avg_value_percentage(data,i,period=period,avg_line=avg_line)
        
    data.to_csv('./stock_data_new/{}.csv'.format(stock),header = True)       


# In[ ]:


"""
# this is the code to download all the stock data to local from using yfinance which is not suggested
tickers = gt.get_tickers()
for i in range(len(tickers)):
    tickers[i] = tickers[i].strip().replace('/','-')
tickers = list(set(tickers))
total_batches = len(tickers)//100+1
print(total_batches)
for i in range(129):
    temp = tickers[i*100:min((i+1)*100,len(tickers))]
    tickers_batch =  ' '.join(temp)
    data = yf.download(tickers_batch, period = "max", interval = "1d",group_by='tickers')
    print(data)
    input()
    if not os.path.exists('./stock_data/'):
        os.makedirs('./stock_data/')
    print("saving a batch "+str(i) + "...")
            
    for s in temp:
        df = data[s].dropna()
        df.to_csv('./stock_data/'+s+'.csv')
    if (i+1)% 20 == 0:
        print("System asleep...")
        time.sleep(120)
"""


# In[ ]:


"""
# this is the code to upload the stock data to bigquery
import numpy
from google.cloud import bigquery

PROJECT_ID = ''
DATASET_ID = ''
CSV_DIR = './stock_data_new/'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=""

# create a client instance for your project
client = bigquery.Client(project=PROJECT_ID, location="US")
client.create_dataset(DATASET_ID,exists_ok=True)

for file in os.listdir(CSV_DIR):
    file_id = file.split('.')[0]
    client.delete_table(f"{PROJECT_ID}.{DATASET_ID}.{file_id}",not_found_ok=True)
    client.create_table(f"{PROJECT_ID}.{DATASET_ID}.{file_id}",exists_ok=True)
    
    dataset_ref = client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(file_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.autodetect = True
      
    with open(CSV_DIR + file, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config = job_config )
    
    job.result()
     
    print("Loaded {} rows into {}:{}.".format(job.output_rows, DATASET_ID, file_id))
    break"""

