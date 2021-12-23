#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import pandas as pd
import os
from get_all_tickers import get_tickers as gt
from tqdm import tqdm
import time
import numpy as np
import datetime
import random
import json
import collections
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re


# In[ ]:


# new json for K line
topic_list=['policy','industry','market','investment','business']
def get_topic(news):
    count = collections.Counter(news)
    curr_max = 0
    topic = None
    for t in topic_list:
        if t == 'market':
            count[t] = count[t]/2
        if count[t]>curr_max:
            curr_max = count[t]
            topic = t
    return topic if topic else 'else'

lancaster=LancasterStemmer()

def stemSentence(news):
    stem_sentence=[]
    for word in news:
        stem_sentence.append(lancaster.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def get_impact(news):
    score = analyser.polarity_scores(stemSentence(news))
    if score['neu'] > 0.9:
        return 'neutral', score['neu']*len(news)/100
    else:
        if score['pos'] > score['neg']:
            return 'positive', score['pos']*len(news)/100
        else:
            return 'negative', score['neg']*len(news)/100
    
    
    
nltk.download('vader_lexicon')
regEx = re.compile('\W')
new_words =  {'falls': -9, 'drops': -9, 'rise': 9, 'increases': 9, 'gain': 9, 'hiked': -9, 'dips': -9, 'declines': -9, 'decline': -9, 'hikes': -9, 'jumps': 9,
                  'lose': -9, 'profit': 9, 'loss': -9, 'shreds': -9, 'sell': -9, 'buy': 9, 'recession': -9, 'rupee weakens': -9, 'record low': -9, 'record high': 9,
                  'sensex up': 9, 'down': -9, 'sensex down': -9, 'up': 9,'forbidden':-9,'crackdown':-9,'tougher':-9,'bans':-9,'restrictions':-9,'ban':-9,'restriction':-9,
                  'stricter':-9, 'worst':-9,'deprive':-9, 'hardly':-9,'limiting':-9,'enacting':-9,'fear':-9,'little':-9,'resist':-9,'contrarian':-9,'cold':-9,
               'danger':-9,'lurks':-9,'fining':-9,'decimate':-9} 

analyser = SentimentIntensityAnalyzer()
analyser.lexicon.update(new_words)
        
whole_dict=dict()
for stock in ['amzn','tal','baba','googl','tsla','aapl','dis','fb','nflx','nvda','uber','se']:
    stock_dict = dict()
    stock_data = yf.download(stock,period = 'max',interval ='1d')
    stock_data.sort_index(ascending=True,inplace=True)
    news_data = pd.read_csv('./news_data/'+stock+'.csv')
    news_data['date'] = pd.to_datetime(news_data['date'])
    news_data.sort_values(by = ['date'],ascending=True,inplace=True)
    news_data.reset_index(drop=True,inplace=True)
    earilest_day = news_data['date'][0]
    
    for i in tqdm(range(news_data.shape[0])):
        news_text = word_tokenize(news_data.iloc[i]['news'])
        news_data.at[i,'topic'] = get_topic(news_text)
        senti,impact = get_impact(news_text)
        news_data.at[i,'impact'] = impact
        news_data.at[i,'sentiment'] = senti
    
    i = 0
    for rows in tqdm(stock_data.itertuples()):
        if rows[0]<earilest_day:
            continue 
            
        day = rows[0].strftime('%Y-%m-%d')
        stock_dict[day] = dict()
        stock_dict[day]['Open'] = rows[1]
        stock_dict[day]['High'] = rows[2]
        stock_dict[day]['Low'] = rows[3]
        stock_dict[day]['Close'] = rows[4]
        stock_dict[day]['News_list'] = []
        
        while i < news_data.shape[0] and rows[0]-news_data.iloc[i]['date']>datetime.timedelta(10):
            i+=1
        
        j = i
        while j< news_data.shape[0] and rows[0]>=news_data.iloc[j]['date'] and  rows[0]-news_data.iloc[j]['date']<=datetime.timedelta(10):
            news_dict = dict()
            news_dict['title']=news_data.iloc[j]['title']
            news_dict['url']=news_data.iloc[j]['url']
            news_dict['date'] = news_data.iloc[j]['date'].strftime('%Y/%m/%d %H:%M')
            news_dict['related'] = news_data.iloc[j]['related']
            #news_text = word_tokenize(news_data.iloc[j]['news'])
            news_dict['topic'] = news_data.iloc[j]['topic']
            #senti,impact = get_impact(news_text)
            news_dict['impact'] = news_data.iloc[j]['impact']
            news_dict['sentiment'] = news_data.iloc[j]['sentiment']
            stock_dict[day]['News_list'].append(news_dict)
            j+=1
    whole_dict[stock] = stock_dict


filename='Kline.json'
with open(filename,'w') as file_obj:
    json.dump(whole_dict,file_obj)

    

