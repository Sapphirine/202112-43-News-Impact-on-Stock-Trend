#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from get_all_tickers_new import get_tickers_filtered
from tqdm import tqdm
import time
import numpy as np
import json
import collections


# In[10]:


# news connection for force-directed graph
section_dict = {'Consumer Non-Durables':0,'Capital Goods':1,'Health Care':2,'Energy':3,'Technology':4,'Basic Industries':5,
                'Finance':6,'Consumer Services':7,'Public Utilities':8,'Consumer Durables':9,'Transportation':10}
stock_section_dict= dict()

for se in section_dict.keys():
    for stock in get_tickers_filtered(sectors = se):
        stock_section_dict[stock]=section_dict[se]

over_all_dict = dict()
over_all_dict['nodes'] = []
over_all_dict['links'] = []
ticker_times = collections.defaultdict(int)
ticker_class = dict()
edges_times = collections.defaultdict(int)
for file_dir in os.listdir('./news_data'):
    news_data = pd.read_csv('./news_data/'+file_dir)
    for i,row in news_data.iterrows():
        related_stocks = set(row['related'].split(','))
        for i in related_stocks.copy():
            if i not in stock_section_dict.keys():
                stock_section_dict[i] =  11
        related_stocks = list(related_stocks)
            
        for i in range(len(related_stocks)):
            ticker_times[related_stocks[i]] += 1
            ticker_class[related_stocks[i]] = stock_section_dict[related_stocks[i]]
            for j in range(i,len(related_stocks)):
                if i != j:
                    temp_set = tuple(sorted([related_stocks[i],related_stocks[j]]))
                    edges_times[temp_set] += 1 

max_time =0
max_wides = 0
for s in ticker_times.values():
    max_time = max(max_time,s**0.5)
for x in edges_times.values():
    max_wides = max(max_wides,x**0.5)
node_set = set()
for (x,y) in edges_times.keys():
    temp = dict()
    temp['source'] = x
    temp['target'] = y
    temp['value'] = 10*(edges_times[(x,y)]**0.5/max_wides)
    if  temp['value'] <=1:
        continue
    node_set.add(x)
    node_set.add(y)
    over_all_dict['links'].append(temp)
    
for s in node_set:
    temp = dict()
    temp['id'] = s
    temp['group'] = ticker_class[s]
    temp['times'] = 40*(ticker_times[s]**0.5/max_time)
    over_all_dict['nodes'].append(temp)

filename='./ForceDirectedGraph/miserables.json'
with open(filename,'w') as file_obj:
    json.dump(over_all_dict,file_obj)


# In[ ]:




