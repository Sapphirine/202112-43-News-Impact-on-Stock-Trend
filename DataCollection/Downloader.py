#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from google.cloud import storage


# In[ ]:


# this is the code to download news data from google cloud bucket
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def download_dataset(stock):
    i = 0
    if not os.path.exists("./temp/"):
        os.mkdir("./temp/")
    while i<=200:
        try:
            download_blob(" ","news_data/{}_{}.csv".format(stock,i),"./temp/{}_{}.csv".format(stock,i))
        except:
            pass
        i+=1
def merge_dataset(stock,i):
    # i is the start number in ./test/tickers_i.csv
    df = pd.read_csv("./temp/{}_{}.csv".format(stock,i),encoding='utf_8_sig')
    while i:
        try:
            temp = pd.read_csv("./temp/{}_{}.csv".format(stock,i),encoding='utf_8_sig')
            df = pd.concat([df,temp],axis = 0,ignore_index=True)
        except:
            break
        i+=1
    #old = pd.read_csv('./news_data/'+stock+'.csv',encoding='utf_8_sig')
    #df = pd.concat([df,old],axis = 0,ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.to_csv('./news_data/'+stock+'.csv',index = False, encoding='utf_8_sig',header = True)

download_dataset('aapl')
merge_dataset('aapl',0) 


# In[ ]:




