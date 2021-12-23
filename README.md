# News Impact on Stock Trend
## EECS 6893: Final Project, Fall 2021


[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)]()



This is the repository of source code for our final project Big of EECS E6893: Big Data Analytics.

This repository contains:

1. The code to scrape news and generate data for the following 2 task. 
2. News sentiment classification and regression experiments.
3. Web app and force directed graph to visualize .

## Table of Contents

- [Background](#background)
- [Data Collection](#data-collection)
    - [Install](#install-1)
    - [Usage](#usage-2)
- [Data Analyzing](#data-analyzing)
    - [Install](#install-2)
    - [Usage](#usage-2)
- [Visualization](#visualization)
    - [Install](#install-3)
    - [Usage](#usage-3)
- [Others](#others)
- [Authors](#authors)

## Background

**In this project, we try to achieve the following two goals:**
>1. Using news data to help predict the future performance of stocks.
>2. Visualizing the news topics and sentiments distribution with the stock price.

This project intends help traders and investors make decision immediate when news happens and creates a research platform for researcher, analysts and consultants to qucikly learn about the history of the stock.    

**The whole project can be divided into three parts:**
>1. Data Collection: Spiders for scraping news data from NASDAQ and codes to generater data set used in the following task
>2. Data Analyzing: Direct regression to forecast the stock price and trend classification using news sentiment with advanced NLP models
>3. Visualization: Our web app and directed forced graph


## Data Collection
### Install

This project uses [yfinance](https://pypi.org/project/yfinance/) and [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#) to collect stock price data and news data. Go check them out if you don't have them locally installed.

```sh
$ cd DataCollection
$ pip install -r requirements.txt
```

### Usage
We strongly suggest you to use [NASDAQ_crawer_gcp.py](NASDAQ_crawer_gcp.py) to scrape news data from NASDAQ website on [GCP](https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1009892&utm_content=text-ad-none-any-DEV_c-CRE_491349594127-ADGP_Desk%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20Google%20Cloud%20Platform%20Core-KWID_43700060017921809-kwd-87853815&utm_term=KW_gcp-ST_gcp&gclid=CjwKCAiAtouOBhA6EiwA2nLKH59b5SmglmYbXWUS7LDXthhKeRssZP42aF3l2c_aieWNQBSH1ydtjhoCrP0QAvD_BwE&gclsrc=aw.ds) and save the scaped data to bucket just as we did. Scraping the data to local machine will be slow and has potential to be **retricted** from visiting the website. 

```
$ python crawers.py --tickers_list=aapl,tsla,baba,fb
# or
$ gcloud dataproc jobs submit pyspark --cluster <clustername> gs://<bucketname>/NASDAQ_crawer_gcp.py --region=<region> -- --tickers_list=aapl,tsla,baba,fb
```

Note that download the scraped news data from bucket will require Google Cloud Storage library and credentials. Please refer to [GCS document](https://cloud.google.com/docs/authentication/getting-started) for more information.

The stock data set can be generated by running [getstock.py](getstock.py) which will calculate all the factors we used in the experiments.


Please check [VisualizationData](DataCollection/VisualizationData) folder for more codes used to process the data for visualization.


## Data Analyzing

### Install
To run T5 and Bert model on news sentiment classification, go to director of NLPExperiment and install all the required package
```
$ cd DataProcessing/NLPExperiment
$ pip install -r requirements.txt
```

### Usage

You can use [news_process.ipynb](DataAnalyzing/NLPExperiment/news_process.ipynb) to run T5 and Bert models on news sentiment classification tasks.  

To get dataset of NLP experiments, run [NLPdataset.py](DataCollection/NLPdataset.py) .

You can use [regression.ipynb](DataAnalyzing/RegressionExperiment/regression.ipynb) to test more regression results on different features and stocks.

## Visualization
### Install
To run the web server,  django framework need to be downloaded
```sh
$ pip install django
```

### Usage
After installation, cd to folder [TREND](Visualization/TREND) and run the web server.
```sh
$ cd visualization/TREND/
$ python manage.py runserver
```
Then open web browser  and enter http://localhost:8000/cover to see the homepage of our website

Here is one screen shot of our web app: 
![Alt text](https://github.com/Sapphirine/202112-43-News-Impact-on-Stock-Trend/blob/main/readmeImage/web_demo.png)


## Others
To know more details about our project, please check our youtube link: https://youtu.be/dYse7Cuao4Y



## Authors
* Haoming Sun
* Yukun Huang
* Xintong Liu 




