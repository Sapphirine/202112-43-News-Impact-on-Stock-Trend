from django.http import HttpResponse
from django.shortcuts import render
import pandas_gbq
from google.oauth2 import service_account
import json
import sys
import os

# Make sure you have installed pandas-gbq at first;
# You can use the other way to query BigQuery.
# please have a look at
# https://cloud.google.com/bigquery/docs/reference/libraries#client-libraries-install-nodejs
# To get your credential



def base(request):
    context = {}
    context['content1'] = 'Hello World!'
    # Reading data back
    with open('static/data/stocks_data.json', 'r') as f:
        stocks_data = json.load(f)
    with open('static/data/stocks_news.json', 'r') as f:
        stocks_news = json.load(f)
    with open('static/data/stocks_images.json', 'r') as f:
        stocks_images = json.load(f)

    return render(request, 'base.html', {'stocks_data':stocks_data,
    'stocks_news':stocks_news, 'stocks_images':stocks_images})

def cover(request):
    context = {}
    context['content1'] = 'Hello World!'
    return render(request, 'cover.html', context)

def draw(request):
    context = {}
    context['content1'] = 'Hello World!'
    return render(request, 'draw.html', context)


