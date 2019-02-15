import requests
import json
import re
import pandas as pd
import numpy as np
import random
import csv
import datetime
import os

def get_financial_data(stock, start_date):
    '''
    Get financial data for given company from ih.advfn.com from start day up to today.
    Saves csv of the data as well. 
    params: stock - stock symbol of company
            start_date - from this date up to today, fetch the data
    returns: financial_data - pandas dataframe of the financial data of the chosen stock
    '''
    # get number of quartals from start date
    today = datetime.date.today()
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    qrts = int(np.ceil(((today.year - start_date.year) * 12 + (today.month - start_date.month))/3))

    # get available quarterly report dates
    exchange = 'NASDAQ'
    result_dict = {}
    url = 'https://ih.advfn.com/stock-market/{0}/{1}/financials?btn=istart_date&istart_date=1&mode=quarterly_reports'.format(exchange,stock)
    response = requests.get(url)
    html_content = str(response.content)

    # get report dates since start_date
    report_date_dict = {}
    report_dates_re = r"n<option  value=\\\'(?P<VALUE>\d{1,3})\\\'>(?P<YEAR>\d{4})/(?P<QRT>\d{2})</option>"
    matches = re.findall(report_dates_re, html_content)

    try:
        report_dates = matches[-qrts:] # only last x report dates
    except:
        report_dates = matches

    for date in report_dates:
        report_date_dict[int(date[0])] = datetime.date(int(date[1]), int(date[2]), 1)
    date_values = list(report_date_dict.keys())
    
    # add data to result dict 
    for date in date_values:
        # read html quarterly report for date
        url = 'https://ih.advfn.com/stock-market/{0}/{1}/financials?btn=istart_date&istart_date={2}&mode=quarterly_reports'.format(exchange,stock,date)
        response = requests.get(url)
        html_content = str(response.content)
        
        # get indicators and values for report date
        indicators_re = r"<td class=\\\'sb?\\\' width=\\\'200\\\' align=\\\'left\\\'>(?P<INDICATOR>[%-/,\)\( \w]+)</td><td class=\\\'sb?\\\' align=\\\'right\\\'>(?P<VALUE>[,0-9\.]+)</td>"
        indicators = re.findall(indicators_re, html_content)
        
        date_str = report_date_dict[date].strftime('%Y-%m-%d')
        result_dict[date_str] = {}
        
        for i in indicators:
            result_dict[date_str][i[0].lower()]= float(i[1].replace(',',''))
    
    # save and return dataframe
    financial_data = pd.DataFrame.from_dict(result_dict, orient='index')
    
    directory = './datafiles/{}/'.format(stock)
    try:
        os.makedirs(directory)
        financial_data.to_csv(directory + stock + '.csv')
    except:
        financial_data.to_csv(directory + stock + '.csv')

    return financial_data