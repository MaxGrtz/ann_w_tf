import numpy as np
import pandas as pd
import datetime
import os
import csv
import json

# APIs and data requests
import alpha_vantage
import requests

def get_price_data(stock, start_date):

    API_URL = "https://www.alphavantage.co/query"

    # api call config for chosen stock
    stock_data = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock,
        "outputsize": "full",
        "datatype": "json",
        "apikey": "E8X1JONC18APVAQW"
        }

    # get data from alphavantage
    response = requests.get(API_URL, stock_data)
    data = response.json()
    metaData = data.pop("Meta Data", None)
    stock_prices = pd.DataFrame.from_dict(data, orient="columns")
    stock_prices = stock_prices["Time Series (Daily)"].apply(pd.Series).apply(pd.to_numeric)
    volume = stock_prices["6. volume"].to_frame('volume') # safe volume as time series df
    stock_prices = stock_prices["5. adjusted close"].to_frame('price') # save closing prices as series over dates as index

    stock_prices = stock_prices.loc[start_date:]
    
    # api call config for S&P 500 prices
    s_p_data = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": "^GSPC",
        "outputsize": "full",
        "datatype": "json",
        "apikey": "E8X1JONC18APVAQW"
        }
    
    response = requests.get(API_URL, s_p_data)
    data = response.json()
    metaData = data.pop("Meta Data", None)
    s_p_prices = pd.DataFrame.from_dict(data, orient="columns")
    s_p_prices = s_p_prices["Time Series (Daily)"].apply(pd.Series).apply(pd.to_numeric)
    s_p_prices = s_p_prices["5. adjusted close"].to_frame('s&p500') # save closing prices as series over dates as index
    s_p_prices = s_p_prices.loc[start_date:]
    
    return pd.concat([stock_prices,volume,s_p_prices], axis=1, sort=False)