import numpy as np
import pandas as pd
import datetime
import os
import re
import random
import csv
import json

# APIs and data requests
import alpha_vantage
import quandl
import requests

# scrapers
import economic_data_scraper
import financial_data_scraper
import price_data_scraper



class Data(object):
    '''data object holding all relevant data of a single company'''

    def __init__(self, stock, start_date):
        self.stock = stock  # stock symbol String
        self.start_date = start_date  # data - start Date
        # try to read already processed dataframe
        try:
            self.read_processed_data()
            print('read processed data')
        except: 
            # if no processed data available read unprocessed data, and get price data from web
            try:
                self.read_unprocessed_data()
                print('read unprocessed data')
            except:
                # if no unprocessed data available read all data from web
                self.scrape_all_data() 
                print('read all data from web')
            # preprocess unprocessed dataframe
            self.preprocess()
        
        # save dataframe to csv
        directory = './datafiles/{}/'.format(self.stock)
        try:
            self.data_frame.to_csv(directory + self.stock + '_dataset.csv')
        except:
            print('Could not save processed dataset!')

    def date_preprocessing(self, df):
        '''dataframe index to date and sort by date'''
        df.index = pd.to_datetime(df.index, yearfirst=True)
        df.sort_index(inplace=True)

    def preprocess(self):
        '''preprocess dataframe: filtering and interpolation'''
        economic_indicators = ['gdp', 'interest_rates', 'unemployment', 'inflation']
        financial_indicators = ['ebit', 'total current assets', 'total current liabilities', 'net cash from total operating activities']
        price_indicators = ['price', 's&p500']
        self.indicator_list = [item for sublist in [price_indicators, financial_indicators, economic_indicators] for item in sublist]

        self.data_frame = self.data_frame.interpolate(method='linear')
        self.data_frame = np.round(self.data_frame[self.indicator_list], decimals=2)
        self.data_frame = self.data_frame.dropna()

    def read_processed_data(self):
        '''read already processed dataframe and check if it covers desired timeframe'''
        self.data_frame = pd.read_csv('./datafiles/{0}/{0}_dataset.csv'.format(self.stock), index_col=0)
        self.date_preprocessing(self.data_frame)
        # check if all data desired is in processed data
        start_date_with_tolerance = pd.to_datetime(self.start_date) + datetime.timedelta(days=5)
        if (pd.to_datetime(self.data_frame.index[0]) > start_date_with_tolerance) or (pd.to_datetime(self.data_frame.index[-1]) < pd.to_datetime(datetime.date.today())):
            raise ValueError('processed data does not cover desired timespan')
        # self.data_frame = self.data_frame.loc[self.start_date:]

    def read_unprocessed_data(self):
        '''read unprocessed data from files, get price from web'''
        # read economic data
        economics = pd.read_csv('./datafiles/economic_data.csv', index_col=0)    # read economic data USA DataFrame from csv
        self.date_preprocessing(economics)
        if (economics.index[0] > pd.to_datetime(self.start_date)):
            raise ValueError('not all economic data available')
        
        # read financial data
        finances = pd.read_csv('./datafiles/{0}/{0}.csv'.format(self.stock), index_col=0)    # read companies financial data DataFrame from csv  
        self.date_preprocessing(finances)
        if (finances.index[0] > pd.to_datetime(self.start_date)):  # check if the already saved data covers start date 
            raise ValueError('not all financial data available')
        
        # get price data
        prices = price_data_scraper.get_price_data(self.stock, self.start_date)
        self.date_preprocessing(prices)
        # self.sentiment = Sentiment_Data_Scraper.get_sentiment_data(self.stock, self.start_date)

        # construct and return dataframe
        self.data_frame = pd.concat([economics, finances, prices], axis=1, sort=False)

    def scrape_all_data(self):
        '''scrape all data from web'''
        # get economic data
        economics = economic_data_scraper.get_economics_data(self.start_date)
        self.date_preprocessing(economics)

        # get financial data
        finances = financial_data_scraper.get_financial_data(self.stock, self.start_date)
        self.date_preprocessing(finances)

        # get price data
        prices = price_data_scraper.get_price_data(self.stock, self.start_date)
        self.date_preprocessing(prices)
        # self.sentiment = Sentiment_Data_Scraper.get_sentiment_data(self.stock, self.start_date)

        # construct and return dataframe
        self.data_frame = pd.concat([economics, finances, prices], axis=1, sort=False)
