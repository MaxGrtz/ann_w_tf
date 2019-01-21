import numpy as np
import pandas as pd

# APIs and data requests
import quandl

 
def get_economics_data(start_date):
    quandl.ApiConfig.api_key = "cEnL78yMi4hkJTQ1cLy_"

    # get gdp
    gdp = quandl.get("FRED/GDP", start_date=start_date)
    gdp.columns = ['gdp [USA]']

    # get interest rates
    interest_rates = quandl.get("FRED/IOER-Interest-Rate-on-Excess-Reserves", start_date=start_date)
    interest_rates.columns = ['interest_rates [USA]']

    # get unemployment rates
    unemployment = quandl.get("FRED/UNEMPLOY-Unemployment-Level", start_date=start_date)
    unemployment.columns = ['unemployment [USA]']

    # get inflation rate
    inflation = quandl.get("FRED/FPCPITOTLZGUSA-Inflation-consumer-prices-for-the-United-States", start_date=start_date)
    inflation.columns = ['inflation [USA]']

    # create single dataframe for economic data
    economic_data = pd.concat([gdp, interest_rates, unemployment, inflation], axis=1, sort=False)

    # safe economic data to datafiles
    economic_data.to_csv('./datafiles/economic_data.csv')

    return economic_data