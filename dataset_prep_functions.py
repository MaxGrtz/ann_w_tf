import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
 
def normalize(df):
    cols_to_norm = df.columns.difference(['interest_rates [USA]', 'inflation [USA]', 'price'])
    normalized_df = df.copy()
    scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()
    normalized_df[cols_to_norm] = scaler.fit_transform(np.log(normalized_df[cols_to_norm]))
    normalized_df['price'] = price_scaler.fit_transform(np.log(normalized_df['price']).values.reshape(-1,1))
    normalized_df = np.round(normalized_df, decimals=4)
    return normalized_df, price_scaler


def create_labels(df, horizon=30):
    # create labels and add to dataframe
    price_time_series = df['price']
    price_list = price_time_series.values.tolist()
    predictions = [np.asarray(price_list[t+1:t+horizon+1], dtype=np.float32) for t in range(len(price_list)-horizon)]
    predictions.extend([np.NaN for _ in range(horizon)])
    df['predictions']= pd.Series(np.asarray(predictions), index=price_time_series.index)
    df.dropna(inplace=True)
    data = df.values
    x = data[:,:-1].astype(np.float32)
    y = np.stack(data[:,-1])
    return x , y


def split(array, ratio):
    idx = int(array.shape[0] * ratio)
    arr_train = array[:-idx]
    arr_validate = array[-idx:]
    return arr_train, arr_validate
    

def get_subsequences(array, n=50, overlapping=True, data_type=np.float32):
    '''get subsequences of length n from array'''
    list_of_subsequences = []

    if overlapping:
        for i in range(len(array)-n):
            list_of_subsequences.append(np.asarray(array[i:i+n], dtype=data_type))
    else:
        for i in range(len(array)%n, len(array), n):
            list_of_subsequences.append(np.asarray(array[i:i+n],dtype=data_type))

    return np.asarray(list_of_subsequences)
