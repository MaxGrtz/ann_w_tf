import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
 
def normalize(batchX, batchY):
    '''
    Method to normalize batch, every sequence in the batch individually
    params: batchX - features of current batch
            batchY - labels of current batch
    returns: normalized batchX - normalized features
             normalized batchY - normalized labels
             origin - first value of the features of the current sequence for later inversion of normalization
    '''
    norm_batchX, norm_batchY = [], []
    for seqX, seqY in zip(batchX, batchY):
        origin = seqX[0]
        norm_batchX.append((seqX/origin) - 1) 
        norm_batchY.append((seqY/origin) - 1)
    return np.asarray(norm_batchX), np.asarray(norm_batchY), origin
    
def invert_normalization(norm_batch, origin):
    '''
    Method to invert normalization
    params: norm_batch - normalized batch
            origin - anchor for inversin of normalization
    returns: batch - batch with inverted normalization
    '''
    batch = origin * (norm_batch +1)
    return batch

def create_labels(df):
    '''
    Method to create labels for feature dataframe, labels being the features of the next timestep
    params: df - feature dataframe
    returns: x - feature array
             y - label array
    '''
    features = df.values.tolist()
    predictions = [np.array(features[day+1]) for day in range(len(features)-2)]
    predictions.extend([np.NaN for _ in range(2)])
    df['predictions']= pd.Series(np.asarray(predictions), index=df.index)
    df.dropna(inplace=True)
    data = df.values
    x = data[:,:-1].astype(np.float32)
    y = np.stack(data[:,-1])
    return x, y

def split(array, ratio):
    '''
    Method to split given array on axis 0 based on a given ratio (into training/validation data)
    params: array - array to split
            ratio - percentage of validation data (where to split)
    returns: arr_train - training part of input array
             arr_validate - validation part of input array
    '''
    idx = int(array.shape[0] * ratio)
    arr_train = array[:-idx]
    arr_validate = array[-idx:]
    return arr_train, arr_validate
    

def get_subsequences(array, n=50, overlapping=True, data_type=np.float32):
    '''
    Method to et subsequences of length n from given array.
    params: array - array of dimension d to split
            n - length of subsequences
            overlapping - boolean flag for overlapping or non overlapping subsequences
            data_type - data type of the array values (relevant for date format)
    returns: array of dimension d+1 of subsequences of the input array
    '''
    list_of_subsequences = []

    if overlapping:
        for i in range(len(array)-n):
            list_of_subsequences.append(np.asarray(array[i:i+n], dtype=data_type))
    else:
        for i in range(len(array)%n, len(array), n):
            list_of_subsequences.append(np.asarray(array[i:i+n],dtype=data_type))

    return np.asarray(list_of_subsequences)
