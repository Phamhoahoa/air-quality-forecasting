import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
from datetime import datetime
import gc
import glob 
import plotly.express as px 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import geopy.distance
def compute_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    dis = geopy.distance.geodesic(coords_1, coords_2).km
    return dis
def compute_mean(x1,x2):
    A = np.array([x1,x2])
    return np.nanmean(A)
from pandas.io.formats.info import DataFrameInfo
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
def features_engineering(df):
    # Sort by timestamp
    df['timestampStr'] = df['timestamp'].apply(lambda x:datetime.strptime(x, "%d/%m/%Y %H:%M"))
    df['timestamp'] = pd.to_datetime(df['timestamp'],infer_datetime_format=True)

    df['day'] = [int(d.strftime('%d')) for d in df.timestampStr]
    df['dayofyear'] = [int(d.strftime('%j')) for d in df.timestampStr]
    df['month'] = [int(d.strftime('%m')) for d in df.timestampStr]
    df["hour"] = [int(d.strftime('%H')) for d in df.timestampStr]
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year

    gc.collect()
    return df
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

