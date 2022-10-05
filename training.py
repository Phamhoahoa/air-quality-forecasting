import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
from datetime import datetime
import glob 
import plotly.express as px 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf
from model_training.bilstm_model import * 
from utils.utils import * 
def valid_result(X_scaler,Y_scaler, lstm_model,train_df): 
    """
    - predict test 
    - call evaluate function
    - visuzulize result
    """
    df_valid = train_df.drop(["PM2.5_output" , 'timestamp','timestampStr', 'close_dis'], axis = 1).tail(192)
    data_val = X_scaler.fit_transform(df_valid.head(168))
    val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
    pred = lstm_model.predict(val_rescaled)
    pred_Inverse = Y_scaler.inverse_transform(pred)


    validate = train_df.tail(24)
    timeseries_evaluation_metrics_func(validate['PM2.5_output'],pred_Inverse[0])

    plt.figure(figsize=(16,9))
    plt.plot( list(validate['PM2.5_output']))
    plt.plot( list(pred_Inverse[0]))
    plt.title("Actual vs Predicted")
    plt.ylabel("PM2.5")
    plt.legend(('Actual','predicted'))
    plt.show()
def handle(path,  station):
    """
    - input:
        path data cleaned 
        station predict 
    - training and vizulize result
    """
    print('STATION',station )
    df_train = pd.read_csv(path)
    df_train = df_train[df_train['PM2.5_output'].notna()]
    df_train = reduce_mem_usage(df_train)
   
    train_df = features_engineering(df_train)
    train_df = train_df.dropna()
    train_df = train_df.reset_index() 

    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    target = train_df[["PM2.5_output"]]
    features = train_df.drop(["PM2.5_output" , 'timestamp','timestampStr', 'close_dis'], axis = 1)
    X_data = X_scaler.fit_transform(features)
    Y_data = Y_scaler.fit_transform(target)


    TRAIN_SPLIT = 8760
    hist_window = 168 
    horizon = 24
    x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window , horizon)
    x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT , None, hist_window  , horizon )
    input_shape = x_train.shape[-2:]

    batch_size = 1024

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    val_data = val_data.batch(batch_size)

    model = model_lstm(input_shape,horizon)
    lstm_model, history = training(model, train_data ,val_data, station)
    
    valid_result(X_scaler,Y_scaler,lstm_model,train_df)




# run
list_file_train = glob.glob(f'./data/data-clean-top2/*/train.csv')
for file_name in list_file_train:    
    station = file_name.split('/')[3]
    handle(file_name, station)