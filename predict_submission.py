import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
import gc
import glob 
from datetime import datetime

import plotly.express as px 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf
from utils.utils import * 
def create_sub_input(file_name, field):
    place = file_name.split('/')[4][:-4]
    sub_input = pd.read_csv(file_name)
    sub_input = sub_input.iloc[:,1:]
    sub_input = sub_input.set_index('timestamp')
    sub_input = sub_input[[field]]
    sub_input.columns = sub_input.columns + '_' + place
    return sub_input
def handle_field(i, top_close, field):
    input = pd.DataFrame()
    for location_input in top_close:
        file_name_input = f'./data/public-test/input/{i}/{location_input}.csv'
        sub_input =  create_sub_input(file_name_input, field)
        input = pd.concat([input,sub_input ], axis = 1)

    tail_3 = input.columns
    input['mean_tail3'] = input.apply(lambda x: compute_mean(x[tail_3[0]], x[tail_3[1]]),axis=1)
    for i in range(0,2):
        input[ f'{field}_{i}'] = np.where(input.iloc[:,i].notna(),input.iloc[:,i], input['mean_tail3'] )
    list_near_station = [f'{field}_{i}' for i in range(0,2)]
    input = input[list_near_station]
    return input
def create_output(location_output):
    df_output = pd.read_csv(f'./data/data-train/output/{location_output}.csv')
    df_output = df_output.iloc[:,1:]
    df_output = df_output.rename(columns={"PM2.5": "PM2.5_output"})
    df_output = df_output.drop(['humidity', 'temperature'], axis=1)
    df_output = df_output.set_index('timestamp')
    return df_output
def scale_data(path_train):
    df_train = pd.read_csv(path_train)
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    df_train = df_train[df_train['PM2.5_output'].notna()]
    df_train = features_engineering(df_train)
    df_train = df_train.dropna()
    df_train = df_train.reset_index()
    
   
    target = df_train[["PM2.5_output"]]
    features = df_train.drop(["PM2.5_output" , 'timestamp','timestampStr'], axis = 1)
    X_data = X_scaler.fit(features)
    Y_data = Y_scaler.fit(target)
    return X_scaler,Y_scaler
    
def predict_result(X_scaler,Y_scaler, lstm_model ,df_input):
    data_val = X_scaler.fit_transform(df_input)
    val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
    pred = lstm_model.predict(val_rescaled)
    pred_Inverse = Y_scaler.inverse_transform(pred)
    result = pred_Inverse[0]
    return result

import os
from tensorflow import keras
df_location_output = pd.read_csv('./data/public-test/location.csv')
df_location_output = df_location_output.iloc[:,1:]
df_location_output = df_location_output.rename(columns={"station": "station_output", "longitude": "longitude_output", 'latitude':'latitude_output'})


list_field = ["PM2.5", 'temperature', 'humidity']
map_location = pd.read_csv('./data/data-train/map-location.csv')

for i in range(1,101):
    for idx, val in df_location_output.iterrows():
        loc_output = val['station_output']
        location_output = map_location[map_location['station_output'] == loc_output]
        top_close = location_output['station_input'].to_list()

        data_input = pd.DataFrame()
        for field_name in list_field:
            input_field =  handle_field(i,top_close, field_name)
            data_input = pd.concat([data_input, input_field], axis =1)
        data_input['close_dis'] = location_output['dis'].iloc[1]
        

        data_input = data_input.reset_index()
        # data_input = add_lag_feature(data_input)
        data_input= data_input.interpolate(  limit_direction='both')

        test_df = features_engineering(data_input)
        test_df = test_df.reset_index()
        test_df = test_df.drop([ 'timestamp','timestampStr','close_dis'], axis = 1)

        path_train = f'./data/data-clean-top2/{loc_output}/train.csv'
        X_scaler,Y_scaler = scale_data(path_train)
        path_model = f'./tmp/Bidirectional_LSTM_Multivariate_{loc_output}.h5'
        model = keras.models.load_model(path_model)
        PM25 = predict_result(X_scaler,Y_scaler, model ,test_df)
        result = pd.DataFrame(PM25, columns = ['PM2.5'])
        result['PM2.5'] = np.where(result['PM2.5']>=0, result['PM2.5'], np.nan)
        result['PM2.5']  = result['PM2.5'].interpolate( limit_direction='both')

        if not os.path.exists(f'./result/{i}'):
            os.makedirs(f'./result/{i}')
        save_name = f'./result/{i}/res_{i}_{idx+1}.csv'
        result.to_csv(save_name, index = False)
        print(save_name, len(result))

