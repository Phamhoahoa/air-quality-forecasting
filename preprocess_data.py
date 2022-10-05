
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import gc
import glob 
from datetime import datetime
import seaborn as sns
import os
import geopy.distance
def compute_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    dis = geopy.distance.geodesic(coords_1, coords_2).km
    return dis
location_input = pd.read_csv('./data/data-train/location_input.csv')
location_input = location_input.iloc[:,1:]

location_output = pd.read_csv('./data/data-train/location_output.csv')
location_output = location_output.iloc[:,1:]
location_output = location_output.rename(columns={"station": "station_output", "longitude": "longitude_output", 'latitude':'latitude_output'})

close_station = pd.DataFrame()
for idx, val in location_output.iterrows():
    lat_out, lon_out = val['latitude_output'], val['longitude_output']
    list_sub_dict = []
    for idx1, val1 in location_input.iterrows():
        lat_in, lon_in =  val1['latitude'], val1['longitude']
        dis = compute_distance(lat_out, lon_out, lat_in, lon_in)
        sub_dict = {'station_input': val1['station'], 'dis': dis}
        list_sub_dict.append(sub_dict)
    df_dic = pd.DataFrame(list_sub_dict)
    df_dic = df_dic.sort_values('dis')
    df_dic = df_dic.head(5)
    df_dic['station_output'] = val['station_output']
    close_station = pd.concat([close_station, df_dic])
close_station.to_csv('./data/data-train/map-location.csv', index = False)
def create_sub_input(file_name, field):
    place = file_name.split('/')[3][:-4]
    sub_input = pd.read_csv(file_name)
    sub_input = sub_input.iloc[:,1:]
    sub_input = sub_input.set_index('timestamp')
    sub_input = sub_input[[field]]
    sub_input.columns = sub_input.columns + '_' + place
    return sub_input
def compute_mean(x1,x2):
    A = np.array([x1,x2])
    return np.nanmean(A)
def handle_field(top_close, field):
    input = pd.DataFrame()
    for location_input in top_close:
        file_name_input = f'./data/data-train/input/{location_input}.csv'
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
list_field = ["PM2.5", 'temperature', 'humidity']
map_location = pd.read_csv('./data/data-train/map-location.csv')
for loc_output in map_location['station_output'].unique():
    print(loc_output)
    location_output = map_location[map_location['station_output'] == loc_output]
    top_close = location_output['station_input'].to_list()

    data_input = pd.DataFrame()
    for field_name in list_field:
        input_field =  handle_field(top_close, field_name)
        data_input = pd.concat([data_input, input_field], axis =1)
    data_input['close_dis'] = location_output['dis'].iloc[1]
    
    data_output = create_output(loc_output)
    data = pd.concat([data_input, data_output], axis =1)
    data = data.interpolate( limit_direction='both')

    data = data.dropna()

    print(len(data))
    data = data.reset_index()
    if not os.path.exists(f'./data/data-clean-top2/{loc_output}'):
        os.makedirs(f'./data/data-clean-top2/{loc_output}')
    data.to_csv(f'./data/data-clean-top2/{loc_output}/train.csv', index = False)