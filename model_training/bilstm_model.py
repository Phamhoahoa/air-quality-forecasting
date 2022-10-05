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
import tensorflow as tf
import mlflow
def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon

    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])

        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)
def model_lstm(input_shape,horizon):
    dnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                        strides=1, padding="causal",
                        activation='relu',
                        input_shape=input_shape),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation= 'relu')),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=horizon),
    ])
    dnn_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    dnn_model.summary()
    return dnn_model
def training(lstm_model, train_data ,val_data, station):
    mlflow.set_experiment(f'BiLSTM-{station}')
    CHECKPOINT_DIR = "./tmp/"
    model_path = CHECKPOINT_DIR + f'Bidirectional_LSTM_Multivariate_{station}.h5'
    
    early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='auto', verbose=0)
    callbacks=[early_stopings,checkpoint]
    mlflow.mlflow.autolog()
    with mlflow.start_run():
        history = lstm_model.fit(train_data,epochs=100,validation_data=val_data,verbose=1,callbacks=callbacks)
        mlflow.log_artifacts(CHECKPOINT_DIR, 'checkpoints')
    return lstm_model, history


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MDAPE is : {np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
    
