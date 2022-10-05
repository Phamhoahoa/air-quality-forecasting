# Introduction
 predicting PM2.5 concentrations.


## Requirement
- pandas
- numpy
- matplotlib
- sklearn
- tensorflow
- seaborn

## Model architecture

![Model structure](./data/model-image.jpg)


## Description of data and files

- **data (directory)**:
  - **data-train**: The air-quality dataset from the AI4VN 2022 - Air Quality Forecasting Challenge
  - **public-test**: It consist of 100 folder with different timelines. Each forder includes 4 file as 4 station need prediction
- **utils.py**: It contains supporting functions for the data preprocesing or training 
- **models.py**: The core function for model for the prediction task. The model structure displayed bellow.

- **tmp (directory)**: The folder for storing the model trained.
- **eda.py**: EDA data training .
- **preprocess_data.py**: Clean-data for training: Fill missing values, find neighboring stations 

- **training.py**: training, evaluate model and vizulize result

- **predict_submission.py**: predict data to submit 

## Usage instructions

#### EDA: 

#### Preprocess data: 
- run cell-by-cell preprocess_data.py

#### Training the model & Evaluation:
- ensure run preprocess_data.py before 
- run cell-by-cell training.py. model save at folder tmp

#### Predict submission: 
- run cell-by-cell predict_submission.py



## Thanks
