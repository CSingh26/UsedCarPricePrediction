#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import re

# Suppress all warnings
warnings.filterwarnings("ignore")

#loading-data
trainData = pd.read_csv('data/train.csv')
testData = pd.read_csv('data/test.csv')

#pre-processing [TRAIN DATA]

#checking null values
print(trainData.isnull().sum())

#checking duplicate valeus
print(trainData.duplicated().sum())

#dropping null values
trainData.dropna()

#exploring dataset
print(trainData.nunique()) #checking the number of unique values of each column

#cleaning the dataset
trainData['fuel_type'].replace('not supported', 'Electric', inplace = True)

#replacing model_year with age of the car
trainData['age'] = 2024 - trainData['model_year'] #susbtracting from the current year

#getting horsepower extracted from the engine detials 
def extHorsePower(engineDetails):
    horsepower = re.search(r'(\d+\.\d+)HP|\d+\.\d+', engineDetails)
    displacement = re.search(r'(\d+\.\d+L|\d+\.\d+ L)', engineDetails)

    return horsepower.group(1) if horsepower else '', \
           displacement.group(1) if displacement else ''

trainData[['HorsePower', 'Displacement']] = trainData['engine'].apply(extHorsePower).apply(pd.Series)

#horsepower
trainData['HorsePower'] = pd.to_numeric(trainData['HorsePower'], errors= 'coerce')

#engine-displacement
trainData['Displacement'] = trainData['Displacement'].str.replace('L', '')
trainData['Displacement'] = pd.to_numeric(trainData['Displacement'], errors='coerce')

#cleaning accident and clean title column
trainData['accident'] = trainData['accident'].map({'None reported': 0}).fillna(1)
trainData['clean_title'] = trainData['clean_title'].map({'Yes': 1}).fillna(0)

trainData.drop(['model', 'engine', 'model_year', 'ext_col', 'int_col'], axis=1 ,inplace=True)

print(trainData.head())