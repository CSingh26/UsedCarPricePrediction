#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

trainData.drop(['model', 'engine', 'model_year', 'ext_col', 'int_col', 'clean_title', 'accident'], axis=1 ,inplace=True)

#removing the outliers
q1 = trainData['price'].quantile(0.25)
q3 = trainData['price'].quantile(0.75)
IQR = q3 - q1

trainData = trainData[~((trainData['price'] < (q1 - 1.5 * IQR)) | (trainData['price'] > (q3 + 0.7 * IQR)))]

#without proper feature selection
trainData = pd.get_dummies(trainData, columns=['brand', 'fuel_type', 'transmission'], drop_first=True)
trainData.fillna(0, inplace=True)

X = trainData.drop(['price', 'id'], axis=1)
y = trainData['price']

X['age'] = trainData['age']
X['HorsePower'] = trainData['HorsePower']
X['Displacement'] = trainData['Displacement']

XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2, random_state=42)

#ForestRegressor Model

#model
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(XTrain, yTrain)

yPred = model.predict(XValid)
rmse = np.sqrt(mean_squared_error(yValid, yPred))

#predicitng test data
testData['age'] = 2024 - testData['model_year']

testData[['HorsePower', 'Displacement']] = testData['engine'].apply(extHorsePower).apply(pd.Series)

testData['HorsePower'] = pd.to_numeric(testData['HorsePower'], errors= 'coerce')
testData['Displacement'] = testData['Displacement'].str.replace('L', '')
testData['Displacement'] = pd.to_numeric(testData['Displacement'], errors='coerce')

testData.fillna(0, inplace=True)
testData = pd.get_dummies(testData, columns=['brand', 'fuel_type', 'transmission'], drop_first=True)

missing_cols = set(X.columns) - set(testData.columns)
for col in missing_cols:
    testData[col] = 0

testDataPred = testData[X.columns]

testPred = model.predict(testDataPred)

submission = pd.DataFrame({'id': testData['id'], 'price': testPred})
submission.to_csv('submission.csv', index=False)

print(submission.head())
#leaderboard-position : 861