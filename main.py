import pandas as pd
import numpy as np
import re

import keras_tuner as kt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.optimizers import Adam, RMSprop, SGD

import warnings
warnings.filterwarnings("ignore")

# Loading and pre-processing data
trainData = pd.read_csv('data/train.csv')
testData = pd.read_csv('data/test.csv')

testID = testData['id']

def extHorsePower(engineDetails):
    if isinstance(engineDetails, str):
        horsepower = re.search(r'(\d+\.?\d*)HP', engineDetails)
        displacement = re.search(r'(\d+\.?\d*)L', engineDetails)
        return (float(horsepower.group(1)) if horsepower else np.nan, float(displacement.group(1)) if displacement else np.nan)
    return (np.nan, np.nan)

def preprocess_data(data):
    data['age'] = 2024 - data['model_year']
    data[['HorsePower', 'Displacement']] = data['engine'].apply(extHorsePower).apply(pd.Series)
    data['HorsePower'] = pd.to_numeric(data['HorsePower'], errors='coerce')
    data['Displacement'] = data['Displacement'].astype(str).str.replace('L', '')
    data['Displacement'] = pd.to_numeric(data['Displacement'], errors='coerce')
    data.fillna(0, inplace=True)
    data.drop(['model', 'engine', 'ext_col', 'int_col', 'clean_title', 'accident', 'model_year'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['brand', 'fuel_type', 'transmission'], drop_first=True)
    return data

trainData = preprocess_data(trainData)
testData = preprocess_data(testData)

def ensure_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)
    return df

trainData = ensure_numeric(trainData)
testData = ensure_numeric(testData)

# Remove outliers
q1 = trainData['price'].quantile(0.25)
q3 = trainData['price'].quantile(0.75)
IQR = q3 - q1
trainData = trainData[~((trainData['price'] < (q1 - 1.5 * IQR)) | (trainData['price'] > (q3 + 0.7 * IQR)))]

X = trainData.drop(['price', 'id'], axis=1)
y = trainData['price']

# Handle missing columns in test data
missing_cols = set(X.columns) - set(testData.columns)
for col in missing_cols:
    testData[col] = 0

testData = testData[X.columns]  # Ensure the same order of columns

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
testData = scaler.transform(testData)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print('RMSE Score : ', rmse)

# Predicting test data
testPred = model.predict(testData)
submission = pd.DataFrame({'id': testID, 'price': testPred})
submission.to_csv('submission.csv', index=False)
print(submission.head())

# GridSearchCV for XGBRegressor
params = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_depth': [3, 5, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15]
}

gridSearchCV = GridSearchCV(
    estimator=XGBRegressor(),
    param_grid=params,
    cv=3,
    verbose=10
)

gridSearchCV.fit(X_train, y_train)
bestModel = gridSearchCV.best_estimator_

testPred1 = bestModel.predict(testData)
submission = pd.DataFrame({'id': testID, 'price': testPred1})
submission.to_csv('submission1.csv', index=False)
print(submission.head())

# Neural Networks
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))

testPred2 = model.predict(testData).flatten()
submission = pd.DataFrame({'id': testID, 'price': testPred2})
submission.to_csv('submission2.csv', index=False)
print(submission.head())

# Hyperparameter tuning with Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=16, max_value=256, step=16), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=hp.Int('units2', min_value=16, max_value=256, step=16), activation='relu'))
    model.add(Dense(1))
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=lr)
    else:
        opt = SGD(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse')
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=2,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='car_price_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
bestModel = tuner.hypermodel.build(bestHps)
bestModel.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

testPred3 = bestModel.predict(testData).flatten()
submission = pd.DataFrame({'id': testID, 'price': testPred3})
submission.to_csv('submission3.csv', index=False)
print(submission.head())

#desicion-tree
tree = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, splitter='best')
tree.fit(X_train, y_train)

ypred = tree.predict(X_valid)
rmseDT = np.sqrt(mean_squared_error(y_valid, ypred))

print('Decision Tree RMSE Score:', rmseDT)

testPred4 = tree.predict(testData)
submission = pd.DataFrame({'id': testID, 'price': testPred4})
submission.to_csv('submission4.csv', index=False)
print(submission.head())