# Car Price Prediction

This project aims to predict car prices using various machine learning algorithms including Random Forest, XGBoost, Decision Tree, and Neural Networks. The project involves preprocessing car data, training models, and generating predictions.

## Project Structure

- `main.py`: Main script to preprocess data, train models, and make predictions.
- `data/`: Directory containing `train.csv` and `test.csv` files.
- `submission*.csv`: Output files containing the predictions for the test dataset.

## Requirements

Data was fetched from Kagglex competition but here's an alternative <a href="https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset">Link</a>

Ensure you have the following libraries installed:

- pandas
- numpy
- re
- keras_tuner
- scikit-learn
- xgboost
- keras (TensorFlow backend)
- warnings

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Data Preparation
The data preparation steps include :

- Loading the training and test datasets
- Extracting horsepower and displacement from the engine details.
- Adding new features and encoding categorical variables.
- Normalizing the data.

## Model Training and Evaluation
The following models are trained and evaluated in the script:

1. Random Forest Regressor:

    - Trained with 100 estimators.
    - Evaluated using RMSE.
    - Predictions are saved in submission.csv.

2. XGBoost Regressor:

    - Hyperparameters tuned using GridSearchCV.
    - Best model's predictions are saved in submission1.csv.

3. Neural Networks:

    - Constructed using Keras Sequential API.
    - Trained for 100 epochs.
    - Predictions are saved in submission2.csv.

4. Keras Tuner for Hyperparameter Optimization:

    - Random search for best hyperparameters.
    - Best model's predictions are saved in submission3.csv.

5. Decision Tree Regressor:

    - Trained with a maximum depth of 10 and minimum samples per leaf of 10.
    - Predictions are saved in submission4.csv.

## Usage
1. Place the train.csv and test.csv files in the data/ directory.
2. Run the main.py script:
```bash
python main.py
```
3. The script will output the RMSE scores for each model and generate prediction files: submission.csv, submission1.csv, submission2.csv, submission3.csv, and submission4.csv.
