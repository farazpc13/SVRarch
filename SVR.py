import datetime as dt
import arch.data.sp500
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from arch import arch_model
import sys
sys.path.append('../../')
import os
import warnings
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from common.utils import load_data, mape
from scipy.spatial.distance import mahalanobis


def calculate_mahalanobis_distance(X):
    cov_matrix = np.cov(X, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_vector = np.mean(X, axis=0)
    mahalanobis_distances = []
    for sample in X:
        distance = mahalanobis(sample, mean_vector, inv_cov_matrix)
        mahalanobis_distances.append(distance)
    return np.array(mahalanobis_distances)

# Assign weights based on Mahalanobis distance
def assign_weights(X):
    mahalanobis_distances = calculate_mahalanobis_distance(X)
    weights = 1 / mahalanobis_distances
    return weights

# Create a weighted fuzzy SVR model with Mahalanobis distance
def weighted_fuzzy_svr(X_train, y_train, X_test):
    weights = assign_weights(X_train)
    svr = SVR(kernel='rbf', C=1.0)  # You can adjust hyperparameters
    svr.fit(X_train, y_train, sample_weight=weights)
    y_pred = svr.predict(X_test)
    return y_pred


st = dt.datetime(1988, 1, 1)
en = dt.datetime(2018, 1, 1)
data = arch.data.sp500.load()
market = data["Adj Close"]
returns = 100 * market.pct_change().dropna()

# Create 20 time series with different data
time_series = []
for i in range(20):
    time_series_data = returns.sample(frac=1, replace=True)
    time_series.append(time_series_data)

# Fit SVM model and show summary for each time series
for i, time_series_data in enumerate(time_series):
    time_series_data.index = pd.to_datetime(time_series_data.index)  # Convert index to datetime
    
    train_start_dt = st
    test_start_dt = '2014-12-30 00:00:00'
    
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
    .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    
    test['load'] = scaler.transform(test)
    
    # Converting to numpy arrays
    train_data = train.values
    test_data = test.values
    
    timesteps=5
    
    test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
    test_data_timesteps.shape
    
    x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
    x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
    SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    y_train_pred = weighted_svr(x_train, y_train, x_train)
    y_test_pred = weighted_svr(x_train, y_train, x_test)

    print(y_train_pred.shape, y_test_pred.shape)
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    print(len(y_train_pred), len(y_test_pred))
    y_train = scaler.inverse_transform(y_train)
    y_test = scaler.inverse_transform(y_test)

    print(len(y_train), len(y_test))
    train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
    test_timestamps = energy[test_start_dt:].index[timesteps-1:]

    print(len(train_timestamps), len(test_timestamps))
    plt.figure(figsize=(25,6))
    plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.title("Training data prediction")
    plt.show()
    print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
    plt.figure(figsize=(10,3))
    plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.show()
    print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
    # Extracting load values as numpy array
    data = energy.copy().values

    # Scaling
    data = scaler.transform(data)

    # Transforming to 2D tensor as per model input requirement
    data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
    print("Tensor shape: ", data_timesteps.shape)

    # Selecting inputs and outputs from data
    X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
    print("X shape: ", X.shape,"\nY shape: ", Y.shape)
    # Make model predictions
    Y_pred = model.predict(X).reshape(-1,1)

    # Inverse scale and reshape
    Y_pred = scaler.inverse_transform(Y_pred)
    Y = scaler.inverse_transform(Y)
    plt.figure(figsize=(30,8))
    plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(Y_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.show()
    print('MAPE: ', mape(Y_pred, Y)*100, '%')
    
    
    
    
    '''   
        # Prepare data for SVM model
        X = time_series_data.index.values.reshape(-1, 1)
        y = time_series_data.values
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit SVM model
        svm = SVR()
        svm.fit(X_train, y_train)
        
        # Get predictions on the test set
        y_pred = svm.predict(X_test)
        
        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Summary for Time Series {i+1} (SVM Model):")
        print(f"Mean Squared Error: {mse:.4f}")
        print()
        
        # Fit GARCH model
        am = arch_model(time_series_data, vol='Garch', p=1, q=1)
        res = am.fit()

        print(f"Summary for Time Series {i+1} (GARCH Model):")
        print(res.summary())
        print()
        
        # Plot each time series separately
        ax = time_series_data.plot()
        ax.scatter(X_test, y_test, color='red', label='Actual')
        ax.plot(X_test, y_pred, color='green', label='Predicted')
        xlim = ax.set_xlim(returns.index.min(), returns.index.max())
        plt.title(f"Time Series {i+1} (SVM Model)")
        plt.legend()
        plt.show()
    '''
