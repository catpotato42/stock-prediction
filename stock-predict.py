import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#didn't use lstm because it's not on sklearn, but i'll probably use pytorch for the final and add it in.
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import time


def get_error(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(name)
    print(f"mean absolute error = {mae}")
    print(f"mean squared error = {mse}")

def main():
    current_date = time.strftime("%Y-%m-%d", time.localtime())
    ticker = "^GSPC"
    #get stock data (no csv), np array
    stock_data = yf.download(ticker, start="1980-01-01", end=current_date, progress=False)

    #preprocess
    days_to_check = 30 
    
    stock_data['Target_Close'] = stock_data['Close'].shift(-1)

    #lagging features bc i looked up how to do time series regression and it popped up
    for i in range(1, days_to_check + 1):
        stock_data[f'Close_Lag_{i}'] = stock_data['Close'].shift(i)

    #remove nan vals that could have been created by my edits, yf removes on download
    stock_data = stock_data.dropna()

    #using all features that the yahoo finance thing has plus lag features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    lag_cols = [f'Close_Lag_{i}' for i in range(1, days_to_check + 1)]
    features.extend(lag_cols)
    X = stock_data[features]
    y = stock_data['Target_Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #model training
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    nn_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
    nn_model.fit(X_train_scaled, y_train)
    nn_predictions = nn_model.predict(X_test_scaled)

    #eval
    get_error("Multiple Linear Regression", y_test, lr_predictions)
    get_error("Neural Network", y_test, nn_predictions)

if __name__ == "__main__":
    main()