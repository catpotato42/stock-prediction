import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import time


def get_error(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    root_mse = np.sqrt(mse)

    print(name)
    print(f"mean absolute error = {mae}")
    print(f"root mean squared error = {root_mse}")

def main():
    current_date = time.strftime("%Y-%m-%d", time.localtime())
    print(current_date)
    ticker = "^GSPC"
    #get stock data (no csv), np array
    stock_data = yf.download(ticker, start="1980-01-01", end=current_date, progress=False)

    #preprocess
    days_to_check = 30
    predict_days = 30
    
    future_close = stock_data['Close'].shift(-predict_days)

    stock_data['Target_Date'] = stock_data.index.to_series().shift(-predict_days)
    stock_data['Target_Return'] = (future_close - stock_data['Close']) / stock_data['Close']

    #lagging features
    for i in range(1, days_to_check + 1):
        stock_data[f'Close_Lag_{i}'] = stock_data['Close'].shift(i)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    lag_cols = [f'Close_Lag_{i}' for i in range(1, days_to_check + 1)]
    features.extend(lag_cols)
    live_prediction_data = stock_data.tail(predict_days).copy()

    stock_data = stock_data.dropna()
    X = stock_data[features]
    y = stock_data['Target_Return']
    base_prices = stock_data['Close']
    target_dates = stock_data['Target_Date']

    X_train, X_test, y_train, y_test, base_train, base_test, dates_train, dates_test = train_test_split(
        X, y, base_prices, target_dates, test_size=0.2, shuffle=False
    )

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    lstm_model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
        Dense(32),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mae')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    lstm_model.fit(
        X_train_lstm, y_train_scaled, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.1, 
        callbacks=[early_stop], 
        verbose=0
    )

    #flatten the predictions so they match the 1D shape of y_test
    lstm_predictions = lstm_model.predict(X_test_lstm, verbose=0).flatten()

    #lr model training
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    lr_predictions = lr_model.predict(X_test_scaled)

    #nn model training
    nn_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
    nn_model.fit(X_train_scaled, y_train_scaled)
    nn_predictions = nn_model.predict(X_test_scaled)

    volatility_multiplier = 1

    #inverse transform to get the actual predicted returns
    lr_pred_returns = y_scaler.inverse_transform(lr_predictions.reshape(-1, 1)).flatten() * volatility_multiplier
    nn_pred_returns = y_scaler.inverse_transform(nn_predictions.reshape(-1, 1)).flatten() * volatility_multiplier
    lstm_pred_returns = y_scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten() * volatility_multiplier
    actual_returns = y_test.values.flatten()

    base_vals = base_test.values.flatten()

    lr_pred_prices = base_vals * (1 + lr_pred_returns)
    nn_pred_prices = base_vals * (1 + nn_pred_returns)
    lstm_pred_prices = base_vals * (1 + lstm_pred_returns)
    actual_prices = base_vals * (1 + actual_returns)

    # eval
    get_error("Multiple Linear Regression", actual_prices, lr_pred_prices)
    get_error("Neural Network", actual_prices, nn_pred_prices)
    get_error("LSTM", actual_prices, lstm_pred_prices)

    print(f"day before price: {base_test.iloc[-1]}")
    print(f"actual price: {actual_prices[-1]}")
    print(f"predicted price lr: {lr_pred_prices[-1]}")
    print(f"predicted price nn: {nn_pred_prices[-1]}")
    print(f"predicted price lstm: {lstm_pred_prices[-1]}")

    #plot results
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, actual_prices, label='True Price', color='black', linewidth=2)
    plt.plot(dates_test, lr_pred_prices, label='MLR', alpha=0.8, linestyle='--')
    plt.plot(dates_test, nn_pred_prices, label='Neural Network', alpha=0.8, linestyle='-.')
    plt.plot(dates_test, lstm_pred_prices, label='LSTM', alpha=0.8, linestyle=':')
    
    plt.title("Model Predictions vs Actual Prices", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ticker+"Price (USD)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #lstm predict
    today_features = live_prediction_data[features].iloc[-1].values.reshape(1, -1)
    
    today_scaled = X_scaler.transform(today_features)
    today_lstm = today_scaled.reshape((1, 1, today_scaled.shape[1]))
    
    predicted_return_scaled = lstm_model.predict(today_lstm, verbose=0)
    predicted_return = y_scaler.inverse_transform(predicted_return_scaled).flatten()[0] * volatility_multiplier
    
    try:
        today_price = float(live_prediction_data['Close'].iloc[-1].iloc[0])
    except:
        today_price = float(live_prediction_data['Close'].iloc[-1])
    tomorrow_predicted_price = today_price * (1 + predicted_return)

    print(f"\nTarget Prediction Date: {current_date}")
    print(f"Based on Data From: {live_prediction_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Predicted Close: {tomorrow_predicted_price:.2f}")

if __name__ == "__main__":
    main()