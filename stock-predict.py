import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf

def main():
    ticker = "^GSPC"
    #get stock data (no csv), np array
    data = yf.download(ticker, start="1980-01-01", end="2026-01-01")
    print(data.head())
    




if __name__ == "__main__":
    main()