# Codtech AI Internship

This repository contains my projects for the Codtech AI Internship.

## Task 4: Stock Price Forecasting (NIFTY-50)

**Problem:** Predict the next day's closing price of RELIANCE using the previous 60 days of closing prices.

**Dataset:** NIFTY-50 Stock Market Data (2000-2021) from Kaggle: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
(The dataset is not included in this repo because of its size.)

**Approach:**
- Used the Close price of RELIANCE
- Split the data by time: 80% training, 20% testing (no shuffling)
- Scaled the data with MinMaxScaler (fitted on training data only)
- Created sequences of 60 days to predict the next day
- Built a 2-layer LSTM model with Dropout and EarlyStopping
- Compared the results with a naive baseline (tomorrow's price = today's price)

**Results:**
LSTM  RMSE: 78.83 | MAE: 48.00 | MAPE: 3.76%
Naive baseline RMSE: 37.91

**CONCLUSION **: Overall project is completed with succesfull results.with all the outputs graph..which is represented in the ipynb file.

**Tools used:** Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib

**File:** `stock_price_forecasting_nifty50.ipynb`
