import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle


st.title("📈 Stock Predictor 📉")
st.write("Enter a stock ticker to predict next trading day's close compared to yesterday's.")

# --- Input ---
with st.form("predict_form"):
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")
    submit = st.form_submit_button("🔮 PREDICT")


# --- Load model once ---
MODEL_PATH = "stock_cnn.h5"
model = load_model(MODEL_PATH)
SCALER_PATH = "scaler.pkl"

# --- Load model and scaler once --
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# --- Data transformation function ---
def transform_data(look_back, stock_data):
    """Transform stock data into CNN input format."""
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)
        stock_data.columns.name = None

    prices = stock_data.copy()
    prices['Date'] = prices.index
    prices = prices.dropna()

    # Cast numeric
    prices[['Open','High','Low','Close']] = prices[['Open','High','Low','Close']].astype(float)
    prices['Volume'] = prices['Volume'].astype(int)

    # Percent change
    prices[['Open_pc','High_pc','Low_pc','Close_pc']] = prices[['Open','High','Low','Close']].pct_change()
    features = ['Open_pc','High_pc','Low_pc','Close_pc']

    # Look-back window
    for num in range(look_back):
        for price_type in features:
            col_name = f"{price_type}_{num}"
            prices[col_name] = prices[price_type].shift(num + 1)

    cols_to_keep = ['Date'] + [col for col in prices.columns if '_pc' in col]
    prices_pattern = prices[cols_to_keep].dropna()

    # Drop current day's percent change cols
    prices_pattern = prices_pattern.drop(['Open_pc', 'High_pc', 'Low_pc', 'Close_pc'], axis=1)
    X = scaler.transform(prices_pattern.drop(['Date'], axis=1))

    # Reshape for CNN: (samples, timesteps, features)
    X = X.reshape(X.shape[0], look_back, 4).astype(np.float32)
    return X

# --- Predict button ---
if submit:
    if not ticker:
        st.error("Please enter a ticker symbol first, press 'Enter', then 'Predict'.")
    else:
        try:
            st.write(f"Fetching last 7 days of data for {ticker.upper()}...")
            df = yf.download(ticker, period="9d")
            if df.empty:
                st.error("No data found for ticker.")
            else:
                look_back = 7
                X = transform_data(look_back, df)
                preds = model.predict(X).tolist()

                st.success("Prediction complete!")
                for i, p in enumerate(preds):
                    down_prob = p[0] * 100
                    up_prob = p[1] * 100
                    st.write(f"🔻 {down_prob:.1f}% | 🔺 {up_prob:.1f}%")

        except Exception as e:
            st.error(f"Error fetching data or predicting: {e}")



