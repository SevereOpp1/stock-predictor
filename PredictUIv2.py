import streamlit as st
import requests

st.title("📈 Stock Predictor 📉")
ticker = st.text_input("Enter Ticker")

if st.button("Predict"):
    if ticker:
        url = st.secrets["BACKEND_URL"]
        payload = {"ticker": ticker}
        try:
            resp = requests.post(url, json=payload)
            data = resp.json()
            st.write(data)
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

