import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load trained model and expected feature list
model = joblib.load("forecast_model.pkl")
expected_features = joblib.load("expected_features.pkl")

# Streamlit UI
st.set_page_config(page_title="Forecast Demo", layout="centered")
st.title("Simulated Product Demand Forecasting")

st.sidebar.header("Input Parameters")
store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 6)))
date_input = st.sidebar.date_input("Date", value=datetime.today())
onpromotion = st.sidebar.selectbox("Promotion", [0, 1])
transactions = st.sidebar.slider("Transactions", 500, 2500, 1000)
cluster = st.sidebar.selectbox("Cluster", list(range(1, 5)))

families = ["AUTOMOTIVE", "BEAUTY", "CLEANING", "FOODS", "HOME"]

# Input DataFrame builder
def create_input(date_val, fam):
    df = pd.DataFrame({
        "store_nbr": [store_nbr],
        "onpromotion": [onpromotion],
        "transactions": [transactions],
        "cluster": [cluster],
        "year": [date_val.year],
        "month": [date_val.month],
        "day": [date_val.day],
        "family": [fam]
    })
    df = pd.get_dummies(df, columns=["family"], prefix=["family"])
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features].astype(np.float32)

# Forecast logic
if st.button("Forecast Top Demands"):
    predictions = []
    for fam in families:
        input_df = create_input(date_input, fam)
        pred = model.predict(input_df.values)[0]
        pred += np.random.normal(loc=0.5, scale=0.3)
        pred = max(1, round(pred, 2))
        predictions.append((fam, pred))

    # Rank by demand
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    st.subheader("Forecasted Product Demand")
    st.table(pd.DataFrame(sorted_preds, columns=["Family", "Predicted Demand"]))

    # Only top-1 family forecast for next 7 days
    top_family = sorted_preds[0][0]
    st.subheader(f"7-Day Forecast for: {top_family}")
    forecast = []
    future_dates = pd.date_range(start=date_input, periods=7)

    for dt in future_dates:
        input_df = create_input(dt, top_family)
        pred = model.predict(input_df.values)[0]
        pred += np.random.normal(loc=0.3, scale=0.2)
        forecast.append(max(1, round(pred, 2)))

    forecast_df = pd.DataFrame({top_family: forecast}, index=future_dates)
    st.line_chart(forecast_df)

    # CSV download
    csv = forecast_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", data=csv, file_name="forecast_top_family.csv", mime="text/csv")
