import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained model and expected feature list
model = joblib.load("forecast_model.pkl")
expected_features = joblib.load("expected_features.pkl")

# Streamlit UI
st.set_page_config(page_title="Forecast Demo", layout="centered")
st.title("📦 Simulated Product Demand Forecasting")

# Sidebar input
st.sidebar.header("Manual Input Parameters")
store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 6)))
date_input = st.sidebar.date_input("Date", value=datetime.today())
onpromotion = st.sidebar.selectbox("Promotion", [0, 1])
transactions = st.sidebar.slider("Transactions", 500, 2500, 1000)
cluster = st.sidebar.selectbox("Cluster", list(range(1, 5)))

families = ["AUTOMOTIVE", "BEAUTY", "CLEANING", "FOODS", "HOME"]

# Function to create input dataframe
def create_input(date_val, fam, store=store_nbr, promo=onpromotion, trans=transactions, clus=cluster):
    df = pd.DataFrame({
        "store_nbr": [store],
        "onpromotion": [promo],
        "transactions": [trans],
        "cluster": [clus],
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

# ----------- FORECAST FROM MANUAL INPUT -----------
if st.button("🔮 Forecast Top Demands (Manual)"):
    predictions = []
    for fam in families:
        input_df = create_input(date_input, fam)
        pred = model.predict(input_df.values)[0]
        pred += np.random.normal(loc=0.5, scale=0.3)
        pred = max(1, round(pred, 2))
        predictions.append((fam, pred))

    # Rank and display
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    st.subheader("🏆 Forecasted Product Demand")
    st.table(pd.DataFrame(sorted_preds, columns=["Family", "Predicted Demand"]))

    # Top-1 family forecast graph
    top_family = sorted_preds[0][0]
    st.subheader(f"📈 7-Day Forecast for: {top_family}")
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
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_top_family.csv", mime="text/csv")




# ----------- FORECAST FROM UPLOADED FILE ----------
st.subheader("📤 Upload Excel/CSV for Bulk Forecasting")
uploaded_file = st.file_uploader("Upload File", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Load file based on type
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file, engine="openpyxl")

    # Parse date
    if "date" in df_upload.columns:
        df_upload["date"] = pd.to_datetime(df_upload["date"], dayfirst=True, errors="coerce")

    # Fill defaults for missing columns
    if "transactions" not in df_upload.columns:
        df_upload["transactions"] = transactions
    if "cluster" not in df_upload.columns:
        df_upload["cluster"] = cluster

    # Build full input dataset for prediction
    input_records = []
    for _, row in df_upload.iterrows():
        date_val = row["date"]
        fam = row["family"]
        input_df = pd.DataFrame({
            "store_nbr": [row["store_nbr"]],
            "onpromotion": [row["onpromotion"]],
            "transactions": [row["transactions"]],
            "cluster": [row["cluster"]],
            "year": [date_val.year],
            "month": [date_val.month],
            "day": [date_val.day],
            "family": [fam]
        })
        input_df = pd.get_dummies(input_df, columns=["family"], prefix=["family"])
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_records.append(input_df[expected_features])

    # Concatenate all rows and predict in one go (faster)
    X_all = pd.concat(input_records).astype(np.float32)
    preds = model.predict(X_all.values)

    # Add randomness + floor
    preds = [max(1, round(p + np.random.normal(loc=0.3, scale=0.2), 2)) for p in preds]
    df_upload["Predicted_Demand"] = preds

    st.success("✅ Bulk Forecasting Completed!")
    st.dataframe(df_upload)

    # Download
    csv = df_upload.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions CSV", data=csv, file_name="bulk_forecast.csv", mime="text/csv")
