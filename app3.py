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

# ----------- FORECAST FROM UPLOADED EXCEL ----------
st.subheader("📤 Upload Excel for Bulk Forecasting")
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    df_upload = pd.read_excel(uploaded_file)

    # Convert date column if needed
    if "date" in df_upload.columns:
        df_upload["date"] = pd.to_datetime(df_upload["date"], dayfirst=True, errors="coerce")

    results = []
    for _, row in df_upload.iterrows():
        date_val = row["date"]
        fam = row["family"]
        store = row["store_nbr"]
        promo = row["onpromotion"]

        # Using defaults for cluster & transactions if not present
        clus = row["cluster"] if "cluster" in df_upload.columns else cluster
        trans = row["transactions"] if "transactions" in df_upload.columns else transactions

        input_df = create_input(date_val, fam, store, promo, trans, clus)
        pred = model.predict(input_df.values)[0]
        pred += np.random.normal(loc=0.3, scale=0.2)
        pred = max(1, round(pred, 2))
        results.append(pred)

    # Add predictions to dataframe
    df_upload["Predicted_Demand"] = results

    st.success("✅ Forecasting Completed!")
    st.dataframe(df_upload)

    # CSV download
    csv = df_upload.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions CSV", data=csv, file_name="bulk_forecast.csv", mime="text/csv")
