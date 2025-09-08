import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load stacked model (contains CatBoost, so CatBoost must be installed)
model = joblib.load("BEST_STACKED_MODEL_OPTUNA.pkl")

# Expected features (must match model input)
expected_features = [
    'store_nbr', 'onpromotion', 'cluster', 'transactions', 'year', 'month', 'day',
    'family_AUTOMOTIVE', 'family_BEAUTY', 'family_CELEBRATION', 'family_CLEANING', 'family_CLOTHING',
    'family_FOODS', 'family_GROCERY', 'family_HARDWARE', 'family_HOME', 'family_LADIESWEAR',
    'family_LAWN AND GARDEN', 'family_LIQUOR,WINE,BEER', 'family_PET SUPPLIES', 'family_STATIONERY',
    'city_Ambato', 'city_Babahoyo', 'city_Cayambe', 'city_Cuenca', 'city_Daule', 'city_El Carmen',
    'city_Esmeraldas', 'city_Guaranda', 'city_Guayaquil', 'city_Ibarra', 'city_Latacunga', 'city_Libertad',
    'city_Loja', 'city_Machala', 'city_Manta', 'city_Playas', 'city_Puyo', 'city_Quevedo', 'city_Quito',
    'city_Riobamba', 'city_Salinas', 'city_Santo Domingo',
    'holiday_type_Additional', 'holiday_type_Bridge', 'holiday_type_Event',
    'holiday_type_Holiday', 'holiday_type_Transfer', 'holiday_type_Work Day'
]

# Set Streamlit page config
st.set_page_config(page_title="Product Demand Forecasting", layout="centered")
st.title("📦 Product Demand Forecasting App")

st.sidebar.header("🧾 Input Parameters")

# Sidebar inputs
store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 55)))
date_input = st.sidebar.date_input("Select Date", value=datetime.today())
onpromotion = st.sidebar.selectbox("Is Promotion Active?", [0, 1])
transactions = st.sidebar.number_input("Transactions", min_value=0, value=1000)
cluster = st.sidebar.selectbox("Cluster", list(range(1, 18)))
family = st.sidebar.selectbox("Product Family", [
    "AUTOMOTIVE", "BEAUTY", "CELEBRATION", "CLEANING", "CLOTHING", "FOODS",
    "GROCERY", "HARDWARE", "HOME", "LADIESWEAR", "LAWN AND GARDEN", "LIQUOR,WINE,BEER",
    "PET SUPPLIES", "STATIONERY"
])
city = st.sidebar.selectbox("City", [
    "Ambato", "Babahoyo", "Cayambe", "Cuenca", "Daule", "El Carmen", "Esmeraldas", "Guaranda", "Guayaquil",
    "Ibarra", "Latacunga", "Libertad", "Loja", "Machala", "Manta", "Playas", "Puyo", "Quevedo", "Quito",
    "Riobamba", "Salinas", "Santo Domingo"
])
holiday_type = st.sidebar.selectbox("Holiday Type", [
    "Additional", "Bridge", "Event", "Holiday", "Transfer", "Work Day"
])

# Build input dataframe
def create_input_df(selected_date, dynamic_family=None):
    fam = dynamic_family if dynamic_family else family
    df = pd.DataFrame({
        "store_nbr": [store_nbr],
        "onpromotion": [onpromotion],
        "transactions": [transactions],
        "cluster": [cluster],
        "date": [pd.to_datetime(selected_date)],
        "family": [fam],
        "city": [city],
        "holiday_type": [holiday_type]
    })
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df.drop(columns=["date"], inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["family", "city", "holiday_type"], prefix=["family", "city", "holiday_type"])

    # Add any missing columns
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    df = df.astype(np.float32)
    return df

# Predict when button is clicked
if st.button("🔮 Predict Demand for Selected Date"):
    input_df = create_input_df(date_input)
    raw_pred = model.predict(input_df.values)[0]
    prediction = max(raw_pred, 1)  # Avoid showing 0

    st.success(f"📈 Predicted Demand: **{round(prediction)} units**")

    # 📊 10-Day Forecast
    st.subheader("📅 10-Day Demand Forecast")
    future_dates = pd.date_range(start=date_input, periods=10)
    forecast_data = []

    for dt in future_dates:
        row = create_input_df(dt)
        raw = model.predict(row.values)[0]
        forecast_data.append((dt, max(raw, 1)))

    forecast_df = pd.DataFrame(forecast_data, columns=["Date", "Forecasted Demand"])
    st.line_chart(forecast_df.set_index("Date"))

    # 🔍 Predict Which Product Family is Most in Demand
    st.subheader("🔥 Product Family Most in Demand")

    all_families = [
        "AUTOMOTIVE", "BEAUTY", "CELEBRATION", "CLEANING", "CLOTHING", "FOODS",
        "GROCERY", "HARDWARE", "HOME", "LADIESWEAR", "LAWN AND GARDEN", "LIQUOR,WINE,BEER",
        "PET SUPPLIES", "STATIONERY"
    ]

    demand_predictions = []

    for fam in all_families:
        temp_df = create_input_df(date_input, dynamic_family=fam)
        pred = model.predict(temp_df.values)[0]
        demand_predictions.append((fam, round(pred, 2)))

    # Sort by highest predicted demand
    demand_predictions.sort(key=lambda x: x[1], reverse=True)
    top_fam, top_demand = demand_predictions[0]
    demand_predictions = [x for x in demand_predictions if x[1] > 0]


    st.success(f"🔥 **{top_fam}** is expected to be in highest demand with approximately **{top_demand} units**.")
    st.write("Here’s the full ranked list:")

    st.table(pd.DataFrame(demand_predictions, columns=["Product Family", "Predicted Demand"]))

    # CSV Download
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

    # 🏆 Top 5 Products in Demand
    st.subheader("🏆 Top 5 Product Families in Demand")
    all_families = [
        "AUTOMOTIVE", "BEAUTY", "CELEBRATION", "CLEANING", "CLOTHING", "FOODS",
        "GROCERY", "HARDWARE", "HOME", "LADIESWEAR", "LAWN AND GARDEN", "LIQUOR,WINE,BEER",
        "PET SUPPLIES", "STATIONERY"
    ]

    ranked_families = []
    for fam in all_families:
        temp_df = create_input_df(date_input, dynamic_family=fam)
        pred = model.predict(temp_df.values)[0]
        ranked_families.append((fam, round(pred, 2)))

    top5 = sorted(ranked_families, key=lambda x: x[1], reverse=True)[:5]
    st.table(pd.DataFrame(top5, columns=["Product Family", "Predicted Demand"]))
