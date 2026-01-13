import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# PAGE SETUP

st.set_page_config(page_title="UIDAI Biometric Dashboard", layout="wide")

st.title("UIDAI Aadhaar Biometric Analytics Dashboard")
st.write(
    "Excel-style dashboard showing **trends, patterns, anomalies, and predictive indicators** "
    "from anonymised Aadhaar biometric data."
)


# LOAD DATA

DATA_DIR = Path("data")

files = [
    "Biometric_part1.csv",
    "Biometric_part2.csv",
    "Biometric_part3.csv",
    "Biometric_part4.csv"
]

df_list = [pd.read_csv(DATA_DIR / f) for f in files]
data = pd.concat(df_list, ignore_index=True)

data["date"] = pd.to_datetime(data["date"], dayfirst=True)
data["year"] = data["date"].dt.year
data["total_biometric"] = data["bio_age_5_17"] + data["bio_age_17_"]


# FILTERS (Excel slicer style)

st.sidebar.header("Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(data["year"].unique())
)

filtered_data = data[data["year"] == selected_year]


# MONTHLY TREND

monthly_trend = (
    filtered_data
    .groupby(filtered_data["date"].dt.to_period("M"))["total_biometric"]
    .sum()
    .reset_index()
)

monthly_trend["date"] = monthly_trend["date"].dt.to_timestamp()
monthly_trend = monthly_trend.sort_values("date")


# AGE GROUP PATTERN

monthly_age = (
    filtered_data
    .groupby(filtered_data["date"].dt.to_period("M"))[["bio_age_5_17", "bio_age_17_"]]
    .sum()
    .reset_index()
)

monthly_age["date"] = monthly_age["date"].dt.to_timestamp()


# ANOMALY DETECTION

monthly_trend["pct_change"] = monthly_trend["total_biometric"].pct_change() * 100
monthly_trend["pct_change_%"] = (monthly_trend["pct_change"].round(2).astype(str) + "%")
monthly_trend["anomaly"] = monthly_trend["pct_change"].apply(
    lambda x: "Anomaly" if pd.notna(x) and (x > 20 or x < -20) else "Normal"
)


# FORECAST (MOVING AVERAGE)

window = 3
monthly_trend["moving_avg"] = (
    monthly_trend["total_biometric"].rolling(window).mean()
)

last_date = monthly_trend["date"].iloc[-1]
last_avg = monthly_trend["moving_avg"].iloc[-1]

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

forecast = pd.DataFrame({
    "date": future_dates,
    "forecast_biometric": [last_avg] * 3
})


# KPI METRICS (Excel summary cells)

st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Biometric Activity",
    f"{monthly_trend['total_biometric'].sum():,}"
)

col2.metric(
    "Highest Monthly Activity",
    f"{monthly_trend['total_biometric'].max():,}"
)

col3.metric(
    "Lowest Monthly Activity",
    f"{monthly_trend['total_biometric'].min():,}"
)


# TREND + FORECAST CHART

st.header("1️⃣ Monthly Trend & Forecast")

fig1, ax1 = plt.subplots()
ax1.plot(monthly_trend["date"], monthly_trend["total_biometric"], label="Actual")
ax1.plot(forecast["date"], forecast["forecast_biometric"],
         linestyle="--", marker="o", label="Forecast")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Biometric Activity")
ax1.legend()
st.pyplot(fig1)


# AGE GROUP PATTERN

st.header("2️⃣ Age-group Pattern")

fig2, ax2 = plt.subplots()
ax2.plot(monthly_age["date"], monthly_age["bio_age_5_17"], label="Age 5–17")
ax2.plot(monthly_age["date"], monthly_age["bio_age_17_"], label="Age 17+")
ax2.set_xlabel("Month")
ax2.set_ylabel("Biometric Activity")
ax2.legend()
st.pyplot(fig2)


# ANOMALY TABLE

st.header("3️⃣ Anomaly Detection")

st.dataframe(
    monthly_trend[["date", "total_biometric", "pct_change_%", "anomaly"]],
    use_container_width=True
)


# FORECAST TABLE

st.header("4️⃣ Predictive Indicators")

st.dataframe(forecast, use_container_width=True)

