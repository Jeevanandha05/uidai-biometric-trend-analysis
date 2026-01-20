import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(
    page_title="UIDAI Biometric Dashboard",
    layout="wide"
)

st.title("UIDAI Aadhaar Biometric Analytics Dashboard")
st.write(
    "Dashboard showing **monthly trends, age-group contribution, "
    "regional hotspots, anomaly detection, and predictive analysis**."
)

# =================================================
# LOAD DATA FROM EXCEL (vs.py source)
# =================================================
file_path = "outputs/UIDAI_All_Reports.xlsx"

month_df = pd.read_excel(file_path, sheet_name="Month_Wise_Report")
date_df = pd.read_excel(file_path, sheet_name="Date_Wise_Report")
state_df = pd.read_excel(file_path, sheet_name="State_Wise_Report")
district_df = pd.read_excel(file_path, sheet_name="District_Wise_Report")

month_df["month"] = month_df["month"].astype(str)
date_df["date"] = pd.to_datetime(date_df["date"])

# =================================================
# 1️⃣ MONTH-WISE BIOMETRIC ACTIVITY TREND
# =================================================
st.header("1️⃣ Month-wise Biometric Activity Trend")

fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(month_df["month"], month_df["total_biometric"], marker="o")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Biometric Activity")
plt.xticks(rotation=45)
st.pyplot(fig1)

# =================================================
# 2️⃣ AGE-GROUP CONTRIBUTION (STACKED BAR)
# =================================================
st.header("2️⃣ Age-group Contribution by Month")

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.bar(month_df["month"], month_df["total_age_5_17"], label="Age 5–17")
ax2.bar(
    month_df["month"],
    month_df["total_age_17_plus"],
    bottom=month_df["total_age_5_17"],
    label="Age 17+"
)
ax2.set_xlabel("Month")
ax2.set_ylabel("Biometric Activity")
ax2.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)

# =================================================
# 3️⃣ TOP 10 STATES
# =================================================
st.header("3️⃣ Top 10 States by Biometric Activity")

state_df = state_df.dropna(subset=["state"])
top_states = state_df.sort_values("total_biometric", ascending=False).head(10)

fig3, ax3 = plt.subplots(figsize=(9,5))
ax3.barh(top_states["state"], top_states["total_biometric"], color="steelblue")
ax3.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax3.invert_yaxis()
ax3.set_xlabel("Total Biometric Activity")
st.pyplot(fig3)

# =================================================
# 4️⃣ TOP 10 DISTRICTS
# =================================================
st.header("4️⃣ Top 10 Districts by Biometric Activity")

top_districts = district_df.sort_values("total_biometric", ascending=False).head(10)

fig4, ax4 = plt.subplots(figsize=(8,4))
ax4.barh(top_districts["district"], top_districts["total_biometric"])
ax4.invert_yaxis()
ax4.set_xlabel("Total Biometric Activity")
st.pyplot(fig4)

# =================================================
# 5️⃣ ANOMALY DETECTION (DATE-WISE)
# =================================================
st.header("5️⃣ Anomaly Detection (Date-wise Changes)")

date_df["pct_change"] = date_df["total_biometric"].pct_change() * 100
date_df["anomaly"] = date_df["pct_change"].apply(
    lambda x: "Anomaly" if pd.notna(x) and (x > 20 or x < -20) else "Normal"
)

st.dataframe(
    date_df[["date", "total_biometric", "pct_change", "anomaly"]],
    use_container_width=True
)

# =================================================
# 6️⃣ PREDICTIVE ANALYSIS (TREND-BASED FORECAST)
# =================================================
st.header("6️⃣ Predictive Analysis (Next 3 Months Forecast)")

# Trend-based forecasting
month_df["time_index"] = np.arange(len(month_df))
X = month_df["time_index"]
y = month_df["total_biometric"]

coeff = np.polyfit(X, y, 1)
trend_model = np.poly1d(coeff)

future_index = np.arange(len(month_df), len(month_df) + 3)
forecast_values = trend_model(future_index)

forecast_df = pd.DataFrame({
    "Month": ["2026-01", "2026-02", "2026-03"],
    "Forecasted Biometric Activity": forecast_values.astype(int)
})

st.dataframe(forecast_df, use_container_width=True)

# =================================================
# END OF DASHBOARD
# =================================================
