import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. LOAD DATA

DATA_DIR = Path("data")

files = [
    "Biometric_part1.csv",
    "Biometric_part2.csv",
    "Biometric_part3.csv",
    "Biometric_part4.csv"
]

df_list = [pd.read_csv(DATA_DIR / f) for f in files]
data = pd.concat(df_list, ignore_index=True)

# 2. PREPARE DATA

data["date"] = pd.to_datetime(data["date"], dayfirst=True)

# total biometric activity (all ages)
data["total_biometric"] = data["bio_age_5_17"] + data["bio_age_17_"]


# 3. MONTHLY TREND (TREND)

monthly_trend = (
    data
    .groupby(data["date"].dt.to_period("M"))["total_biometric"] # Pattern A month-wise 
    .sum()
    .reset_index()
)
monthly_trend["date"] = monthly_trend["date"].dt.to_timestamp()
monthly_trend = monthly_trend.sort_values("date")

# 4. PATTERNS

# Pattern B: Age-group dominance

monthly_age = (
    data
    .groupby(data["date"].dt.to_period("M"))[["bio_age_5_17", "bio_age_17_"]]
    .sum()
    .reset_index()
)
monthly_age["date"] = monthly_age["date"].dt.to_timestamp()


# 5. ANOMALY DETECTION

monthly_trend["pct_change"] = monthly_trend["total_biometric"].pct_change() * 100

# flag anomalies using ±20% threshold
monthly_trend["anomaly_flag"] = monthly_trend["pct_change"].apply(
    lambda x: "Anomaly" if (pd.notna(x) and (x > 20 or x < -20)) else "Normal"
)

# 6. PREDICTIVE INDICATOR (MOVING AVERAGE)

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


# 7. PLOTS

Path("plots").mkdir(exist_ok=True)

# Trend + Forecast

plt.figure(figsize=(10, 5))
plt.plot(monthly_trend["date"], monthly_trend["total_biometric"], label="Actual")
plt.plot(forecast["date"], forecast["forecast_biometric"],
         linestyle="--", marker="o", label="Forecast (MA)")
plt.xlabel("Month")
plt.ylabel("Total Biometric Activity")
plt.title("Aadhaar Biometric Activity: Trend & Forecast")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/trend_forecast.png", dpi=300)
plt.show()

# Age-group pattern

plt.figure(figsize=(10, 5))
plt.plot(monthly_age["date"], monthly_age["bio_age_5_17"], label="Age 5–17")
plt.plot(monthly_age["date"], monthly_age["bio_age_17_"], label="Age 17+")
plt.xlabel("Month")
plt.ylabel("Biometric Activity")
plt.title("Age-group Pattern in Biometric Activity")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/age_group_pattern.png", dpi=300)
plt.show()


# 8. OUTPUT TABLES 

print("\nMonthly Trend with Anomaly Flags:")
print(monthly_trend[["date", "total_biometric", "pct_change", "anomaly_flag"]])

print("\nForecast (Next 3 Months):")
print(forecast)
