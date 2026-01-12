import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 1. DATA LOADING
# -----------------------------
DATA_DIR = Path("data")

files = [
    "Biometric_part1.csv",
    "Biometric_part2.csv",
    "Biometric_part3.csv",
    "Biometric_part4.csv"
]

df_list = []
for file in files:
    df = pd.read_csv(DATA_DIR / file)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

# -----------------------------
# 2. DATA PREPARATION
# -----------------------------
# Convert date column
data["date"] = pd.to_datetime(data["date"], dayfirst=True)

# Total biometric activity
data["total_biometric"] = (
    data["bio_age_5_17"] + data["bio_age_17_"]
)

# -----------------------------
# 3. MONTHLY TREND AGGREGATION
# -----------------------------
monthly_trend = (
    data
    .groupby(data["date"].dt.to_period("M"))["total_biometric"]
    .sum()
    .reset_index()
)

monthly_trend["date"] = monthly_trend["date"].dt.to_timestamp()
monthly_trend = monthly_trend.sort_values("date")

# -----------------------------
# 4. SIMPLE FORECAST (MOVING AVERAGE)
# -----------------------------
window = 3  # last 3 months
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

# -----------------------------
# 5. PLOT: ACTUAL + FORECAST
# -----------------------------
Path("plots").mkdir(exist_ok=True)

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_trend["date"],
    monthly_trend["total_biometric"],
    label="Actual Biometric Activity",
    linewidth=2
)

plt.plot(
    forecast["date"],
    forecast["forecast_biometric"],
    linestyle="--",
    marker="o",
    label="Forecast (Moving Average)"
)

plt.xlabel("Month")
plt.ylabel("Total Biometric Activity")
plt.title("Aadhaar Biometric Activity Trend with Forecast")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save for PPT / GitHub
plt.savefig("plots/biometric_trend_with_forecast.png", dpi=300)

plt.show()

# -----------------------------
# 6. PRINT SUMMARY (OPTIONAL)
# -----------------------------
print("Monthly Trend:")
print(monthly_trend[["date", "total_biometric"]])

print("\nForecast (Next 3 Months):")
print(forecast)
