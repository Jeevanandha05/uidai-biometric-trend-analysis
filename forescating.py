import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MONTH-WISE DATA
# -----------------------------
file_path = "outputs/UIDAI_All_Reports.xlsx"
df = pd.read_excel(file_path, sheet_name="Month_Wise_Report")

# Clean data
df = df.dropna(subset=["month"])
df = df[df["month"] != "nan"]
df = df.sort_values("month").reset_index(drop=True)

# -----------------------------
# PREPARE TREND DATA
# -----------------------------
df["time_index"] = np.arange(len(df))

X = df["time_index"]
y = df["total_biometric"]

# -----------------------------
# FIT LINEAR TREND
# -----------------------------
coefficients = np.polyfit(X, y, 1)
trend_line = np.poly1d(coefficients)

# -----------------------------
# FORECAST NEXT 3 MONTHS
# -----------------------------
future_index = np.arange(len(df), len(df) + 3)
forecast_values = trend_line(future_index)

forecast_df = pd.DataFrame({
    "month": ["2026-01", "2026-02", "2026-03"],
    "forecast_biometric": forecast_values.astype(int)
})

print("📊 Forecasted Values (Trend-based):")
print(forecast_df)

# -----------------------------
# VISUALIZE FORECAST
# -----------------------------
plt.figure(figsize=(8,4))

plt.plot(
    df["month"],
    df["total_biometric"],
    marker="o",
    label="Actual"
)

plt.plot(
    df["month"],
    trend_line(df["time_index"]),
    linestyle="--",
    label="Trend Line"
)

plt.plot(
    forecast_df["month"],
    forecast_df["forecast_biometric"],
    marker="o",
    linestyle=":",
    label="Forecast (2026)"
)

plt.title("Biometric Activity Forecast using Trend-based Method")
plt.xlabel("Month")
plt.ylabel("Total Biometric Activity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
