import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA FROM EXCEL
# -----------------------------
file_path = "outputs/UIDAI_All_Reports.xlsx"

month_df = pd.read_excel(file_path, sheet_name="Month_Wise_Report")
date_df = pd.read_excel(file_path, sheet_name="Date_Wise_Report")
state_df = pd.read_excel(file_path, sheet_name="State_Wise_Report")
district_df = pd.read_excel(file_path, sheet_name="District_Wise_Report")

# -----------------------------
# 1️⃣ MONTH-WISE TREND (FIXED)
# -----------------------------
month_df["month"] = month_df["month"].astype(str)

plt.figure(figsize=(8,4))
plt.plot(month_df["month"], month_df["total_biometric"], marker="o")
plt.title("Month-wise Biometric Activity Trend")
plt.xlabel("Month")
plt.ylabel("Total Biometric Activity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# -----------------------------
# 2️⃣ AGE-GROUP CONTRIBUTION (STACKED BAR)
# -----------------------------

plt.figure(figsize=(8,4))
plt.bar(month_df["month"], month_df["total_age_5_17"], label="Age 5–17")
plt.bar(
    month_df["month"],
    month_df["total_age_17_plus"],
    bottom=month_df["total_age_5_17"],
    label="Age 17+"
)
plt.title("Age-group Contribution by Month")
plt.xlabel("Month")
plt.ylabel("Biometric Activity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 3️⃣ TOP 10 STATES
# -----------------------------
import matplotlib.ticker as ticker

# Clean state column (safety)
state_df = state_df.dropna(subset=["state"])
state_df["state"] = state_df["state"].astype(str)

# Select top 10 states
top_states = (
    state_df
    .sort_values("total_biometric", ascending=False)
    .head(10)
)

# Plot
plt.figure(figsize=(9,5))
plt.barh(
    top_states["state"],
    top_states["total_biometric"],
    color="steelblue"
)

plt.title("Top 10 States by Biometric Activity")
plt.xlabel("Total Biometric Activity")
plt.ylabel("State")

# Show full numbers on X-axis
plt.gca().xaxis.set_major_formatter(
    ticker.StrMethodFormatter('{x:,.0f}')
)

# Highest value on top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# -----------------------------
# 4️⃣ TOP 10 DISTRICTS
# -----------------------------
top_districts = district_df.sort_values(
    "total_biometric", ascending=False
).head(10)

plt.figure(figsize=(8,4))
plt.barh(
    top_districts["district"],
    top_districts["total_biometric"]
)
plt.title("Top 10 Districts by Biometric Activity")
plt.xlabel("Total Biometric Activity")
plt.ylabel("District")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------
# 5️⃣ ANOMALY VIEW (DATE-WISE CHANGE)
# -----------------------------
"""date_df["pct_change"] = date_df["total_biometric"].pct_change() * 100

plt.figure(figsize=(8,4))
plt.plot(date_df["date"], date_df["pct_change"], marker="o")
plt.axhline(20, color="red", linestyle="--")
plt.axhline(-20, color="red", linestyle="--")
plt.title("Date-wise Percentage Change (Anomaly Indicator)")
plt.xlabel("Date")
plt.ylabel("Percentage Change (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""

print("✅ All visualizations generated successfully")
