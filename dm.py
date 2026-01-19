import pandas as pd
from pathlib import Path

# -----------------------------
# 1. LOAD DATA
# -----------------------------
DATA_DIR = Path("data")

files = [
    "Biometric_part1.csv",
    "Biometric_part2.csv",
    "Biometric_part3.csv",
    "Biometric_part4.csv"
]

df_list = [pd.read_csv(DATA_DIR / f) for f in files]
data = pd.concat(df_list, ignore_index=True)

# -----------------------------
# 2. CLEAN DATE
# -----------------------------
data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")

# -----------------------------
# 3. DATE-WISE TOTAL (NO REPEAT)
# -----------------------------
date_wise_summary = (
    data
    .groupby("date", as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)

# Total biometric
date_wise_summary["total_biometric"] = (
    date_wise_summary["total_age_5_17"]
    + date_wise_summary["total_age_17_plus"]
)

# -----------------------------
# 4. SORT BY DATE
# -----------------------------
date_wise_summary = date_wise_summary.sort_values("date")

# -----------------------------
# 5. SAVE TO EXCEL
# -----------------------------
Path("outputs").mkdir(exist_ok=True)

date_wise_summary.to_excel(
    "outputs/date_wise_total_registration_cleaned.xlsx",
    index=False
)

print("✅ Date-wise Excel created with NO repeated dates")
print(date_wise_summary.head())
