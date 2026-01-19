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
# 2. BASIC CLEANING
# -----------------------------
data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
data["month"] = data["date"].dt.to_period("M").astype(str)

data["state"] = (
    data["state"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

data["district"] = (
    data["district"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

# -----------------------------
# 3. MONTH-WISE REPORT
# -----------------------------
month_wise = (
    data
    .groupby("month", as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)
month_wise["total_biometric"] = (
    month_wise["total_age_5_17"] + month_wise["total_age_17_plus"]
)

# -----------------------------
# 4. DATE-WISE REPORT
# -----------------------------
date_wise = (
    data
    .groupby("date", as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)
date_wise["total_biometric"] = (
    date_wise["total_age_5_17"] + date_wise["total_age_17_plus"]
)
date_wise = date_wise.sort_values("date")

# -----------------------------
# 5. STATE-WISE REPORT
# -----------------------------
state_wise = (
    data
    .groupby("state", as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)
state_wise["total_biometric"] = (
    state_wise["total_age_5_17"] + state_wise["total_age_17_plus"]
)

# -----------------------------
# 6. DISTRICT-WISE REPORT
# -----------------------------
district_wise = (
    data
    .groupby(["state", "district"], as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)
district_wise["total_biometric"] = (
    district_wise["total_age_5_17"] + district_wise["total_age_17_plus"]
)

# -----------------------------
# 7. SAVE ALL INTO ONE EXCEL FILE (MULTI-SHEET)
# -----------------------------
Path("outputs").mkdir(exist_ok=True)

output_file = "outputs/UIDAI_All_Reports.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    month_wise.to_excel(writer, sheet_name="Month_Wise_Report", index=False)
    date_wise.to_excel(writer, sheet_name="Date_Wise_Report", index=False)
    state_wise.to_excel(writer, sheet_name="State_Wise_Report", index=False)
    district_wise.to_excel(writer, sheet_name="District_Wise_Report", index=False)

print("✅ All reports combined into ONE Excel file successfully!")
print("📂 File:", output_file)
