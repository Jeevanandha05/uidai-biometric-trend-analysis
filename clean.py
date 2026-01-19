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
# 2. CLEAN DATE → MONTH
# -----------------------------
data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
data["month"] = data["date"].dt.to_period("M").astype(str)

# -----------------------------
# 3. BASIC STATE CLEANING
# -----------------------------
data["state"] = (
    data["state"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

# -----------------------------
# 4. FIX STATE NAME VARIATIONS
# -----------------------------
state_corrections = {
    "Tamilnadu": "Tamil Nadu",
    "West Bangal": "West Bengal",
    "Westbengal": "West Bengal",
    "West  Bengal": "West Bengal",
    "Orissa": "Odisha",
    "Pondicherry": "Puducherry",
    "Andaman And Nicobar Islands": "Andaman & Nicobar Islands",
    "Jammu And Kashmir": "Jammu & Kashmir",
    "Daman And Diu": "Daman & Diu",
    "Dadra And Nagar Haveli": "Dadra & Nagar Haveli",
    "Dadra And Nagar Haveli And Daman And Diu":
        "Dadra & Nagar Haveli And Daman & Diu",
    "Uttaranchal": "Uttarakhand",
    "Chhatisgarh": "Chhattisgarh"
}

data["state"] = data["state"].replace(state_corrections)

# -----------------------------
# 5. CLEAN DISTRICT
# -----------------------------
data["district"] = (
    data["district"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

# -----------------------------
# 6. SELECT REQUIRED COLUMNS
# -----------------------------
cleaned_data = data[
    ["month", "state", "district", "bio_age_5_17", "bio_age_17_"]
]

# -----------------------------
# 7. STATE-WISE TOTAL REGISTRATION
# -----------------------------
state_summary = (
    cleaned_data
    .groupby("state", as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)

# -----------------------------
# 8. DISTRICT-WISE TOTAL REGISTRATION
# -----------------------------
district_summary = (
    cleaned_data
    .groupby(["state", "district"], as_index=False)
    .agg(
        total_age_5_17=("bio_age_5_17", "sum"),
        total_age_17_plus=("bio_age_17_", "sum")
    )
)

# -----------------------------
# 9. SAVE OUTPUT FILES
# -----------------------------
Path("outputs").mkdir(exist_ok=True)

state_summary.to_csv(
    "outputs/state_total_registration_cleaned.csv", index=False
)

district_summary.to_csv(
    "outputs/district_total_registration_cleaned.csv", index=False
)

# -----------------------------
# 10. CONFIRMATION PRINT
# -----------------------------
print("State-wise rows:", state_summary.shape[0])
print("District-wise rows:", district_summary.shape[0])
print("\nSample State-wise Output:")
print(state_summary.head())
# -----------------------------
# 11. CONVERT CSV → EXCEL
# -----------------------------
state_summary.to_excel(
    "outputs/state_total_registration_cleaned.xlsx",
    index=False
)

district_summary.to_excel(
    "outputs/district_total_registration_cleaned.xlsx",
    index=False
)

print("Excel files created successfully!")
