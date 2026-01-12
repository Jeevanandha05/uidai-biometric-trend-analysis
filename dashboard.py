import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("UIDAI Biometric Activity Dashboard")

DATA_DIR = Path("data")
files = [
    "biometric_part1.csv",
    "biometric_part2.csv",
    "biometric_part3.csv",
    "biometric_part4.csv"
]

df_list = [pd.read_csv(DATA_DIR / f) for f in files]
data = pd.concat(df_list, ignore_index=True)

data["date"] = pd.to_datetime(data["date"], dayfirst=True)
data["total_biometric"] = data["bio_age_5_17"] + data["bio_age_17_"]

monthly = (
    data
    .groupby(data["date"].dt.to_period("M"))["total_biometric"]
    .sum()
    .reset_index()
)

monthly["date"] = monthly["date"].dt.to_timestamp()

st.subheader("Monthly Biometric Trend")
st.line_chart(monthly.set_index("date"))
