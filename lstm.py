import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# 1. LOAD MONTH-WISE DATA
# -----------------------------
data = pd.read_excel(
    "outputs/UIDAI_All_Reports.xlsx",
    sheet_name="Month_Wise_Report"
)

values = data["total_biometric"].values.reshape(-1, 1)

# -----------------------------
# 2. NORMALIZE DATA
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)

# -----------------------------
# 3. CREATE SEQUENCES
# -----------------------------
def create_sequences(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window=3)

# -----------------------------
# 4. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    LSTM(50, activation="relu", input_shape=(X.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
model.fit(X, y, epochs=50, verbose=1)

print("✅ LSTM model trained successfully")

# -----------------------------
# 6. MAKE PREDICTIONS
# -----------------------------
predicted_scaled = model.predict(X)

# Convert back to original scale
predicted = scaler.inverse_transform(predicted_scaled)
actual = scaler.inverse_transform(y)

print("Sample Predictions vs Actual:")
for i in range(5):
    print(f"Predicted: {int(predicted[i][0])}, Actual: {int(actual[i][0])}")

import matplotlib.pyplot as plt

# -----------------------------
# 7. PLOT RESULTS
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual Biometric Activity")
plt.plot(predicted, label="LSTM Prediction")
plt.title("Actual vs LSTM Predicted Biometric Activity")
plt.xlabel("Time")
plt.ylabel("Biometric Activity")
plt.legend()
plt.show()

