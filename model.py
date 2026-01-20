import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

DATA_DIR = Path("data")

files = [
    "Biometric_part1.csv",
    "Biometric_part2.csv",
    "Biometric_part3.csv",
    "Biometric_part4.csv"
]

df = pd.concat([pd.read_csv(DATA_DIR / f) for f in files], ignore_index=True)

df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

monthly = (
    df.groupby(pd.Grouper(key='date', freq='ME'))[
        ['bio_age_5_17', 'bio_age_17_']
    ]
    .sum()
    .reset_index()
)

monthly['total_bio'] = monthly['bio_age_5_17'] + monthly['bio_age_17_']
print(monthly.head())
print(monthly.isna().sum())

if len(monthly) < 12:
    print("⚠️ Warning: Data insufficient for reliable LSTM training (need 12+ months).")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(monthly[['total_bio']])

def create_sequences(data, window=6):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, window=6)

if len(X) == 0:
    print("❌ Error: Not enough data to create sequences.")
    exit()

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

rnn_model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(6, 1)),
    Dense(1)
])

rnn_model.compile(
    optimizer='adam',
    loss='mse'
)

rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

lstm_model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(6, 1)),
    Dense(1)
])

lstm_model.compile(
    optimizer='adam',
    loss='mse'
)

lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=8,
    validation_data=(X_test, y_test)
)

def forecast_next(model, last_sequence, steps=3):
    preds = []
    seq = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(seq.reshape(1, 6, 1), verbose=0)
        preds.append(pred[0,0])
        seq = np.append(seq[1:], pred)

    return scaler.inverse_transform(np.array(preds).reshape(-1,1))

last_6_months = scaled[-6:]
if len(last_6_months) < 6:
    print("⚠️ Not enough data to generate forecast (need at least 6 months).")
    rnn_forecast = np.array([])
    lstm_forecast = np.array([])
else:
    rnn_forecast = forecast_next(rnn_model, last_6_months)
    lstm_forecast = forecast_next(lstm_model, last_6_months)
    print("RNN Forecast:", rnn_forecast.flatten())
    print("LSTM Forecast:", lstm_forecast.flatten())

# Ensure plots directory exists
Path("plots").mkdir(exist_ok=True)
print("Saving plots to 'plots/' directory...")

fig_rnn = plt.figure()
plt.plot(rnn_history.history['loss'], label='Training Loss')
plt.plot(rnn_history.history['val_loss'], label='Validation Loss')
plt.title('RNN Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig("plots/rnn_loss.png")
st.pyplot(fig_rnn)

fig_lstm = plt.figure()
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig("plots/lstm_loss.png")
st.pyplot(fig_lstm)

if len(lstm_forecast) > 0:
    future_dates = pd.date_range(
        start=monthly['date'].iloc[-1] + pd.offsets.MonthEnd(1),
        periods=3,
        freq='ME'
    )
    fig_forecast = plt.figure()
    plt.plot(monthly['date'], monthly['total_bio'], label='Historical Data')
    plt.plot(future_dates, lstm_forecast.flatten(), label='Forecast (Next 3 Months)')
    plt.title('LSTM Forecast for Aadhaar Biometric Transactions')
    plt.xlabel('Date')
    plt.ylabel('Total Biometric Transactions')
    plt.legend()
    plt.savefig("plots/forecast.png")
    st.pyplot(fig_forecast)
