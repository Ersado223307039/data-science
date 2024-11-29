import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Fungsi untuk menguji stasioneritas
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    st.write("Hasil Uji Dickey-Fuller:")
    st.write(f"Test Statistic: {result[0]:.4f}")
    st.write(f"P-Value: {result[1]:.4f}")
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f"{key}: {value:.4f}")

# Judul aplikasi
st.title("Analisis dan Prediksi Data Time Series")

# Membaca file CSV
file_path = "Beras yang Masuk Perbulan Melalui Perum Bulog (Ton).csv"
try:
    df = pd.read_csv(file_path, delimiter=";")
    df.columns = ["bulan", "nilai"]
    df['nilai'] = df['nilai'].astype(np.int32)
    df['bulan'] = pd.to_datetime(df['bulan'], format='%Y-%m')
    df = df.set_index('bulan')

    # Tampilkan data awal
    st.subheader("Data Awal")
    st.write(df.head())

    # Visualisasi awal
    st.subheader("Visualisasi Data")
    plt.figure(figsize=(10, 4))
    plt.plot(df['nilai'], label="Nilai", color="blue")
    plt.title("Data Beras Per Bulan")
    plt.xlabel("Bulan")
    plt.ylabel("Nilai")
    plt.legend()
    st.pyplot(plt)

    # Uji Stasioneritas
    st.subheader("Uji Stasioneritas")
    test_stationarity(df['nilai'])

    # Dekomposisi data
    st.subheader("Dekomposisi Data")
    decomposition = seasonal_decompose(df['nilai'], period=12, model='additive')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

    # Pilih tanggal akhir untuk prediksi
    st.subheader("Parameter Prediksi")
    max_date = df.index.max()
    min_date = df.index.min()
    st.write(f"Tanggal tersedia: {min_date.date()} - {max_date.date()}")
    end_date = st.date_input("Pilih tanggal akhir prediksi:", value=max_date, min_value=min_date, max_value=max_date)

    # Konversi tanggal akhir ke datetime
    end_date = pd.to_datetime(end_date)

    # Prediksi menggunakan ARIMA
    st.subheader("Prediksi Menggunakan ARIMA")
    df_filtered = df[df.index <= end_date]
    size = int(len(df_filtered) - 30)
    train, test = df_filtered['nilai'][:size], df_filtered['nilai'][size:]

    # Build model
    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()

    # Forecasting
    history = list(train)
    predictions = []
    for obs in test:
        model = ARIMA(history, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast()
        predictions.append(forecast[0])
        history.append(obs)

    # Tampilkan hasil prediksi
    predictions_series = pd.Series(predictions, index=test.index)
    st.line_chart({
        "Actual": test.values,
        "Predicted": predictions_series.values
    })

    # Evaluasi Error
    rmse = np.sqrt(mean_squared_error(test, predictions))
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

except FileNotFoundError:
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip ini.")
