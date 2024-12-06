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

# Sidebar untuk navigasi dengan dropdown
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Deskripsi", "View Dataset", "Visualisasi Data", "Uji Stasioneritas", "Dekomposisi", "Prediksi", "Overview ARIMA"])

# Membaca file CSV
file_path = "Beras yang Masuk Perbulan Melalui Perum Bulog (Ton).csv"
try:
    df = pd.read_csv(file_path, delimiter=";")
    df.columns = ["bulan", "nilai"]
    df['nilai'] = df['nilai'].astype(np.int32)
    df['bulan'] = pd.to_datetime(df['bulan'], format='%Y-%m')
    df = df.set_index('bulan')

    if page == "Deskripsi":
        st.subheader("Deskripsi Aplikasi")
        st.write("""
        Aplikasi ini digunakan untuk menganalisis dan memprediksi data time series terkait **Jumlah Beras yang Masuk Per Bulan melalui Perum Bulog (Ton)**. 
        Data ini digunakan untuk memprediksi jumlah beras yang masuk pada bulan berikutnya dengan menggunakan model ARIMA.
        """)

        st.subheader("Informasi Dataset")
        st.write("""
        - **Nama Pembuat**: Ersado Cahya Buana
        - **Sumber Dataset**: Dataset ini diambil dari data internal Perum Bulog, yang berfokus pada distribusi beras yang masuk per bulan di Indonesia. 
          Anda dapat mengakses dataset ini di [Sumber Dataset](https://www.kaggle.com/code/sandi07/prediksi-beras-lokal-yang-masuk-perbulan/input).
        """)

    elif page == "View Dataset":
        st.subheader("View Dataset")
        st.write(df.head())

    elif page == "Visualisasi Data":
        st.subheader("Visualisasi Data")
        plt.figure(figsize=(10, 4))
        plt.plot(df['nilai'], label="Nilai", color="blue")
        plt.title("Data Beras Per Bulan")
        plt.xlabel("Bulan")
        plt.ylabel("Nilai")
        plt.legend()
        st.pyplot(plt)

    elif page == "Uji Stasioneritas":
        st.subheader("Uji Stasioneritas")
        test_stationarity(df['nilai'])

    elif page == "Dekomposisi":
        st.subheader("Dekomposisi Data")
        decomposition = seasonal_decompose(df['nilai'], period=12, model='additive')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        decomposition.observed.plot(ax=ax1, title='Observed')
        decomposition.trend.plot(ax=ax2, title='Trend')
        decomposition.seasonal.plot(ax=ax3, title='Seasonal')
        decomposition.resid.plot(ax=ax4, title='Residual')
        plt.tight_layout()
        st.pyplot(fig)

    elif page == "Prediksi":
        st.subheader("Parameter Prediksi")
        max_date = df.index.max()
        min_date = df.index.min()
        st.write(f"Tanggal tersedia: {min_date.date()} - {max_date.date()}")
        end_date = st.date_input("Pilih tanggal akhir prediksi:", value=max_date, min_value=min_date, max_value=max_date)

        # Konversi tanggal akhir ke datetime
        end_date = pd.to_datetime(end_date)

        # Prediksi menggunakan ARIMA
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
        st.subheader("Hasil Prediksi")
        st.line_chart({
            "Actual": test.values,
            "Predicted": predictions_series.values
        })

        # Evaluasi Error
        rmse = np.sqrt(mean_squared_error(test, predictions))
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    elif page == "Overview ARIMA":
        st.subheader("Overview Algoritma ARIMA")
        st.write("""
        ARIMA (AutoRegressive Integrated Moving Average) adalah salah satu algoritma yang digunakan untuk memodelkan time series. 
        Algoritma ini terdiri dari tiga komponen utama:
        - **AR (AutoRegressive)**: Model ini menggunakan hubungan antara observasi dengan lag-nya.
        - **I (Integrated)**: Proses diferensiasi digunakan untuk membuat data menjadi stasioner.
        - **MA (Moving Average)**: Model ini mengukur pengaruh kesalahan pada model sebelumnya.

        Sebagai contoh, jika kita memprediksi jumlah beras yang masuk untuk bulan berikutnya, ARIMA memanfaatkan data historis untuk menghasilkan prediksi dengan memperhatikan pola musiman dan tren yang ada dalam data.
        """)

except FileNotFoundError:
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip ini.")
