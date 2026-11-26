import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="SIPREDI - Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered"
)

# Memuat model Machine Learning
@st.cache_resource
def load_model():
    artifact = joblib.load("model_harga_rumah.pkl")
    return artifact

artifact = load_model()

model = artifact['model']
feature_names = artifact['feature_names']
model_name = artifact['model_name']


# Judul aplikasi
st.title("🏠 SIPREDI")

st.subheader("Sistem Prediksi Harga Rumah")

st.write(
    "SIPREDI merupakan aplikasi Machine Learning "
    "yang digunakan untuk memperkirakan harga rumah "
    "berdasarkan karakteristik rumah."
)

st.divider()


# Informasi algoritma
st.info(
    f"Algoritma yang digunakan: {model_name}"
)


# Input data rumah
st.subheader("Masukkan Data Rumah")

luas_tanah = st.number_input(
    "Luas Tanah (m²)",
    min_value=1,
    value=100,
    step=1
)

luas_bangunan = st.number_input(
    "Luas Bangunan (m²)",
    min_value=1,
    value=90,
    step=1
)

kamar_tidur = st.number_input(
    "Jumlah Kamar Tidur",
    min_value=1,
    value=3,
    step=1
)

kapasitas_parkir = st.number_input(
    "Kapasitas Parkir",
    min_value=1,
    value=2,
    step=1
)


# Tombol prediksi
if st.button(
    "Prediksi Harga Rumah",
    use_container_width=True
):

    # Feature Engineering
    rasio = luas_bangunan / luas_tanah

    # Membuat DataFrame input
    input_data = pd.DataFrame([{
        'luas tanah': luas_tanah,
        'luas bangunan': luas_bangunan,
        'kamar tidur': kamar_tidur,
        'kapasitas parkir': kapasitas_parkir,
        'rasio_bangunan_tanah': rasio
    }])

    # Menyesuaikan urutan fitur
    input_data = input_data[feature_names]

    # Melakukan prediksi
    hasil_prediksi = model.predict(input_data)[0]

    # Menampilkan hasil
    st.success("Prediksi harga rumah berhasil!")

    st.subheader("Hasil Prediksi")

    st.metric(
        label="Estimasi Harga Rumah",
        value=f"Rp {hasil_prediksi:,.0f}"
    )

    st.caption(
        "Hasil prediksi merupakan estimasi berdasarkan "
        "pola data yang digunakan dalam pelatihan model."
    )


st.divider()

st.caption(
    "SIPREDI - Project Machine Learning"
)
