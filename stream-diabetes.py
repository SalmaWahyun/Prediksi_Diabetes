import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Judul aplikasi
st.title('Data Mining Prediksi Diabetes')

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.header("Deskripsi Aplikasi")
    st.write("""
    Aplikasi ini dirancang untuk membantu memprediksi risiko diabetes berdasarkan sejumlah variabel kunci.
    Dengan memasukkan data seperti jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, BMI, 
    fungsi diabetes pedigree, dan usia, aplikasi akan memberikan hasil prediksi apakah pasien memiliki risiko diabetes atau tidak.
    Model prediksi ini menggunakan teknologi *Machine Learning* yang telah dilatih pada dataset historis untuk memberikan hasil yang akurat.
    """)

# Fungsi untuk halaman Dataset
def show_dataset():
    st.header("Dataset")
    try:
        # Memuat dataset dari file diabetes.csv
        df = pd.read_csv('diabetes.csv')
        st.dataframe(df)
        st.write("Dataset ini digunakan untuk melatih model prediksi diabetes.")
    except FileNotFoundError:
        st.error("File dataset 'diabetes.csv' tidak ditemukan. Pastikan file tersedia di direktori yang benar.")

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Prediksi Diabetes")
    st.write("Masukkan nilai variabel untuk melakukan prediksi:")

    # Input variabel
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
    insulin = st.number_input("Insulin", min_value=0, step=1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, step=1)

    if st.button("Prediksi"):
        # Prediksi menggunakan model
        diab_prediction = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        if diab_prediction[0] == 1:
            st.success("Hasil: Pasien memiliki risiko diabetes.")
        else:
            st.success("Hasil: Pasien tidak memiliki risiko diabetes.")

# Sidebar untuk navigasi
add_selectbox = st.sidebar.selectbox(
    "Pilih Menu",
    ("Deskripsi", "Dataset", "Prediksi")
)

# Logika navigasi
if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Prediksi":
    show_prediksi()
