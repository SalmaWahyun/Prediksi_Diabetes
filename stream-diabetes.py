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
    # Contoh dataset buatan jika tidak ada file dataset
    df = pd.DataFrame({
        "Pregnancies": np.random.randint(0, 10, 100),
        "Glucose": np.random.randint(80, 200, 100),
        "BloodPressure": np.random.randint(60, 120, 100),
        "SkinThickness": np.random.randint(10, 50, 100),
        "Insulin": np.random.randint(15, 200, 100),
        "BMI": np.random.uniform(18.0, 40.0, 100),
        "DiabetesPedigreeFunction": np.random.uniform(0.1, 2.5, 100),
        "Age": np.random.randint(20, 80, 100),
        "Outcome": np.random.choice([0, 1], 100)
    })
    st.dataframe(df)
    st.write("Dataset ini hanya contoh. Pastikan menggunakan dataset aktual untuk prediksi yang valid.")

# Fungsi untuk halaman Grafik
def show_grafik():
    st.header("Grafik Data")
    # Contoh grafik buatan
    df = pd.DataFrame({
        "Glucose": np.random.randint(80, 200, 100),
        "BMI": np.random.uniform(18.0, 40.0, 100),
        "Age": np.random.randint(20, 80, 100),
    })
    st.line_chart(df)

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
    ("Deskripsi", "Dataset", "Grafik", "Prediksi")
)

# Logika navigasi
if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Grafik":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()
