import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------
# 📌 Konfigurasi Halaman
# -----------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered",
    page_icon="🫀"
)

st.title("🫀 Heart Disease Prediction App")
st.write(
    "Aplikasi ini memprediksi potensi penyakit jantung menggunakan Machine Learning "
    "(Random Forest, Logistic Regression, dan Voting Ensemble)."
)

# -----------------------------------------------------------
# 📌 Load model dan preprocessor
# -----------------------------------------------------------
@st.cache_resource
def load_all():
    rf = joblib.load("model_rf.pkl")
    lr = joblib.load("model_lr.pkl")
    voting = joblib.load("model_voting.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    df = pd.read_csv("heart_original.csv")
    return rf, lr, voting, preprocessor, df

rf, lr, voting, preprocessor, df = load_all()

# -----------------------------------------------------------
# 📌 Sidebar — Akurasi model (HASIL ASLI COLAB)
# -----------------------------------------------------------
st.sidebar.header("📊 Model Performance")

st.sidebar.write("**Random Forest Accuracy:** 100.00%")
st.sidebar.write("**Logistic Regression Accuracy:** 80.98%")
st.sidebar.write("**Voting Ensemble Accuracy:** 95.61% ✔")

st.sidebar.info(
    "Akurasi di atas merupakan hasil aktual dari proses training di Google Colab. "
    "Performanya dapat sedikit berbeda jika model dilatih ulang."
)

# -----------------------------------------------------------
# 📌 Input Form
# -----------------------------------------------------------
st.subheader("📝 Input Data Pasien")

# cari kolom target dari dataset
target_col = [
    c for c in df.columns
    if c.lower() in ("target", "heartdisease", "output", "label")
]
target_col = target_col[0] if target_col else df.columns[-1]

# fitur yang digunakan untuk prediksi
feature_cols = [c for c in df.columns if c != target_col]

user_data = {}

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        default_val = float(df[col].median())
        user_data[col] = st.number_input(col, value=default_val)
    else:
        options = df[col].unique().tolist()
        user_data[col] = st.selectbox(col, options)

# -----------------------------------------------------------
# 📌 Pilih model untuk prediksi
# -----------------------------------------------------------
st.subheader("⚙ Pilih Model Prediksi")

model_choice = st.selectbox(
    "Pilih model:",
    ("Voting Ensemble (Rekomendasi)", "Random Forest", "Logistic Regression")
)

# -----------------------------------------------------------
# 📌 Prediksi
# -----------------------------------------------------------
if st.button("🔍 Predict"):
    X_input = pd.DataFrame([user_data])

    # Preprocessing input
    X_transformed = preprocessor.transform(X_input)

    # pilih model
    if model_choice == "Random Forest":
        model = rf
    elif model_choice == "Logistic Regression":
        model = lr
    else:
        model = voting

    # prediksi
    pred = model.predict(X_transformed)[0]

    st.subheader("📢 Hasil Prediksi")

    if pred == 1:
        st.error("💔 Pasien berpotensi memiliki penyakit jantung.")
    else:
        st.success("💚 Pasien tidak memiliki indikasi penyakit jantung.")

    # probabilitas
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_transformed)[0]
        st.write("**Probabilitas:**")
        st.write(f"- Tidak sakit: {prob[0]*100:.2f}%")
        st.write(f"- Sakit: {prob[1]*100:.2f}%")
