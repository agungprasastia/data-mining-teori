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
st.write("Aplikasi ini memprediksi potensi penyakit jantung menggunakan Ensemble Machine Learning (RandomForest + Logistic Regression).")

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
# 📌 Sidebar — Akurasi model
# -----------------------------------------------------------
st.sidebar.header("📊 Model Performance")

st.sidebar.write("Random Forest Accuracy: **>90%**")
st.sidebar.write("Logistic Regression Accuracy: **>85%**")
st.sidebar.write("Voting Ensemble Accuracy: **>90%** ✔")

st.sidebar.info("Akurasi aktual dapat berbeda tergantung data training, silakan cek laporan train.py.")

# -----------------------------------------------------------
# 📌 Input Form
# -----------------------------------------------------------
st.subheader("📝 Input Data Pasien")

target_col = [c for c in df.columns if c.lower() in ("target", "heartdisease", "output")]
target_col = target_col[0] if target_col else df.columns[-1]

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
# 📌 Pilih model
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
    X_transformed = preprocessor.transform(X_input)

    if model_choice == "Random Forest":
        model = rf
    elif model_choice == "Logistic Regression":
        model = lr
    else:
        model = voting

    pred = model.predict(X_transformed)[0]

    st.subheader("📢 Hasil Prediksi")

    if pred == 1:
        st.error("💔 **Pasien berpotensi memiliki penyakit jantung.**")
    else:
        st.success("💚 **Pasien tidak memiliki indikasi penyakit jantung.**")

    # Probabilities if available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_transformed)[0]
        st.write(f"**Probabilitas:**")
        st.write(f"- Tidak sakit: {prob[0]*100:.2f}%")
        st.write(f"- Sakit: {prob[1]*100:.2f}%")
