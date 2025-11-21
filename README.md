# 🫀 Heart Disease Prediction using Ensemble Machine Learning  
**Akurasi Model: RF 100.00% | LR 80.98% | Voting Ensemble 95.61%**

## 📌 Deskripsi Proyek
Proyek ini adalah aplikasi Machine Learning untuk memprediksi potensi penyakit jantung berdasarkan data medis pasien.

Model yang digunakan:
- **Random Forest**  
- **Logistic Regression**  
- **Voting Ensemble (RF + LR)**  

Ensemble Voting digunakan untuk meningkatkan akurasi hingga **95.61%**.

Aplikasi dibangun menggunakan **Streamlit**.

---

## 🔗 Google Colab (Training Notebook)
Model dan preprocessing **dilatih sepenuhnya di Google Colab**:  
👉 **https://colab.research.google.com/drive/1BxnGwSpRW-6RUOE_y1sR9ALR642IvDOn?usp=sharing**

Silakan buka link di atas untuk melihat:
- Proses preprocessing
- Confusion matrix
- Akurasi asli
- Pelatihan Random Forest, Logistic Regression, dan Voting Ensemble

---

## 📊 Dataset
Dataset berasal dari sumber terbuka seperti Kaggle atau UCI.  
Dataset disimpan sebagai:

```
heart_original.csv
```

File ini **wajib ada** karena:
- Digunakan untuk membaca struktur kolom
- Dipakai untuk membuat form input di Streamlit
- Menjaga kesesuaian dengan preprocessor

---

## 🤖 Algoritma & Akurasi Model (Asli dari Google Colab)

### **1️⃣ Random Forest**
- **Akurasi: 100.00%**
- Tidak ada kesalahan prediksi (Confusion Matrix sempurna)

### **2️⃣ Logistic Regression**
- **Akurasi: 80.98%**
- Performa standar, lebih rendah dari RF (wajar untuk dataset ini)

### **3️⃣ Voting Ensemble (RF + LR)**
- **Akurasi: 95.61%**
- Model terbaik  
- Menggabungkan kekuatan RF & LR  
- Sesuai ketentuan tugas: **Akurasi Super (>90%) ✔**

---

## 🔧 Preprocessing
Preprocessing dilakukan sepenuhnya di **Google Colab**, mencakup:

- Handling Missing Values  
- StandardScaler untuk fitur numerik  
- OneHotEncoder untuk fitur kategorikal  
- SMOTE (opsional)  
- Train-test split  
- Penyimpanan pipeline ke `preprocessor.pkl`

Model disimpan sebagai:
```
model_rf.pkl
model_lr.pkl
model_voting.pkl
preprocessor.pkl
```

---

## 📁 Struktur Folder
```
📦 heart-disease-prediction
│── main.py               
│── requirements.txt
│── heart_original.csv
│── model_rf.pkl
│── model_lr.pkl
│── model_voting.pkl
│── preprocessor.pkl
└── README.md
```

---

## 🚀 Cara Menjalankan Aplikasi
### Install dependency:
```
pip install -r requirements.txt
```

### Jalankan Streamlit:
```
streamlit run main.py
```

---

## 📝 Training Ulang Model (di Google Colab)
Training dilakukan di notebook Colab.  
Untuk menyimpan model:

```python
joblib.dump(model_voting, "model_voting.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
```

Kemudian upload file `.pkl` ke GitHub.

---

## 🧪 Teknologi yang Digunakan
- Python  
- Pandas  
- Scikit-Learn  
- Imbalanced-Learn  
- Streamlit  
- Joblib  

---

## ✨ Author
**Agung Prasasti Abadi**  
Proyek Machine Learning — Heart Disease Prediction
