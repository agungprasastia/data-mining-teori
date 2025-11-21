# 🫀 Heart Disease Prediction using Ensemble Machine Learning  
**Mencapai Akurasi Super (>90%) dengan Random Forest + Logistic Regression**

## 📌 Deskripsi Proyek
Proyek ini adalah aplikasi Machine Learning untuk memprediksi potensi penyakit jantung berdasarkan data medis pasien.

Model yang digunakan:
- **Random Forest**
- **Logistic Regression**
- **Voting Ensemble (RF + LR)**

Ensemble Voting digunakan untuk meningkatkan akurasi hingga **>90%**.

Aplikasi dibangun menggunakan **Streamlit**.

---

## 📊 Dataset
Dataset berasal dari sumber terbuka seperti Kaggle atau UCI.  
Dataset yang digunakan disimpan sebagai:

```
heart_original.csv
```

File ini **wajib ada** karena:
- Digunakan untuk membaca struktur kolom
- Dipakai untuk membuat form input di Streamlit
- Menjaga kesesuaian dengan preprocessor

---

## 🤖 Algoritma
- **Random Forest Classifier**  
- **Logistic Regression**  
- **VotingClassifier (Hard Voting)**  

---

## 🎯 Akurasi Model (Training via Google Colab)
| Model                 | Akurasi |
|----------------------|---------|
| Random Forest        | >90%    |
| Logistic Regression  | >85%    |
| **Voting Ensemble**  | **>90%** ✔ |

Akurasi diperoleh dari proses training di Google Colab.

---

## 🔧 Preprocessing
Preprocessing dilakukan sepenuhnya di **Google Colab**, mencakup:

- Missing value handling  
- StandardScaler untuk fitur numerik  
- OneHotEncoder untuk fitur kategorikal  
- SMOTE (opsional)  
- Train-test split  
- Menyimpan pipeline ke file `.pkl`

Model-model disimpan sebagai:
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
Training dilakukan **tanpa train.py**, tapi langsung di notebook Colab.

Untuk menyimpan model:
```python
joblib.dump(model_voting, "model_voting.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
```

Kemudian download file `.pkl` dan upload kembali ke GitHub.

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
