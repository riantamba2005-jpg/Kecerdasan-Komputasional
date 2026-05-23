# Sistem Prediksi Risiko Diabetes Melitus Berbasis Web

## Deskripsi Project

Proyek ini membangun sistem prediksi risiko diabetes melitus menggunakan pendekatan supervised learning dengan Artificial Neural Network (ANN) dan algoritma Backpropagation.

Model dilatih menggunakan dataset "Pima Indians Diabetes Dataset" dari Kaggle dan diintegrasikan ke dalam aplikasi web menggunakan Streamlit.

## Tujuan Project

- Mengimplementasikan konsep Computational Intelligence pada ANN.
- Menggunakan algoritma Backpropagation untuk training model.
- Membangun aplikasi web yang user friendly untuk prediksi risiko diabetes.
- Menyediakan pipeline preprocessing, training, evaluasi, dan visualisasi.

## Teknologi yang Digunakan

- Python 3
- Scikit-Learn
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## Dataset

Dataset utama: **Pima Indians Diabetes Dataset**

Karakteristik dataset:
- Jumlah data: 768 baris
- Fitur: 8 input
- Target: 1 output (`Outcome`)
- Distribusi kelas: sekitar 500 pasien negatif (tidak diabetes) dan 268 pasien positif (diabetes)

Fitur/input:
- `Pregnancies`: Jumlah kehamilan
- `Glucose`: Kadar glukosa dalam darah (mg/dL)
- `BloodPressure`: Tekanan darah diastolik (mm Hg)
- `SkinThickness`: Ketebalan lipatan kulit trisep (mm)
- `Insulin`: Konsentrasi insulin serum (mu U/ml)
- `BMI`: Indeks massa tubuh (kg/m²)
- `DiabetesPedigreeFunction`: Fungsi kepedulian genetik diabetes
- `Age`: Usia (tahun)

Target/output:
- `Outcome`: 0 = tidak diabetes, 1 = diabetes

## Proses Utama

1. Preprocessing data:
   - load dataset dari `data/pima_diabetes.csv`
   - membersihkan kolom dengan nilai 0 yang tidak valid
   - mengganti nilai kosong dengan median kolom
   - normalisasi fitur menggunakan StandardScaler

2. Data cleaning dan handling missing values:
   - Mengganti nilai 0 pada fitur medis yang memungkinkan sebagai missing value
   - Imputasi median untuk menjaga distribusi data

3. Splitting data:
   - Memisahkan data menjadi training dan testing dengan rasio 70:30

4. Arsitektur Neural Network:
   - Input layer: 8 fitur
   - Hidden layer 1: 16 neuron, ReLU
   - Hidden layer 2: 12 neuron, ReLU
   - Hidden layer 3: 8 neuron, ReLU
   - Output layer: 1 neuron, Sigmoid

5. Training model:
   - Optimizer SGD dengan learning rate 0.01
   - Loss function: binary crossentropy
   - Algoritma Backpropagation pada ANN Scikit-Learn untuk update weight dan minimisasi error

6. Evaluasi model:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix

7. Visualisasi:
   - Training loss
   - Accuracy graph
   - Confusion matrix
   - Distribusi fitur dataset

## Struktur Folder

```
Kecerdasan-Komputasional/
├── data/
│   └── pima_diabetes.csv       # letakkan dataset ini sendiri
├── models/
│   └── diabetes_ann.pkl       # disimpan setelah training
├── reports/
│   └── training_history.png
│   └── confusion_matrix.png
│   └── feature_distribution.png
│   └── evaluation_report.txt
├── src/
│   ├── __init__.py
│   ├── app.py                 # aplikasi web Streamlit
│   ├── model.py               # pipeline ANN dan evaluasi
│   ├── train.py               # script training lengkap
├── requirements.txt
├── .gitignore
└── README.md
```

## Cara Instalasi

1. Buat virtual environment (opsional):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Siapkan dataset:
   - Download dari Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
   - Simpan sebagai `data/pima_diabetes.csv` atau `data/diabetes.csv`

1. Training model:
   ```bash
   python src/train.py
   ```
2. Jalankan aplikasi web Streamlit:
   ```bash
   streamlit run src/app.py
   ```

## Hasil Evaluasi Model

Setelah menjalankan `src/train.py`, hasil metrik evaluasi akan tersimpan di `reports/evaluation_report.txt`.

Contoh metrik yang akan dihasilkan:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Penggunaan Aplikasi

1. Jalankan `streamlit run src/app.py`.
2. Isi form di sidebar dengan nilai kesehatan pengguna.
3. Klik tombol "Prediksi Risiko Diabetes".
4. Aplikasi akan menampilkan hasil prediksi dan probabilitas risiko.

## Catatan Konsep Computational Intelligence

Model ini menggunakan Artificial Neural Network sebagai metode utama. Backpropagation memproses error dari output kembali ke hidden layer, lalu mengupdate bobot (weight update) dengan gradient descent dan learning rate untuk meminimalkan fungsi loss.

---

### Penjelasan Atribut Dataset

- `Pregnancies`: jumlah kehamilan sebelumnya.
- `Glucose`: pengukuran gula darah; nilai tinggi menunjukkan kemungkinan resistensi insulin.
- `BloodPressure`: tekanan darah; tekanan tinggi dapat berhubungan dengan komplikasi diabetes.
- `SkinThickness`: indikasi persentase lemak tubuh.
- `Insulin`: kadar insulin darah; penting untuk mendeteksi gangguan metabolisme.
- `BMI`: indeks massa tubuh; nilai tinggi sering berkaitan dengan risiko diabetes.
- `DiabetesPedigreeFunction`: faktor genetik; nilai lebih tinggi mengindikasikan riwayat keluarga lebih kuat.
- `Age`: usia pasien; risiko diabetes meningkat seiring bertambahnya usia.
- `Outcome`: kelas target, 0 = tidak diabetes, 1 = diabetes.
