# 🤫 Sistem Deteksi Emosi Multikategori Komentar YouTube Berbasis IndoBERT & Streamlit

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

Aplikasi web interaktif untuk analisis sentimen & **deteksi emosi multikategori** pada komentar video YouTube bahasa Indonesia. Sistem ini dikembangkan menggunakan model Transformer **IndoBERT** (`naufalsrfdn/indobert-emotion`), terintegrasi dengan antarmuka **Streamlit**, database **SQLite**, serta mendukung alur **Preprocessing Teks** dan mekanisme **Incremental Learning** (pelatihan ulang berbasis validasi manusia) secara **100% lokal (offline)**.

Aplikasi ini dibangun sebagai produk akhir dari penelitian Skripsi Program Studi Informatika, Fakultas Sains dan Teknologi, Universitas PGRI Yogyakarta.

---

## 📌 Ringkasan Skripsi

* **Judul Skripsi**: Deteksi Emosi Multikategori pada Komentar Video YouTube Menggunakan Model IndoBERT
* **Penulis**: Naufal Syarifuddin (NPM: `22111100047`)
* **Dosen Pembimbing**: Nurirwan Saputra, S.Kom., M.Eng.
* **Institusi**: Program Studi Informatika, Universitas PGRI Yogyakarta (UPY)
* **Dataset**: 5.500 komentar dari video YouTube kanal *Rakyat Bersuara* (*"Dugaan Dalang Demo Rusuh..."*) yang dilabeli manual oleh anotator ahli bahasa (Sudaluwan, S.Pd., guru PBSI MTsN 4 Bantul).

---

## ✨ Fitur Utama Aplikasi

1. **🧹 Preprocessing Teks Otomatis**
   - Mengimplementasikan alur prapemrosesan teks sesuai Bab 3 & 4 Dokumen Skripsi:
     - **Case Folding**: Mengubah huruf kapital menjadi huruf kecil (*lowercase*).
     - **Text Cleaning**: Menghapus URL (`http/www`), emoji, tanda baca, angka, dan spasi berlebih.
   - Diterapkan pada prediksi real-time, scraping YouTube, maupun sebelum proses training ulang.

2. **📝 Real-time Emotion Prediction (`app.py`)**
   - Mengklasifikasikan ekspresi emosi pada teks komentar yang dimasukkan secara manual oleh pengguna.
   - **Pemilih Versi Model (Model Selector)**: Pengguna dapat memilih versi model lokal mana saja (`v1`, `v2`, dll) pada Halaman Utama.
   - Menampilkan label emosi terdeteksi, lencana warna emosi (*emotion badges*), serta rincian probabilitas 6 emosi secara visual.
   - Menyimpan hasil prediksi secara otomatis ke database SQLite.

3. **📥 YouTube Comment Scraper + Auto Classification (`pages/2_Scraping_YouTube.py`)**
   - Pengambilan komentar otomatis dari platform YouTube (Mode Studi Kasus Preset / Link YouTube Bebas).
   - Fitur filter berdasarkan rentang tanggal dan limit jumlah komentar.
   - Mendukung pemilihan versi model lokal sebelum scraping.
   - Menggabungkan alur Scraping $\rightarrow$ Preprocessing $\rightarrow$ Prediksi Emosi IndoBERT $\rightarrow$ Auto Save ke Database.

4. **📊 Dashboard Analisis Emosi (`pages/1_Dashboard_Analisis.py`)**
   - Visualisasi interaktif menggunakan **Plotly Express** dengan skema warna emosi terintegrasi.
   - Menampilkan statistik total data dalam *Metric Cards*, grafik batang distribusi emosi prediksi model, grafik distribusi label validasi manusia (*human label*), serta *pie chart* perbandingan akurasi validasi (*Benar vs Salah*).

5. **📚 History & Log Management (`pages/3_History.py`)**
   - Menampilkan seluruh riwayat komentar, hasil prediksi, *confidence score*, tanggal pembuatan, dan label validasi dalam bentuk tabel interaktif.
   - Dilengkapi fitur pencarian kata kunci dan filter kategori emosi.

6. **📝 Validasi Label Emosi / Human Correction (`pages/4_Validasi.py`)**
   - Sistem *Human-in-the-loop* (dilindungi sistem login autentikasi password).
   - Pakar/Pengguna dapat memverifikasi atau mengoreksi label emosi prediksi model.
   - Mendukung validasi per komentar maupun simpan validasi massal (*batch update*).

7. **🔁 Incremental Learning / Training Ulang Model (`pages/5_Training_Ulang_Model.py`)**
   - Pelatihan ulang (*fine-tuning*) model IndoBERT secara **100% lokal** berbasis data validasi manusia terbaru.
   - **Pencegahan Kunci Versi**: Secara otomatis menggunakan versi model lokal **terbaru** sebagai dasar pelatihan ulang.
   - Menggenerasi versi model baru (contoh: `v1`, `v2`, `v3`) secara otomatis di folder `model_versions/`.
   - Menyajikan preview data hasil preprocessing, metrik evaluasi (*Accuracy*, *Precision*, *Recall*, *F1-Score*), dan visualisasi *Confusion Matrix* (Seaborn/Matplotlib).

---

## 🎯 6 Kategori Emosi

Sistem mengklasifikasikan teks ke dalam 6 kategori emosi utama:

| Label Emosi | Warna Badge | Keterangan | Contoh Teks Komentar |
| :--- | :---: | :--- | :--- |
| **Marah** | 🔴 Merah | Kesal, benci, atau tidak setuju | *"Terus gimana? Coba lu jelasin, jgn nyocot doang"* |
| **Sedih** | 🔵 Biru | Kehilangan, kecewa, empati | *"Ferry di pojokin terus. Bahkan host juga pojokin ferry. Kasihan."* |
| **Senang** | 🟢 Hijau | Kebahagiaan, kepuasan, apresiasi | *"Diskusi mantulll"* |
| **Takut** | 🟣 Ungu | Ancaman, bahaya, kekhawatiran | *"Indonesia hancur karna di adudomba"* |
| **Terkejut** | 🟠 Oranye | Kejadian tak terduga / mengejutkan | *"Daginggg semuaa gelooo"* |
| **Netral (Biasa)** | ⚪ Abu-abu | Tidak menunjukkan emosi signifikan | *"Banyakk yang dipotong videonyaa"* |

---

## 📊 Hasil Pengujian Model IndoBERT

Model `indobenchmark/IndoBERT-base-p1` di-fine-tune menggunakan 3 skenario pembagian data latih dan uji (*data splitting*):

| Skenario Split (Latih:Uji) | Akurasi | Macro F1-Score | Weighted F1-Score | Status |
| :---: | :---: | :---: | :---: | :---: |
| **80 : 20** | **68.0%** | **0.55** | **0.67** | 🥇 **Performa Terbaik** |
| **70 : 30** | 65.7% | 0.47 | 0.64 | 🥈 |
| **60 : 40** | 63.0% | 0.45 | 0.62 | 🥉 |

> **Hyperparameter Fine-Tuning**: Epochs = 3, Batch Size = 8/16, Learning Rate = $2 \times 10^{-5}$, Optimizer = AdamW, Max Length = 256.

---

## 📂 Struktur Direktori Proyek

```text
emotiondetectionv2/
├── app.py                      # Entry point aplikasi Streamlit & Prediksi Real-time
├── load_model.py               # Loader model IndoBERT 100% lokal (offline)
├── ui_components.py            # Design system, Preprocessing Teks, & Emotion Badges
├── auth.py                     # Authentikasi password admin session state
├── db.py                       # Helper & koneksi SQLite Database (emotion.db)
├── emotion.db                  # Database SQLite penampung hasil prediksi & validasi
├── requirements.txt            # Package dependencies
├── .gitignore                  # Aturan ignore file model berat & cache Git
├── model_versions/             # Direktori penampung versi model lokal (v1, v2, dll.)
│   ├── active.txt              # Penanda versi model aktif saat ini
│   └── v1/                     # Folder bobot & konfigurasi model versi 1
├── pages/                      # Multi-page Streamlit routes
│   ├── 1_Dashboard_Analisis.py # Dashboard visualisasi & statistik Plotly
│   ├── 2_Scraping_YouTube.py   # Web Scraper komentar YouTube + Auto Predict
│   ├── 3_History.py            # Menampilkan seluruh riwayat data SQLite & Search
│   ├── 4_Validasi.py           # Interface validasi label emosi (Human-in-the-loop)
│   └── 5_Training_Ulang_Model.py # Fitur Incremental Learning & Evaluasi
└── DRAF SKRIPSI NAUFAL SYARIFUDDIN-INFORMATIKA.pdf # Dokumen Laporan Skripsi Lengkap
```

---

## 🛠️ Skema Database (`emotion.db`)

Tabel Utama: `hasil_prediksi`

```sql
CREATE TABLE IF NOT EXISTS hasil_prediksi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    komentar TEXT,
    emosi TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    label_benar TEXT
);
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Prasyarat
- **Python**: versi 3.8, 3.9, 3.10, atau 3.11.
- **Git** & **Pip** terinstall.

### 2. Kloning & Persiapan Environment

```bash
# Clone repository ini (atau buka folder projek)
cd /path/to/emotiondetectionv2

# Buat virtual environment di direktori Home (Disarankan untuk pengguna Linux/Ubuntu)
python3 -m venv ~/venv_emotion

# Aktifkan virtual environment
source ~/venv_emotion/bin/activate  # Linux/macOS
# venv\Scripts\activate            # Windows
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. 💾 Download Model Versi Pertama (v1)
Karena bobot file model tidak di-commit ke GitHub demi menghemat kuota repositori, Anda dapat mengunduh model versi pertama (`v1`) yang sudah di-fine tune dari Google Drive:
- 🔗 **Link Download Model v1**: [Google Drive Folder Model v1](https://drive.google.com/drive/folders/1o9Qbsd_IUDAWX5OnDN_SHKHFMhrXgrsC)

> **Petunjuk**: Setelah diunduh, ekstrak/masukkan folder `v1` ke dalam direktori `model_versions/v1/` pada proyek ini.

### 5. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

### 🔑 Autentikasi Password
Beberapa fitur administratif (*Validasi* & *Training Ulang Model*) dilindungi login password:
- **Password Default**: `nopalganteng` *(Dapat diubah pada `auth.py`)*.

---

## 📜 Lisensi & Hak Cipta

© 2025/2026 **Naufal Syarifuddin** — Program Studi Informatika, Fakultas Sains dan Teknologi, Universitas PGRI Yogyakarta. Hak cipta dilindungi undang-undang.
