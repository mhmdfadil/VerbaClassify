# VerbaClassify — Sistem Klasifikasi Respon Pelecehan Seksual Verbal

**Skripsi:** Klasifikasi Respon Masyarakat Terhadap Isu Pelecehan Seksual Secara Verbal Pada Platform X Menggunakan ID3 Modifikasi  
**Penulis:** Rizka Mardiah Putri Buyung Lubis (NIM: 220170183)  
**Prodi:** Teknik Informatika — Universitas Malikussaleh, 2026

---

## Struktur Proyek

```
skripsi_app/
├── app.py                  # Flask application utama
├── id3_modified.py         # Implementasi algoritma ID3 Modifikasi
├── preprocessing.py        # Pipeline preprocessing Bahasa Indonesia
├── pipeline.py             # Pipeline training lengkap
├── database.py             # Koneksi & operasi Supabase
├── schema.sql              # SQL untuk membuat tabel di Supabase
├── requirements.txt        # Dependensi Python
├── .env                    # Konfigurasi environment (JANGAN commit)
├── uploads/                # Folder file CSV yang diupload
├── models/                 # Folder model yang disimpan (.pkl)
└── templates/
    ├── base.html           # Base template (navbar, footer)
    ├── index.html          # Halaman beranda publik
    ├── classify.html       # Halaman klasifikasi publik
    ├── about.html          # Halaman tentang
    ├── login.html          # Halaman login admin
    ├── dashboard.html      # Dashboard admin
    ├── datasets.html       # Daftar dataset
    ├── upload_dataset.html # Upload dataset baru
    ├── dataset_detail.html # Detail dataset
    ├── run_experiment.html # Form jalankan eksperimen
    ├── experiment_progress.html  # Progress training
    ├── experiments.html    # Daftar hasil eksperimen
    ├── experiment_detail.html    # Detail eksperimen & chart
    ├── statistics.html     # Statistik sistem
    └── predictions.html    # Riwayat prediksi
```

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Database Supabase
- Buka Supabase Dashboard → SQL Editor
- Copy-paste isi file `schema.sql` dan jalankan
- Tabel yang dibuat: `datasets`, `raw_data`, `experiments`, `experiment_details`, `depth_results`, `predictions`, `admin_sessions`

### 3. Konfigurasi .env
```env
SUPABASE_URL=https://lbzrvjrcyzmszgkijmoq.supabase.co
SUPABASE_KEY=sb_publishable_DgyQCpnwZtYIlIsHtRyv3A_xUX_Ru4H
SECRET_KEY=rizka_skripsi_secret_key_2026
ADMIN_USERNAME=admin
ADMIN_PASSWORD=rizka2026
```

### 4. Jalankan Aplikasi
```bash
python app.py
# Buka http://localhost:5000
```

---

## Alur Penggunaan

### Admin (Login: /admin/login)
1. **Upload Dataset** → `/admin/datasets/upload` → Upload file CSV
2. **Jalankan Eksperimen** → `/admin/experiments/run` → Pilih dataset, proporsi (80:20 / 70:30 / dll), max_depth, SMOTE
3. **Lihat Hasil** → `/admin/experiments` → Bandingkan akurasi antar proporsi
4. **Set Model Terbaik** → Klik "Set Terbaik" pada eksperimen dengan akurasi tertinggi
5. **Statistik** → `/admin/statistics` → Lihat grafik perbandingan

### Pengguna Publik (Tanpa Login)
- **Klasifikasi** → `/classify` → Masukkan teks komentar → Hasil otomatis

---

## Format Dataset CSV

```csv
full_text,label
"Pelaku pelecehan harus dihukum berat!","positif"
"hai cantik mau diajak jalan?","negatif"
```

- `full_text` : teks komentar dari Platform X
- `label` : `positif` (tidak mengandung pelecehan) atau `negatif` (mengandung pelecehan)

---

## Formula ID3 Modifikasi

**Entropy:**
```
Ent(D) = (1/ln2) × [ln(p+n) − (p·ln(p) + n·ln(n))/(p+n)]
```

**Conditional Entropy:**
```
EntA(D) = Σ (|Dj|/|D|) × [ln(|Dj|) − (pj/|Dj|)·ln(pj) − (nj/|Dj|)·ln(nj)]
```

**Information Gain:**
```
Gain(A) = Ent(D) − EntA(D)
```

*Sumber: Asrianda, Mawengkang, Sihombing & Nasution (2025) · JUTIF & EJET*

---

## Teknologi

| Komponen | Library |
|---|---|
| Web Framework | Flask |
| Database | Supabase (PostgreSQL) |
| Preprocessing | NLTK + PySastrawi |
| Feature Extraction | scikit-learn (TF-IDF) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Klasifikasi | ID3 Modifikasi (custom) |
| Frontend | HTML/CSS vanilla |

---

## Admin Credentials
- **Username:** admin  
- **Password:** rizka2026  
*(Ubah di file .env sebelum deploy produksi)*
