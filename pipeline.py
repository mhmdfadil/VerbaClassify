"""
pipeline.py
===========
Modul ini mengimplementasikan pipeline machine learning end-to-end untuk
sistem klasifikasi respon masyarakat terhadap isu pelecehan seksual verbal.

Pipeline mencakup seluruh alur dari data mentah hingga evaluasi model,
sesuai dengan skema sistem yang dirancang dalam proposal skripsi
Rizka Mardiah Putri Buyung Lubis (2026):

    [Data Mentah]
        ↓  preprocessing.preprocess_batch()
    [Teks Bersih]
        ↓  TfidfVectorizer
    [Matriks Fitur TF-IDF]
        ↓  train_test_split (stratified)
    [Train Set] + [Test Set]
        ↓  SMOTE (jika use_smote=True)
    [Balanced Train Set]
        ↓  ID3ModifiedClassifier.fit()
    [Model Terlatih]
        ↓  model.predict()
    [Prediksi] → Evaluasi (accuracy, precision, recall, F1, CM)

Fungsi utama yang tersedia:
    run_experiment()   : Menjalankan satu siklus training dan evaluasi lengkap.
    run_depth_sweep()  : Menjalankan eksperimen untuk berbagai nilai max_depth (1-20).
    save_model()       : Menyimpan model ke file .pkl di disk.
    load_model()       : Memuat model dari file .pkl di disk.
    predict_single()   : Melakukan prediksi untuk satu teks baru.

Dependensi eksternal:
    - scikit-learn : TfidfVectorizer, train_test_split, metrics evaluasi, LabelEncoder.
    - imbalanced-learn: SMOTE untuk oversampling kelas minoritas.
    - numpy         : Operasi array.
    - pandas        : Pengelolaan data (tidak langsung di fungsi utama).
    - pickle        : Serialisasi/deserialisasi model ke file.

Dependensi internal:
    - preprocessing.py   : Pipeline preprocessing teks Bahasa Indonesia.
    - id3_modified.py    : Implementasi classifier ID3 Modifikasi.

Direktori model:
    Model yang disimpan disimpan di subdirektori 'models/' relatif terhadap
    lokasi file pipeline.py ini. Direktori dibuat otomatis jika belum ada.

Author   : Rizka Mardiah Putri Buyung Lubis (220170183)
Institusi: Universitas Malikussaleh, 2026
"""

import numpy as np
import pandas as pd
import pickle
import os
import math
import time
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from preprocessing import preprocess_batch
from id3_modified import ID3ModifiedClassifier

# ── Konfigurasi direktori penyimpanan model ───────────────────────────────────
# Semua file model (.pkl) disimpan di subdirektori 'models/' agar terorganisir
# dan mudah dikelola. Direktori dibuat otomatis jika belum ada.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# FUNGSI UTAMA PIPELINE
# ============================================================

def run_experiment(
    texts: list,
    labels: list,
    train_ratio: float = 0.8,
    max_depth: int = 3,
    use_smote: bool = True,
    max_features: int = 300,
    preprocessed_texts: list = None,
) -> dict:
    """
    Menjalankan satu siklus eksperimen klasifikasi lengkap dari awal hingga evaluasi.

    Ini adalah fungsi inti pipeline yang mengeksekusi semua tahap secara berurutan:
    preprocessing → TF-IDF → split data → SMOTE → training ID3 Modifikasi → evaluasi.
    Setiap tahap dilaporkan melalui print() untuk memudahkan monitoring di konsol.

    Optimasi performa:
        Jika preprocessed_texts sudah disediakan (misal diambil dari kolom
        'preprocessed_text' di database), tahap preprocessing dilewati dan
        teks bersih langsung digunakan. Ini menghindari komputasi ulang
        yang mahal (terutama stemming Sastrawi) saat eksperimen dijalankan
        ulang pada dataset yang sama.

    Strategi split data:
        Menggunakan stratified split (stratify=y) agar distribusi kelas
        pada data training dan testing proporsional terhadap distribusi
        kelas di dataset asli. Ini penting untuk dataset tidak seimbang.

    Penanganan SMOTE:
        SMOTE (Synthetic Minority Over-sampling Technique) diterapkan HANYA
        pada data training untuk menambah sampel sintetis kelas minoritas.
        Data testing TIDAK disentuh oleh SMOTE agar evaluasi tetap realistis.
        Jika SMOTE gagal (misal kelas minoritas terlalu sedikit), proses
        dilanjutkan tanpa SMOTE dengan pesan peringatan.

    Pengukuran waktu:
        Waktu training dan prediksi diukur secara terpisah menggunakan
        time.perf_counter() untuk presisi tinggi (sub-milidetik).

    Args:
        texts             (list[str])       : Daftar teks komentar asli (mentah).
        labels            (list[str])       : Daftar label kelas ('positif'/'negatif'),
                                              sejajar dengan texts.
        train_ratio       (float)           : Proporsi data untuk training (0.0-1.0).
                                              Contoh: 0.8 untuk split 80:20.
                                              Default: 0.8.
        max_depth         (int)             : Kedalaman maksimum pohon keputusan ID3.
                                              Berdasarkan jurnal, nilai optimal
                                              umumnya 3–6. Default: 3.
        use_smote         (bool)            : True untuk mengaktifkan SMOTE oversampling
                                              pada data training. Default: True.
        max_features      (int)             : Jumlah maksimum fitur (kata) yang disimpan
                                              oleh TF-IDF vectorizer. Semakin tinggi
                                              semakin kaya representasi, tapi semakin
                                              lambat training. Default: 300.
        preprocessed_texts (list[str] | None): Teks yang sudah dipreproses sebelumnya.
                                               Jika disediakan (tidak None) dan panjangnya
                                               sama dengan texts, akan digunakan langsung.
                                               Jika None, preprocessing dijalankan ulang.
                                               Default: None.

    Returns:
        dict: Dictionary komprehensif berisi semua hasil eksperimen:

            Data dan konfigurasi:
                preprocessed (list[str])   : Teks hasil preprocessing.
                vectorizer   (TfidfVectorizer): Vectorizer yang sudah di-fit.
                model        (ID3ModifiedClassifier): Model yang sudah dilatih.
                le           (LabelEncoder): Encoder label yang sudah di-fit.
                class_names  (list[str])   : Nama kelas ['negatif', 'positif'].
                X_train      (np.ndarray)  : Fitur training (setelah SMOTE jika aktif).
                X_test       (np.ndarray)  : Fitur testing.
                y_train      (np.ndarray)  : Label training (setelah SMOTE jika aktif).
                y_test       (np.ndarray)  : Label testing.
                y_pred       (np.ndarray)  : Hasil prediksi pada data testing.
                train_count  (int)         : Jumlah sampel training (setelah SMOTE).
                test_count   (int)         : Jumlah sampel testing.

            Metrik evaluasi:
                accuracy     (float): Akurasi keseluruhan (0.0-1.0).
                precision    (float): Presisi berbobot (weighted average).
                recall       (float): Recall berbobot (weighted average).
                f1           (float): F1-score berbobot (weighted average).
                tp           (int)  : True Positive dari confusion matrix.
                tn           (int)  : True Negative dari confusion matrix.
                fp           (int)  : False Positive dari confusion matrix.
                fn           (int)  : False Negative dari confusion matrix.
                confusion_matrix (list): CM dalam format list 2D.
                report       (dict) : Laporan klasifikasi lengkap per kelas.

            Parameter eksperimen:
                train_ratio  (float): Proporsi training yang digunakan.
                test_ratio   (float): Proporsi testing yang digunakan.
                max_depth    (int)  : Nilai max_depth yang digunakan.
                use_smote    (bool) : Status penggunaan SMOTE.

            Waktu eksekusi:
                execution_time (float): Total waktu (training + prediksi) dalam detik.
                train_time     (float): Waktu training saja dalam detik.
                predict_time   (float): Waktu prediksi saja dalam detik.

    Catatan:
        Confusion matrix TP/TN/FP/FN hanya diekstrak jika jumlah kelas = 2.
        Untuk kasus lain (tidak mungkin terjadi di sistem ini), nilai diset 0.
    """
    # ── Tahap 1: Preprocessing ──────────────────────────────────────────────
    # Gunakan preprocessed_text dari database jika tersedia untuk efisiensi.
    # Jika tidak ada, jalankan pipeline preprocessing dari awal.
    if preprocessed_texts and len(preprocessed_texts) == len(texts):
        print(f"  [1] Menggunakan preprocessed_text dari database ({len(texts)} teks)...")
        preprocessed = preprocessed_texts
    else:
        print(f"  [1] Preprocessing {len(texts)} teks...")
        preprocessed = preprocess_batch(texts)

    # ── Tahap 2: Label Encoding ─────────────────────────────────────────────
    # LabelEncoder mengubah label string ke integer numerik yang dibutuhkan
    # oleh sklearn dan classifier:
    #   'negatif' → 0 (kelas negatif / tidak mengandung pelecehan)
    #   'positif' → 1 (kelas positif / mengandung pelecehan)
    # LabelEncoder mengurutkan secara alfabetis: negatif < positif → 0, 1
    le = LabelEncoder()
    y = le.fit_transform(labels)  # Fit dan transform sekaligus
    class_names = list(le.classes_)  # Simpan nama kelas untuk laporan

    # ── Tahap 3: TF-IDF Feature Extraction ─────────────────────────────────
    # TF-IDF (Term Frequency-Inverse Document Frequency) mengubah teks
    # menjadi representasi numerik berdasarkan frekuensi dan kepentingan kata.
    #
    # Parameter yang digunakan:
    #   max_features: Batasi vocabulary ke N kata terpenting (kurangi dimensi)
    #   ngram_range=(1,2): Gunakan unigram DAN bigram (pasangan 2 kata berurutan)
    #                      untuk menangkap konteks (misal "pelecehan verbal")
    #   min_df=2: Abaikan kata yang hanya muncul di < 2 dokumen (noise reduction)
    #   sublinear_tf=True: Terapkan log(tf)+1 agar frekuensi tinggi tidak mendominasi
    print(f"  [2] TF-IDF extraction (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(preprocessed)  # Fit vectorizer dan transform teks

    # ── Tahap 4: Split Data ─────────────────────────────────────────────────
    # Membagi data menjadi training dan testing set menggunakan stratified split.
    # random_state=42 memastikan reproducibility (hasil yang sama setiap run).
    # stratify=y menjaga distribusi kelas proporsional di kedua set.
    test_ratio = round(1.0 - train_ratio, 2)
    print(f"  [3] Split data {int(train_ratio*100)}:{int(test_ratio*100)}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=42,
        stratify=y       # Pastikan distribusi kelas seimbang di train dan test
    )

    # ── Tahap 5: SMOTE Oversampling ─────────────────────────────────────────
    # SMOTE menambahkan sampel sintetis untuk kelas minoritas (positif) pada
    # data training, sehingga model tidak bias ke kelas mayoritas (negatif).
    # SMOTE HANYA diterapkan pada training set; testing set dibiarkan asli.
    #
    # k_neighbors disesuaikan secara adaptif: min(5, jumlah_minoritas-1)
    # untuk menghindari error jika sampel minoritas terlalu sedikit.
    if use_smote:
        print(f"  [4] SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y_train).min() - 1))
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        except Exception as e:
            # Fallback: lanjutkan tanpa SMOTE jika terjadi error
            # (misal: kelas minoritas hanya 1 sampel)
            print(f"  SMOTE failed ({e}), skip SMOTE")
            X_train_sm, y_train_sm = X_train, y_train
    else:
        # SMOTE tidak diaktifkan: gunakan data training apa adanya
        X_train_sm, y_train_sm = X_train, y_train

    # ── Tahap 6: Training ID3 Modifikasi ───────────────────────────────────
    # Membangun pohon keputusan menggunakan formula entropy modifikasi dari
    # jurnal Asrianda et al. (2025). Waktu training diukur secara presisi.
    print(f"  [5] Training ID3 Modifikasi (max_depth={max_depth})...")
    t_start = time.perf_counter()  # Mulai penghitung waktu training
    model = ID3ModifiedClassifier(max_depth=max_depth, min_samples_split=2)
    model.fit(X_train_sm, y_train_sm)
    t_train = time.perf_counter() - t_start  # Selesai hitung waktu training

    # ── Tahap 7: Prediksi dan Evaluasi ─────────────────────────────────────
    # Prediksi dilakukan pada data testing (yang tidak dilihat model saat training).
    # Waktu prediksi diukur terpisah dari waktu training.
    print(f"  [6] Evaluasi model...")
    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_pred = time.perf_counter() - t_pred_start
    execution_time = round(t_train + t_pred, 6)  # Total waktu eksekusi

    # Hitung metrik evaluasi standar dengan average='weighted' agar
    # memperhitungkan ketidakseimbangan kelas secara proporsional
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]] untuk kasus biner
    cm = confusion_matrix(y_test, y_pred)

    # Laporan klasifikasi lengkap per kelas (precision, recall, F1, support)
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,   # Format dictionary agar mudah diproses
        zero_division=0     # Hindari warning jika suatu kelas tidak diprediksi
    )

    # Ekstrak TP, TN, FP, FN dari confusion matrix (hanya untuk kasus biner)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()  # .ravel() meratakan CM 2x2 menjadi array 1D
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(
        f"  ✓ Akurasi: {acc:.4f} | Presisi: {prec:.4f} | "
        f"Recall: {rec:.4f} | F1: {f1:.4f} | Waktu: {execution_time:.4f}s"
    )

    # ── Kembalikan semua hasil dalam satu dictionary ────────────────────────
    return {
        # Objek pipeline yang dibutuhkan untuk penyimpanan dan prediksi
        'preprocessed': preprocessed,
        'vectorizer':   vectorizer,
        'model':        model,
        'le':           le,
        'class_names':  class_names,
        # Data training dan testing (untuk analisis lebih lanjut)
        'X_train':      X_train_sm,
        'X_test':       X_test,
        'y_train':      y_train_sm,
        'y_test':       y_test,
        'y_pred':       y_pred,
        'train_count':  X_train_sm.shape[0],
        'test_count':   X_test.shape[0],
        # Metrik evaluasi keseluruhan
        'accuracy':     float(acc),
        'precision':    float(prec),
        'recall':       float(rec),
        'f1':           float(f1),
        # Elemen confusion matrix
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'confusion_matrix': cm.tolist(),
        'report':           report,
        # Parameter yang digunakan
        'train_ratio':  train_ratio,
        'test_ratio':   test_ratio,
        'max_depth':    max_depth,
        'use_smote':    use_smote,
        # Waktu eksekusi
        'execution_time': execution_time,
        'train_time':     round(t_train, 6),
        'predict_time':   round(t_pred, 6),
    }


def run_depth_sweep(
    texts: list,
    labels: list,
    train_ratio: float = 0.8,
    depths: list = None,
    use_smote: bool = True,
    max_features: int = 300,
    preprocessed_texts: list = None,
) -> list:
    """
    Menjalankan eksperimen untuk berbagai nilai max_depth secara berurutan.

    Fungsi ini digunakan untuk menganalisis pengaruh parameter kedalaman
    pohon keputusan terhadap akurasi model, menghasilkan data yang
    divisualisasikan sebagai grafik "Akurasi vs. Max Depth" pada halaman
    detail eksperimen. Sesuai dengan metodologi dalam jurnal Asrianda et al.
    (2025) yang menguji depth 1 hingga 20.

    Optimasi efisiensi:
        Preprocessing, TF-IDF, split data, dan SMOTE hanya dijalankan SEKALI
        di awal. Loop kemudian hanya mengulang tahap training dan prediksi
        untuk setiap nilai depth. Ini jauh lebih efisien daripada memanggil
        run_experiment() berulang kali yang akan melakukan preprocessing
        berulang-ulang.

    Args:
        texts             (list[str])       : Daftar teks komentar asli.
        labels            (list[str])       : Daftar label kelas, sejajar dengan texts.
        train_ratio       (float)           : Proporsi data training (0.0-1.0). Default: 0.8.
        depths            (list[int] | None): Daftar nilai max_depth yang akan diuji.
                                               Jika None, gunakan range(1, 21) = [1..20].
                                               Default: None.
        use_smote         (bool)            : Aktifkan SMOTE pada data training. Default: True.
        max_features      (int)             : Jumlah fitur TF-IDF maksimum. Default: 300.
        preprocessed_texts (list[str] | None): Teks yang sudah dipreproses. Default: None.

    Returns:
        list[dict]: Daftar hasil untuk setiap nilai depth yang diuji.
                    Setiap dict berisi key-key berikut:
            - max_depth      (int)  : Nilai kedalaman pohon yang diuji.
            - accuracy       (float): Akurasi pada depth tersebut.
            - precision      (float): Presisi berbobot.
            - recall         (float): Recall berbobot.
            - f1             (float): F1-score berbobot.
            - execution_time (float): Waktu training + prediksi dalam detik.

    Contoh output:
        [
            {'max_depth': 1, 'accuracy': 0.857, 'precision': 0.84, ...},
            {'max_depth': 2, 'accuracy': 0.866, 'precision': 0.86, ...},
            ...
            {'max_depth': 20, 'accuracy': 0.799, 'precision': 0.79, ...},
        ]

    Catatan:
        Hasil yang dikembalikan dapat langsung disimpan ke database
        menggunakan fungsi insert_depth_results() di database.py.
    """
    # Gunakan depth default 1-20 jika tidak dispesifikasi
    if depths is None:
        depths = list(range(1, 21))

    # ── Preprocessing (sekali saja) ──────────────────────────────────────────
    if preprocessed_texts and len(preprocessed_texts) == len(texts):
        print(f"  [Sweep] Menggunakan preprocessed_text dari database...")
        preprocessed = preprocessed_texts
    else:
        preprocessed = preprocess_batch(texts)

    # ── Label Encoding (sekali saja) ─────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ── TF-IDF (sekali saja, sama dengan run_experiment) ─────────────────────
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(preprocessed)

    # ── Split Data (sekali saja) ──────────────────────────────────────────────
    # random_state=42 dan stratify=y sama dengan run_experiment agar
    # perbandingan antar depth fair (data training/testing identik)
    test_ratio = round(1.0 - train_ratio, 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    # ── SMOTE (sekali saja) ───────────────────────────────────────────────────
    if use_smote:
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, np.bincount(y_train).min() - 1)
        )
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        except Exception:
            # Fallback tanpa SMOTE
            X_train_sm, y_train_sm = X_train, y_train
    else:
        X_train_sm, y_train_sm = X_train, y_train

    # ── Loop sweep: training dan evaluasi untuk setiap nilai depth ───────────
    results = []
    for d in depths:
        # Ukur waktu training untuk depth ini
        t0 = time.perf_counter()
        model = ID3ModifiedClassifier(max_depth=d, min_samples_split=2)
        model.fit(X_train_sm, y_train_sm)
        t_train_d = time.perf_counter() - t0

        # Ukur waktu prediksi untuk depth ini
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        t_pred_d = time.perf_counter() - t1

        exec_time = round(t_train_d + t_pred_d, 6)
        acc_val   = float(accuracy_score(y_test, y_pred))

        # Simpan hasil depth ini ke list
        results.append({
            'max_depth':      d,
            'accuracy':       acc_val,
            'precision':      float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall':         float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1':             float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'execution_time': exec_time,
        })
        print(f"    depth={d}: acc={acc_val:.4f} | time={exec_time:.4f}s")

    return results


# ============================================================
# FUNGSI MANAJEMEN MODEL
# Fungsi-fungsi untuk menyimpan dan memuat model ke/dari disk.
# Model disimpan sebagai file pickle (.pkl) yang mengandung
# tiga komponen: TF-IDF vectorizer, classifier ID3 Modifikasi,
# dan LabelEncoder.
# ============================================================

def save_model(experiment_id: int, vectorizer, model, le) -> str:
    """
    Menyimpan komponen model terlatih ke file pickle di disk.

    Menyimpan tiga komponen yang dibutuhkan untuk prediksi ke depannya
    dalam satu file pickle:
        1. vectorizer : TfidfVectorizer yang sudah di-fit dengan vocabulary training.
        2. model      : ID3ModifiedClassifier yang sudah dilatih.
        3. le         : LabelEncoder yang sudah di-fit dengan nama kelas.

    File disimpan dengan nama 'model_{experiment_id}.pkl' di direktori MODEL_DIR.
    Penamaan berdasarkan experiment_id memudahkan identifikasi model mana
    yang berkorespondensi dengan eksperimen mana di database.

    Args:
        experiment_id (int)              : ID eksperimen dari database, digunakan
                                           sebagai nama file model.
        vectorizer    (TfidfVectorizer)  : Vectorizer yang sudah di-fit saat training.
        model         (ID3ModifiedClassifier): Model classifier yang sudah dilatih.
        le            (LabelEncoder)     : Encoder label yang sudah di-fit.

    Returns:
        str: Path absolut ke file pickle yang berhasil disimpan.
             Format: '/path/to/models/model_{experiment_id}.pkl'

    Catatan:
        Pastikan direktori MODEL_DIR writable. File lama dengan nama yang sama
        akan ditimpa (overwrite) tanpa peringatan.
    """
    path = os.path.join(MODEL_DIR, f'model_{experiment_id}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': model, 'le': le}, f)
    return path


def load_model(experiment_id: int):
    """
    Memuat komponen model dari file pickle yang tersimpan di disk.

    Kebalikan dari save_model(). Memuat file pickle yang sebelumnya
    disimpan dan mengekstrak tiga komponen: vectorizer, model, dan le.

    File yang dicari: 'model_{experiment_id}.pkl' di direktori MODEL_DIR.

    Args:
        experiment_id (int): ID eksperimen yang model-nya ingin dimuat.

    Returns:
        tuple: (vectorizer, model, le) — tiga komponen model:
            - vectorizer (TfidfVectorizer)   : Siap untuk transform teks baru.
            - model      (ID3ModifiedClassifier): Siap untuk predict.
            - le         (LabelEncoder)      : Siap untuk inverse_transform prediksi.
        Jika file tidak ditemukan, mengembalikan (None, None, None).

    Contoh penggunaan:
        vectorizer, model, le = load_model(5)
        if model is not None:
            X = vectorizer.transform(["teks baru"])
            pred = model.predict(X)
            label = le.inverse_transform(pred)
    """
    path = os.path.join(MODEL_DIR, f'model_{experiment_id}.pkl')
    if not os.path.exists(path):
        # File tidak ditemukan, kembalikan tuple None
        return None, None, None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['vectorizer'], data['model'], data['le']


# ============================================================
# FUNGSI PREDIKSI TUNGGAL
# ============================================================

def predict_single(text: str, experiment_id: int) -> dict:
    """
    Melakukan klasifikasi untuk satu teks baru menggunakan model yang tersimpan.

    Fungsi ini adalah titik masuk (entry point) untuk fitur klasifikasi
    pengguna di halaman /classify dan endpoint API /api/classify.
    Menggabungkan seluruh pipeline prediksi: load model → preprocess
    → vectorize → predict → decode label.

    Alur kerja:
        1. Muat model (vectorizer + classifier + encoder) dari file pickle.
        2. Preproses teks input menggunakan pipeline preprocessing.
        3. Validasi teks: jika kosong setelah preprocessing, kembalikan error.
        4. Transformasi teks ke representasi TF-IDF menggunakan vectorizer.
        5. Prediksi kelas menggunakan model ID3 Modifikasi.
        6. Hitung confidence menggunakan predict_proba().
        7. Decode label numerik kembali ke nama kelas string.

    Args:
        text          (str): Teks komentar yang akan diklasifikasikan.
                             Bisa berupa teks mentah dari pengguna, belum dipreproses.
        experiment_id (int): ID eksperimen yang model-nya akan digunakan
                             untuk prediksi. Biasanya merupakan ID eksperimen
                             terbaik yang sudah ditandai admin (is_best=True).

    Returns:
        dict: Dictionary hasil prediksi. Ada dua kemungkinan format:

            Jika berhasil:
                {
                    'label'      : str   — Label prediksi ('positif' atau 'negatif').
                    'confidence' : float — Nilai kepercayaan prediksi (0.0-1.0).
                    'preprocessed': str  — Teks setelah preprocessing.
                    'raw_pred'   : int   — Nilai prediksi numerik (0 atau 1).
                }

            Jika gagal:
                {
                    'error': str — Pesan error yang menjelaskan penyebab kegagalan.
                }

    Contoh penggunaan:
        result = predict_single("ini tindakan pelecehan verbal", 5)
        if 'error' not in result:
            print(f"Label: {result['label']}, Confidence: {result['confidence']:.2f}")
        else:
            print(f"Error: {result['error']}")

    Catatan:
        Confidence yang dikembalikan adalah nilai sederhana (0.85 untuk prediksi
        utama) dari predict_proba() di ID3ModifiedClassifier, bukan probabilitas
        kalibrasi sebenarnya.
    """
    # Langkah 1: Muat model dari disk
    vectorizer, model, le = load_model(experiment_id)
    if model is None:
        # File model tidak ditemukan di disk
        return {'error': 'Model tidak ditemukan'}

    # Langkah 2: Preproses teks input (sebagai list berisi satu elemen)
    preprocessed = preprocess_batch([text])[0]

    # Langkah 3: Validasi hasil preprocessing
    if not preprocessed.strip():
        # Seluruh konten teks terhapus setelah preprocessing
        return {'error': 'Teks kosong setelah preprocessing'}

    # Langkah 4: Transformasi ke representasi TF-IDF
    # Gunakan transform() bukan fit_transform() karena vectorizer
    # sudah di-fit saat training dan tidak boleh berubah
    X = vectorizer.transform([preprocessed])

    # Langkah 5 & 6: Prediksi kelas dan confidence
    pred    = model.predict(X)[0]         # Prediksi kelas: 0 atau 1
    proba   = model.predict_proba(X)[0]   # Probabilitas untuk kedua kelas
    confidence = float(proba[pred])       # Ambil confidence untuk kelas yang diprediksi

    # Langkah 7: Decode label numerik kembali ke string nama kelas
    label = le.inverse_transform([pred])[0]  # 0 → 'negatif', 1 → 'positif'

    return {
        'label':        label,
        'confidence':   confidence,
        'preprocessed': preprocessed,
        'raw_pred':     int(pred)
    }