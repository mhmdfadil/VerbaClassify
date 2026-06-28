"""
database.py
===========
Modul ini menangani semua operasi CRUD (Create, Read, Update, Delete)
ke database Supabase yang digunakan oleh sistem klasifikasi respon
pelecehan seksual verbal pada platform X.

Modul ini berfungsi sebagai lapisan akses data (Data Access Layer) yang
memisahkan logika bisnis dari detail koneksi database. Semua operasi
database yang diperlukan oleh pipeline, app, dan modul lain harus
memanggil fungsi-fungsi di sini, bukan mengakses Supabase secara langsung.

Tabel yang dikelola:
    - datasets          : Metadata file CSV yang diunggah
    - raw_data          : Data komentar mentah beserta label dan hasil preprocessing
    - experiments       : Konfigurasi dan hasil setiap eksperimen klasifikasi
    - experiment_details: Metrik per kelas (precision, recall, F1) dari setiap eksperimen
    - depth_results     : Hasil akurasi untuk setiap nilai max_depth (sweep)
    - predictions       : Riwayat prediksi yang dilakukan pengguna

Dependensi:
    - supabase-py  : Client library untuk Supabase
    - python-dotenv: Membaca konfigurasi dari file .env

Variabel lingkungan yang dibutuhkan (.env):
    SUPABASE_URL  : URL project Supabase
    SUPABASE_KEY  : API key (anon/service role) Supabase

Author  : Rizka Mardiah Putri Buyung Lubis (220170183)
Institusi: Universitas Malikussaleh, 2026
"""

import os
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

# Muat variabel lingkungan dari file .env
load_dotenv()

# Variabel global untuk menyimpan instance Supabase (Singleton Pattern).
# Diinisialisasi sekali pada pemanggilan pertama get_db() agar tidak
# membuat koneksi baru di setiap pemanggilan fungsi database.
_supabase: Client = None


# ============================================================
# KONEKSI DATABASE
# ============================================================

def get_db() -> Client:
    """
    Mengembalikan instance Supabase Client (Singleton).

    Fungsi ini mengimplementasikan pola Singleton: instance client
    hanya dibuat satu kali selama siklus hidup aplikasi. Jika instance
    sudah ada, langsung dikembalikan tanpa membuat koneksi baru.

    Variabel lingkungan yang dibaca:
        SUPABASE_URL : URL project Supabase (contoh: https://xxx.supabase.co)
        SUPABASE_KEY : API key Supabase (anon key atau service_role key)

    Returns:
        Client: Instance Supabase yang siap digunakan untuk query.

    Catatan:
        Pastikan variabel SUPABASE_URL dan SUPABASE_KEY sudah diset
        di file .env sebelum memanggil fungsi ini.
    """
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        _supabase = create_client(url, key)
    return _supabase


# ============================================================
# DATASETS
# Fungsi-fungsi untuk mengelola metadata dataset yang diunggah.
# Tabel 'datasets' menyimpan informasi ringkasan tentang setiap
# file CSV yang diupload oleh admin.
# ============================================================

def insert_dataset(filename, total, positif, negatif, notes=''):
    """
    Menyimpan metadata dataset baru ke tabel 'datasets'.

    Dipanggil setelah admin berhasil mengunggah file CSV dan
    sistem selesai menghitung statistik distribusi kelas.

    Args:
        filename (str) : Nama file CSV yang diunggah (setelah di-sanitasi).
        total    (int) : Jumlah total baris data yang valid dalam CSV.
        positif  (int) : Jumlah baris berlabel 'positif'.
        negatif  (int) : Jumlah baris berlabel 'negatif'.
        notes    (str) : Catatan opsional dari admin. Default: ''.

    Returns:
        dict | None: Record dataset yang baru dibuat (termasuk ID yang
                     di-generate database), atau None jika insert gagal.
    """
    db = get_db()
    res = db.table('datasets').insert({
        'filename': filename,
        'total_data': total,
        'positif_count': positif,
        'negatif_count': negatif,
        'notes': notes
    }).execute()
    return res.data[0] if res.data else None


def get_all_datasets():
    """
    Mengambil semua record dataset, diurutkan dari yang terbaru.

    Digunakan pada halaman daftar dataset admin dan dropdown
    pemilihan dataset saat menjalankan eksperimen.

    Returns:
        list[dict]: Daftar semua record dari tabel 'datasets',
                    diurutkan berdasarkan 'uploaded_at' secara descending.
                    Mengembalikan list kosong [] jika tidak ada data.
    """
    db = get_db()
    res = db.table('datasets').select('*').order('uploaded_at', desc=True).execute()
    return res.data or []


def get_dataset_by_id(dataset_id):
    """
    Mengambil satu record dataset berdasarkan ID-nya.

    Digunakan untuk menampilkan detail dataset, memvalidasi keberadaan
    dataset sebelum menjalankan eksperimen, atau mengaitkan data
    eksperimen dengan datasetnya.

    Args:
        dataset_id (int): ID unik dataset yang ingin diambil.

    Returns:
        dict | None: Record dataset jika ditemukan, atau None jika
                     tidak ada dataset dengan ID tersebut.
    """
    db = get_db()
    res = db.table('datasets').select('*').eq('id', dataset_id).execute()
    return res.data[0] if res.data else None


def delete_dataset(dataset_id):
    """
    Menghapus record dataset beserta semua data terkait dari database.

    Dipanggil ketika admin memilih untuk menghapus dataset dari sistem.
    Perlu diperhatikan bahwa penghapusan ini akan menghapus record di
    tabel 'datasets'. Pastikan foreign key constraint atau cascade delete
    sudah dikonfigurasi di Supabase agar data terkait (raw_data,
    experiments, dll.) juga ikut terhapus.

    Args:
        dataset_id (int): ID unik dataset yang akan dihapus.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.
    """
    db = get_db()
    db.table('datasets').delete().eq('id', dataset_id).execute()


# ============================================================
# RAW DATA
# Fungsi-fungsi untuk mengelola data komentar mentah.
# Tabel 'raw_data' menyimpan setiap baris komentar dari CSV
# beserta label, dataset_id, dan hasil preprocessing-nya.
# ============================================================

def insert_raw_data_batch(records: list):
    """
    Menyimpan banyak record data mentah sekaligus menggunakan batch insert.

    Menggunakan strategi batch sebesar 500 record per request untuk
    menghindari timeout dan membatasi ukuran payload ke Supabase.
    Seluruh data preprocessing (preprocessed_text) sudah harus diisi
    sebelum fungsi ini dipanggil.

    Args:
        records (list[dict]): Daftar dict yang masing-masing berisi:
            - dataset_id        (int) : ID dataset induk.
            - full_text         (str) : Teks komentar asli.
            - label             (str) : Label kelas ('positif' atau 'negatif').
            - preprocessed_text (str) : Hasil preprocessing teks.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.

    Catatan:
        Jika total records > 500, insert dilakukan secara iteratif
        dalam beberapa batch hingga semua record berhasil disimpan.
    """
    db = get_db()
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        db.table('raw_data').insert(batch).execute()


def get_raw_data_by_dataset(dataset_id, limit=100, offset=0):
    """
    Mengambil sebagian data mentah milik dataset tertentu (paginasi).

    Digunakan pada halaman detail dataset untuk menampilkan preview
    data dengan paginasi, sehingga tidak semua data dimuat sekaligus.

    Args:
        dataset_id (int): ID dataset yang data-nya ingin ditampilkan.
        limit      (int): Jumlah maksimal baris yang diambil. Default: 100.
        offset     (int): Nomor baris awal (untuk paginasi). Default: 0.
                          Contoh: page=2, limit=20 → offset=20.

    Returns:
        list[dict]: Daftar record raw_data sesuai rentang yang diminta.
                    Mengembalikan list kosong [] jika tidak ada data.
    """
    db = get_db()
    res = (db.table('raw_data')
           .select('*')
           .eq('dataset_id', dataset_id)
           .range(offset, offset + limit - 1)
           .execute())
    return res.data or []


def get_all_raw_texts_labels(dataset_id):
    """
    Mengambil SEMUA teks, label, dan hasil preprocessing dari dataset tertentu.

    Fungsi ini dirancang untuk kebutuhan training model di mana seluruh
    data harus dimuat ke memori. Menggunakan teknik streaming/pagination
    internal dengan batch 1000 record per request untuk menghindari
    timeout pada dataset besar.

    Args:
        dataset_id (int): ID dataset yang seluruh datanya ingin diambil.

    Returns:
        list[dict]: Daftar seluruh record dengan field:
            - full_text         (str): Teks komentar asli.
            - label             (str): Label kelas ('positif'/'negatif').
            - preprocessed_text (str): Hasil preprocessing (bisa kosong '').

    Catatan:
        Fungsi ini dapat memakan waktu cukup lama dan memori yang besar
        untuk dataset dengan ribuan hingga puluhan ribu record.
        Loop akan berhenti otomatis ketika tidak ada lagi data yang
        dikembalikan atau jumlah data dalam satu batch kurang dari 1000.
    """
    db = get_db()
    all_data = []
    batch = 1000
    offset = 0
    while True:
        res = (db.table('raw_data')
               .select('full_text,label,preprocessed_text')
               .eq('dataset_id', dataset_id)
               .range(offset, offset + batch - 1)
               .execute())
        if not res.data:
            break
        all_data.extend(res.data)
        if len(res.data) < batch:
            break
        offset += batch
    return all_data


def update_preprocessed_text(row_id, preprocessed):
    """
    Memperbarui kolom 'preprocessed_text' untuk satu record raw_data.

    Dapat digunakan jika preprocessing perlu dijalankan ulang untuk
    satu baris tertentu tanpa harus memproses ulang seluruh dataset.

    Args:
        row_id      (int): ID unik record di tabel 'raw_data'.
        preprocessed (str): String hasil preprocessing baru yang akan disimpan.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.
    """
    db = get_db()
    db.table('raw_data').update({'preprocessed_text': preprocessed}).eq('id', row_id).execute()


# ============================================================
# EXPERIMENTS
# Fungsi-fungsi untuk mengelola eksperimen klasifikasi.
# Tabel 'experiments' menyimpan konfigurasi (parameter) dan hasil
# evaluasi dari setiap percobaan training model ID3 Modifikasi.
# ============================================================

def insert_experiment(dataset_id, proportion_label, train_ratio, test_ratio, max_depth, use_smote):
    """
    Membuat record eksperimen baru dengan status awal 'pending'.

    Dipanggil tepat sebelum proses training dimulai untuk setiap
    kombinasi proporsi split data yang dipilih admin. Status eksperimen
    akan diperbarui menjadi 'running', 'done', atau 'error' seiring
    berjalannya proses training.

    Args:
        dataset_id       (int)  : ID dataset yang digunakan untuk eksperimen.
        proportion_label (str)  : Label proporsi split, misal '80:20' atau '70:30'.
        train_ratio      (float): Proporsi data training, misal 0.8 untuk 80%.
        test_ratio       (float): Proporsi data testing, misal 0.2 untuk 20%.
        max_depth        (int)  : Parameter kedalaman maksimum pohon keputusan ID3.
        use_smote        (bool) : True jika SMOTE oversampling diaktifkan.

    Returns:
        dict | None: Record eksperimen yang baru dibuat (termasuk ID-nya),
                     atau None jika insert gagal.
    """
    db = get_db()
    res = db.table('experiments').insert({
        'dataset_id': dataset_id,
        'proportion_label': proportion_label,
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'max_depth': max_depth,
        'use_smote': use_smote,
        'status': 'pending'
    }).execute()
    return res.data[0] if res.data else None


def update_experiment_result(exp_id, result: dict):
    """
    Menyimpan hasil evaluasi eksperimen ke database dan mengubah status menjadi 'done'.

    Dipanggil setelah proses training dan evaluasi model berhasil diselesaikan.
    Semua metrik evaluasi (akurasi, presisi, recall, F1-score, confusion matrix,
    dan waktu eksekusi) disimpan dalam satu operasi update.

    Args:
        exp_id  (int) : ID eksperimen yang akan diperbarui.
        result  (dict): Dictionary hasil dari fungsi run_experiment() di pipeline.py,
                        yang wajib mengandung key-key berikut:
            - accuracy       (float): Nilai akurasi keseluruhan.
            - precision      (float): Nilai presisi berbobot (weighted).
            - recall         (float): Nilai recall berbobot (weighted).
            - f1             (float): Nilai F1-score berbobot (weighted).
            - tp             (int)  : Jumlah True Positive.
            - tn             (int)  : Jumlah True Negative.
            - fp             (int)  : Jumlah False Positive.
            - fn             (int)  : Jumlah False Negative.
            - train_count    (int)  : Jumlah data training (setelah SMOTE jika aktif).
            - test_count     (int)  : Jumlah data testing.
            - execution_time (float): Total waktu eksekusi (training + prediksi) dalam detik.
            - train_time     (float): Waktu training saja dalam detik.
            - predict_time   (float): Waktu prediksi saja dalam detik.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.
    """
    db = get_db()
    db.table('experiments').update({
        'accuracy':        result['accuracy'],
        'precision_score': result['precision'],
        'recall_score':    result['recall'],
        'f1_score':        result['f1'],
        'tp':              result['tp'],
        'tn':              result['tn'],
        'fp':              result['fp'],
        'fn':              result['fn'],
        'train_count':     result['train_count'],
        'test_count':      result['test_count'],
        'execution_time':  result.get('execution_time'),
        'train_time':      result.get('train_time'),
        'predict_time':    result.get('predict_time'),
        'status':          'done',
        'finished_at':     datetime.now(timezone.utc).isoformat()
    }).eq('id', exp_id).execute()


def update_experiment_error(exp_id, msg):
    """
    Menandai eksperimen sebagai gagal dan menyimpan pesan error-nya.

    Dipanggil di dalam blok except ketika terjadi exception selama
    proses training atau evaluasi. Pesan error dipotong maksimal
    500 karakter agar sesuai dengan batas kolom di database.

    Args:
        exp_id (int): ID eksperimen yang gagal dijalankan.
        msg    (str): Pesan error atau traceback yang menjelaskan penyebab kegagalan.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.
    """
    db = get_db()
    db.table('experiments').update({
        'status': 'error',
        'error_msg': str(msg)[:500],
        'finished_at': datetime.now(timezone.utc).isoformat()
    }).eq('id', exp_id).execute()


def update_experiment_running(exp_id):
    """
    Mengubah status eksperimen menjadi 'running'.

    Dipanggil setelah insert eksperimen berhasil dan tepat sebelum
    proses training dimulai, untuk memberi indikasi visual pada
    antarmuka admin bahwa eksperimen sedang berjalan.

    Args:
        exp_id (int): ID eksperimen yang statusnya akan diubah.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.
    """
    db = get_db()
    db.table('experiments').update({'status': 'running'}).eq('id', exp_id).execute()


def get_experiments_by_dataset(dataset_id):
    """
    Mengambil semua eksperimen yang menggunakan dataset tertentu.

    Digunakan pada halaman detail dataset untuk menampilkan semua
    eksperimen yang pernah dijalankan menggunakan dataset tersebut,
    beserta status dan hasilnya.

    Args:
        dataset_id (int): ID dataset yang eksperimennya ingin ditampilkan.

    Returns:
        list[dict]: Daftar record eksperimen, diurutkan dari yang terbaru.
                    Mengembalikan list kosong [] jika belum ada eksperimen.
    """
    db = get_db()
    res = (db.table('experiments')
           .select('*')
           .eq('dataset_id', dataset_id)
           .order('created_at', desc=True)
           .execute())
    return res.data or []


def get_experiment_by_id(exp_id):
    """
    Mengambil satu record eksperimen berdasarkan ID-nya.

    Digunakan untuk menampilkan halaman detail eksperimen atau
    memvalidasi keberadaan eksperimen sebelum operasi tertentu
    (misal: menetapkan sebagai model terbaik).

    Args:
        exp_id (int): ID unik eksperimen yang ingin diambil.

    Returns:
        dict | None: Record eksperimen jika ditemukan, atau None jika tidak ada.
    """
    db = get_db()
    res = db.table('experiments').select('*').eq('id', exp_id).execute()
    return res.data[0] if res.data else None


def get_best_experiment():
    """
    Mengambil eksperimen terbaik yang telah ditandai oleh admin.

    Strategi pengambilan menggunakan dua tahap:
    1. Utama   : Mencari eksperimen yang sudah ditandai is_best=True dengan
                 status 'done', diurutkan berdasarkan akurasi tertinggi.
    2. Fallback: Jika tidak ada yang ditandai, otomatis mengambil eksperimen
                 dengan akurasi tertinggi di antara semua yang berstatus 'done'.

    Fungsi ini digunakan oleh:
        - Halaman publik klasifikasi (classify) untuk memuat model yang aktif.
        - Halaman beranda (index) untuk menampilkan informasi model terbaik.
        - Endpoint API klasifikasi (/api/classify).

    Returns:
        dict | None: Record eksperimen terbaik, atau None jika belum ada
                     eksperimen yang selesai (status 'done') sama sekali.
    """
    db = get_db()
    res = (db.table('experiments')
           .select('*')
           .eq('is_best', True)
           .eq('status', 'done')
           .order('accuracy', desc=True)
           .limit(1)
           .execute())
    if res.data:
        return res.data[0]
    # Fallback: ambil yang akurasi tertinggi
    res2 = (db.table('experiments')
            .select('*')
            .eq('status', 'done')
            .order('accuracy', desc=True)
            .limit(1)
            .execute())
    return res2.data[0] if res2.data else None


def set_best_experiment(exp_id):
    """
    Menetapkan satu eksperimen sebagai model terbaik (is_best=True).

    Fungsi ini mereset flag is_best pada SEMUA eksperimen lain menjadi
    False terlebih dahulu, kemudian menetapkan is_best=True hanya pada
    eksperimen dengan ID yang diberikan. Dengan demikian dipastikan
    hanya ada satu eksperimen yang berstatus 'best' pada satu waktu.

    Args:
        exp_id (int): ID eksperimen yang akan dijadikan model terbaik.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.

    Catatan:
        Setelah fungsi ini dipanggil, model yang tersimpan untuk exp_id
        ini akan digunakan oleh sistem untuk semua prediksi publik.
    """
    db = get_db()
    db.table('experiments').update({'is_best': False}).neq('id', exp_id).execute()
    db.table('experiments').update({'is_best': True}).eq('id', exp_id).execute()


def get_all_done_experiments():
    """
    Mengambil semua eksperimen yang telah selesai (status='done').

    Digunakan pada halaman daftar eksperimen admin dan halaman statistik
    untuk menampilkan perbandingan hasil antar eksperimen, diurutkan
    dari akurasi tertinggi ke terendah.

    Returns:
        list[dict]: Daftar semua eksperimen berstatus 'done',
                    diurutkan berdasarkan akurasi secara descending.
                    Mengembalikan list kosong [] jika tidak ada.
    """
    db = get_db()
    res = (db.table('experiments')
           .select('*')
           .eq('status', 'done')
           .order('accuracy', desc=True)
           .execute())
    return res.data or []


# ============================================================
# EXPERIMENT DETAILS
# Fungsi-fungsi untuk menyimpan dan mengambil metrik evaluasi
# per kelas (per-class metrics) dari setiap eksperimen.
# Tabel 'experiment_details' menyimpan precision, recall, F1-score,
# dan support untuk setiap kelas ('positif'/'negatif').
# ============================================================

def insert_experiment_details(exp_id, report: dict, class_names: list):
    """
    Menyimpan metrik evaluasi per kelas dari laporan klasifikasi.

    Mengambil data dari dictionary report yang dihasilkan oleh
    sklearn.metrics.classification_report() dan menyimpannya
    per baris per kelas ke tabel 'experiment_details'.

    Args:
        exp_id      (int)       : ID eksperimen induk.
        report      (dict)      : Dictionary hasil classification_report() dengan
                                  output_dict=True dari sklearn. Setiap key adalah
                                  nama kelas (misal 'positif', 'negatif') dan
                                  value-nya adalah dict berisi 'precision', 'recall',
                                  'f1-score', dan 'support'.
        class_names (list[str]) : Daftar nama kelas yang akan disimpan detailnya.
                                  Hanya kelas yang ada di list ini yang akan diproses.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.

    Catatan:
        Kelas yang ada di class_names tetapi tidak ada di report akan
        dilewati (tidak menghasilkan error).
    """
    db = get_db()
    records = []
    for cls in class_names:
        if cls in report:
            r = report[cls]
            records.append({
                'experiment_id': exp_id,
                'class_label': cls,
                'precision_val': float(r.get('precision', 0)),
                'recall_val': float(r.get('recall', 0)),
                'f1_val': float(r.get('f1-score', 0)),
                'support': int(r.get('support', 0))
            })
    if records:
        db.table('experiment_details').insert(records).execute()


def get_experiment_details(exp_id):
    """
    Mengambil semua record metrik per kelas untuk satu eksperimen.

    Digunakan pada halaman detail eksperimen untuk menampilkan tabel
    metrik evaluasi per kelas (precision, recall, F1, support) secara
    terpisah untuk kelas 'positif' dan 'negatif'.

    Args:
        exp_id (int): ID eksperimen yang detailnya ingin diambil.

    Returns:
        list[dict]: Daftar record detail per kelas. Setiap dict berisi:
            - class_label   (str)  : Nama kelas ('positif' atau 'negatif').
            - precision_val (float): Nilai presisi untuk kelas tersebut.
            - recall_val    (float): Nilai recall untuk kelas tersebut.
            - f1_val        (float): Nilai F1-score untuk kelas tersebut.
            - support       (int)  : Jumlah sampel aktual untuk kelas tersebut.
        Mengembalikan list kosong [] jika tidak ada detail tersimpan.
    """
    db = get_db()
    res = db.table('experiment_details').select('*').eq('experiment_id', exp_id).execute()
    return res.data or []


# ============================================================
# DEPTH RESULTS
# Fungsi-fungsi untuk menyimpan dan mengambil hasil eksperimen
# sweep max_depth (depth 1 sampai 20). Digunakan untuk analisis
# pengaruh kedalaman pohon terhadap akurasi model.
# ============================================================

def insert_depth_results(exp_id, depth_results: list):
    """
    Menyimpan hasil akurasi untuk setiap nilai max_depth dari sweep experiment.

    Dipanggil setelah fungsi run_depth_sweep() di pipeline.py selesai
    mengevaluasi model untuk depth 1 hingga 20. Data ini digunakan untuk
    membuat grafik akurasi vs. kedalaman pohon pada halaman detail eksperimen.

    Args:
        exp_id        (int)       : ID eksperimen induk yang menjalankan depth sweep.
        depth_results (list[dict]): Daftar hasil per depth dari run_depth_sweep(),
                                    di mana setiap dict mengandung:
            - max_depth      (int)  : Nilai kedalaman pohon yang diuji.
            - accuracy       (float): Akurasi pada depth tersebut.
            - precision      (float): Presisi berbobot pada depth tersebut.
            - recall         (float): Recall berbobot pada depth tersebut.
            - f1             (float): F1-score berbobot pada depth tersebut.
            - execution_time (float): Waktu eksekusi dalam detik (opsional).

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.
    """
    db = get_db()
    records = [{
        'experiment_id':  exp_id,
        'max_depth':      r['max_depth'],
        'accuracy':       r['accuracy'],
        'precision_score': r['precision'],
        'recall_score':   r['recall'],
        'f1_score':       r['f1'],
        'execution_time': r.get('execution_time'),
    } for r in depth_results]
    if records:
        db.table('depth_results').insert(records).execute()


def get_depth_results(exp_id):
    """
    Mengambil semua hasil sweep max_depth untuk satu eksperimen.

    Data yang dikembalikan digunakan untuk merender grafik
    "Akurasi vs. Max Depth" dan "Waktu Eksekusi vs. Max Depth"
    pada halaman detail eksperimen.

    Args:
        exp_id (int): ID eksperimen yang hasil sweep-nya ingin diambil.

    Returns:
        list[dict]: Daftar hasil per depth, diurutkan berdasarkan
                    nilai max_depth dari yang terkecil (1) ke terbesar (20).
                    Mengembalikan list kosong [] jika sweep belum dijalankan.
    """
    db = get_db()
    res = (db.table('depth_results')
           .select('*')
           .eq('experiment_id', exp_id)
           .order('max_depth')
           .execute())
    return res.data or []


# ============================================================
# PREDICTIONS
# Fungsi-fungsi untuk menyimpan dan mengambil riwayat prediksi.
# Tabel 'predictions' mencatat setiap permintaan klasifikasi yang
# dilakukan pengguna, baik melalui form web maupun API.
# ============================================================

def insert_prediction(input_text, preprocessed, label, confidence, exp_id):
    """
    Menyimpan satu record hasil prediksi ke tabel 'predictions'.

    Dipanggil setiap kali pengguna berhasil mengklasifikasikan teks,
    baik melalui halaman /classify maupun endpoint API /api/classify.
    Teks input dipotong maksimal 1000 karakter dan preprocessed_text
    maksimal 500 karakter untuk menjaga ukuran database.

    Args:
        input_text   (str)  : Teks asli yang diinputkan pengguna.
        preprocessed (str)  : Hasil preprocessing dari teks input.
        label        (str)  : Label hasil prediksi ('positif' atau 'negatif').
        confidence   (float): Nilai kepercayaan prediksi (0.0 - 1.0).
        exp_id       (int)  : ID eksperimen (model) yang digunakan untuk prediksi ini.

    Returns:
        dict | None: Record prediksi yang baru disimpan (termasuk ID dan
                     timestamp), atau None jika insert gagal.
    """
    db = get_db()
    res = db.table('predictions').insert({
        'input_text': input_text[:1000],
        'preprocessed_text': preprocessed[:500] if preprocessed else '',
        'predicted_label': label,
        'confidence': float(confidence),
        'experiment_id': exp_id
    }).execute()
    return res.data[0] if res.data else None


def get_recent_predictions(limit=50):
    """
    Mengambil prediksi terbaru, diurutkan dari yang paling baru.

    Digunakan pada halaman riwayat prediksi admin untuk menampilkan
    aktivitas klasifikasi terbaru dari pengguna.

    Args:
        limit (int): Jumlah maksimal prediksi yang diambil. Default: 50.

    Returns:
        list[dict]: Daftar record prediksi terbaru. Setiap dict berisi
                    semua kolom tabel 'predictions' termasuk timestamp.
                    Mengembalikan list kosong [] jika belum ada prediksi.
    """
    db = get_db()
    res = (db.table('predictions')
           .select('*')
           .order('created_at', desc=True)
           .limit(limit)
           .execute())
    return res.data or []


def get_prediction_stats():
    """
    Menghitung statistik distribusi label dari seluruh prediksi yang tersimpan.

    Mengambil semua nilai kolom 'predicted_label' dan menghitung jumlah
    prediksi per kelas. Digunakan pada dashboard admin dan halaman
    statistik untuk menampilkan proporsi kelas yang diprediksi.

    Returns:
        dict: Dictionary dengan key-key berikut:
            - total   (int): Total seluruh prediksi yang pernah dilakukan.
            - positif (int): Jumlah prediksi berlabel 'positif'.
            - negatif (int): Jumlah prediksi berlabel 'negatif'
                             (dihitung sebagai total - positif).

    Catatan:
        Fungsi ini memuat SEMUA record prediksi ke memori untuk dihitung.
        Pertimbangkan optimasi dengan query COUNT jika data prediksi sangat besar.
    """
    db = get_db()
    res = db.table('predictions').select('predicted_label').execute()
    data = res.data or []
    total = len(data)
    positif = sum(1 for d in data if d['predicted_label'] == 'positif')
    negatif = total - positif
    return {'total': total, 'positif': positif, 'negatif': negatif}


# ============================================================
# STATISTICS
# Fungsi untuk mengumpulkan berbagai statistik ringkasan
# yang ditampilkan pada dashboard utama admin.
# ============================================================

def get_dashboard_stats():
    """
    Mengumpulkan semua statistik ringkasan untuk ditampilkan di dashboard admin.

    Memanggil beberapa fungsi database lainnya secara berurutan dan
    menggabungkan hasilnya dalam satu dictionary. Digunakan sebagai
    satu titik panggil tunggal untuk kebutuhan data halaman dashboard.

    Returns:
        dict: Dictionary yang berisi statistik ringkasan berikut:
            - total_datasets    (int)        : Jumlah total dataset yang diunggah.
            - total_experiments (int)        : Jumlah total eksperimen berstatus 'done'.
            - total_predictions (int)        : Jumlah total prediksi yang pernah dilakukan.
            - best_accuracy     (float|None) : Nilai akurasi model terbaik, atau None
                                               jika belum ada eksperimen selesai.
            - best_experiment   (dict|None)  : Record lengkap eksperimen terbaik.
            - pred_positif      (int)        : Jumlah prediksi berlabel 'positif'.
            - pred_negatif      (int)        : Jumlah prediksi berlabel 'negatif'.

    Catatan:
        Fungsi ini melakukan beberapa query database sekaligus, sehingga
        sebaiknya tidak dipanggil berulang kali dalam satu request yang sama.
    """
    db = get_db()
    datasets = get_all_datasets()
    experiments = get_all_done_experiments()
    pred_stats = get_prediction_stats()
    best = get_best_experiment()
    return {
        'total_datasets': len(datasets),
        'total_experiments': len(experiments),
        'total_predictions': pred_stats['total'],
        'best_accuracy': best['accuracy'] if best else None,
        'best_experiment': best,
        'pred_positif': pred_stats['positif'],
        'pred_negatif': pred_stats['negatif'],
    }