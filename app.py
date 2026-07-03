"""
app.py
======
Aplikasi web Flask untuk sistem klasifikasi respon masyarakat terhadap isu
pelecehan seksual secara verbal pada platform X menggunakan ID3 Modifikasi.

Modul ini adalah entry point utama aplikasi. Mengelola semua routing HTTP,
autentikasi admin, eksekusi eksperimen di background thread, dan endpoint API.

Struktur akses:
    ┌─ Publik (tanpa login) ─────────────────────────────────────────────────┐
    │  GET  /             → Halaman beranda                                  │
    │  GET  /classify     → Form klasifikasi teks (pengguna umum)            │
    │  POST /classify     → Proses klasifikasi dan tampilkan hasil           │
    │  GET  /about        → Halaman tentang sistem                           │
    │  POST /api/classify → REST API klasifikasi (JSON request/response)     │
    │  GET  /api/stats    → REST API statistik dashboard (JSON)             │
    └────────────────────────────────────────────────────────────────────────┘
    ┌─ Admin (perlu login) ──────────────────────────────────────────────────┐
    │  GET  /admin                     → Dashboard admin                    │
    │  GET  /admin/datasets            → Daftar dataset                     │
    │  GET/POST /admin/datasets/upload → Upload dataset CSV baru            │
    │  GET  /admin/datasets/<id>       → Detail dataset + paginasi          │
    │  POST /admin/datasets/<id>/delete→ Hapus dataset                      │
    │  GET  /admin/experiments         → Daftar semua eksperimen            │
    │  GET/POST /admin/experiments/run → Form + eksekusi training model     │
    │  GET  /admin/experiments/progress/<task_id> → Halaman monitoring      │
    │  GET  /admin/experiments/<id>    → Detail hasil eksperimen            │
    │  POST /admin/experiments/<id>/set-best → Tandai model terbaik        │
    │  GET  /admin/statistics          → Statistik prediksi & eksperimen   │
    │  GET  /admin/predictions         → Riwayat prediksi pengguna          │
    │  GET  /api/task-status/<task_id> → Status background task (JSON)     │
    │  GET  /api/experiments/<id>/depth-results → Data grafik depth (JSON) │
    │  GET  /api/experiments/<id>/model-detail  → Info model pickle (JSON) │
    └────────────────────────────────────────────────────────────────────────┘

Pengelolaan task asinkron:
    Proses training model dijalankan dalam background thread agar tidak
    memblokir web server. Status setiap task disimpan dalam dictionary
    in-memory (_task_status) dan dapat dipantau melalui polling dari
    frontend menggunakan endpoint /api/task-status/<task_id>.

Konfigurasi (Environment Variables):
    SECRET_KEY       : Secret key Flask untuk session (default: 'rizka_skripsi_2026').
    ADMIN_USERNAME   : Username admin panel (default: 'admin').
    ADMIN_PASSWORD   : Password admin panel (default: 'rizka2026').
    SUPABASE_URL     : URL project Supabase (digunakan oleh database.py).
    SUPABASE_KEY     : API key Supabase (digunakan oleh database.py).

Dependensi:
    - Flask          : Web framework utama.
    - pandas         : Parsing dan validasi file CSV saat upload.
    - python-dotenv  : Membaca konfigurasi dari file .env.
    - werkzeug       : secure_filename untuk sanitasi nama file upload.
    - threading      : Menjalankan training model di background thread.

Author   : Rizka Mardiah Putri Buyung Lubis (220170183)
Institusi: Universitas Malikussaleh, 2026
"""

import os
import io
import csv
import json
import threading
import traceback
import pandas as pd
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Muat variabel lingkungan dari file .env sebelum inisialisasi aplikasi
load_dotenv()

# ── Inisialisasi aplikasi Flask ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rizka_skripsi_2026')
# Batasi ukuran file upload maksimal 50MB untuk mencegah file terlalu besar
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB dalam bytes

# ── Direktori penyimpanan file upload ─────────────────────────────────────────
# File CSV yang diunggah admin disimpan sementara di sini sebelum diproses
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Buat direktori jika belum ada

# ── Kredensial admin ──────────────────────────────────────────────────────────
# Dibaca dari environment variable dengan nilai default untuk development.
# PENTING: Ubah nilai default ini sebelum deploy ke production!
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'rizka2026')

# ── Status tracking untuk background tasks ────────────────────────────────────
# Dictionary in-memory untuk menyimpan status setiap task training yang berjalan.
# Key   : task_id (string unik berformat "task_{dataset_id}_{timestamp}")
# Value : dict berisi 'status', 'progress' (0-100), dan 'message'
# Catatan: Data ini hilang jika server direstart. Untuk produksi, pertimbangkan
#          menggunakan Redis atau penyimpanan persisten lainnya.
_task_status = {}


# ============================================================
# DEKORATOR AUTENTIKASI
# ============================================================

def login_required(f):
    """
    Dekorator untuk melindungi route admin agar hanya dapat diakses setelah login.

    Menggunakan pola dekorator Flask standar dengan functools.wraps untuk
    mempertahankan metadata fungsi asli (nama, docstring, dll.).

    Cara kerja:
        1. Cek apakah kunci 'admin_logged_in' ada dan bernilai True di session Flask.
        2. Jika YA  → Lanjutkan ke fungsi route yang dilindungi.
        3. Jika TIDAK → Tampilkan pesan flash peringatan dan redirect ke halaman login.

    Penggunaan:
        @app.route('/admin/halaman-rahasia')
        @login_required
        def halaman_rahasia():
            ...

    Args:
        f (function): Fungsi view Flask yang akan dilindungi.

    Returns:
        function: Fungsi wrapper yang menambahkan logika cek autentikasi
                  sebelum memanggil fungsi view asli.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# PUBLIC ROUTES
# Route-route yang dapat diakses tanpa login oleh siapapun.
# ============================================================

@app.route('/')
def index():
    """
    Menampilkan halaman beranda publik sistem.

    Memuat informasi model terbaik (eksperimen dengan is_best=True atau
    akurasi tertinggi) dan statistik ringkasan prediksi untuk ditampilkan
    sebagai highlight kepada pengunjung.

    Template: templates/index.html

    Context yang dikirim ke template:
        best  (dict|None): Record eksperimen terbaik dari database.
        stats (dict)     : Statistik prediksi {'total', 'positif', 'negatif'}.
    """
    from database import get_best_experiment, get_prediction_stats
    best = get_best_experiment()
    stats = get_prediction_stats()
    return render_template('index.html', best=best, stats=stats)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """
    Halaman klasifikasi teks untuk pengguna umum (tanpa autentikasi).

    Menyediakan form input teks dan menampilkan hasil klasifikasi apakah
    teks mengandung unsur pelecehan seksual verbal (positif) atau tidak (negatif).

    Alur kerja (POST request):
        1. Validasi input teks (tidak boleh kosong, minimal 5 karakter).
        2. Panggil predict_single() dari pipeline menggunakan model terbaik.
        3. Simpan hasil prediksi ke database (tabel 'predictions').
        4. Tampilkan hasil beserta label dan confidence ke pengguna.

    Penanganan error:
        - Model belum tersedia → Tampilkan pesan error informatif.
        - Teks terlalu pendek → Validasi form dengan pesan error.
        - Error prediksi → Tangkap exception dan tampilkan ke pengguna.

    Template: templates/classify.html

    Context yang dikirim ke template:
        result     (dict|None) : Hasil prediksi {'label', 'confidence', 'preprocessed'}.
        error      (str|None)  : Pesan error jika ada.
        input_text (str)       : Teks yang diinput pengguna (untuk re-populate form).
        best       (dict|None) : Info model terbaik yang digunakan.
    """
    from database import get_best_experiment, insert_prediction
    from pipeline import predict_single

    result = None
    error = None
    input_text = ''

    # Cek ketersediaan model terbaik sebelum menerima input
    best = get_best_experiment()
    if not best:
        error = 'Model belum tersedia. Hubungi administrator.'

    if request.method == 'POST' and best:
        input_text = request.form.get('text', '').strip()

        # Validasi input
        if not input_text:
            error = 'Teks tidak boleh kosong.'
        elif len(input_text) < 5:
            error = 'Teks terlalu pendek (min. 5 karakter).'
        else:
            try:
                # Jalankan prediksi menggunakan model terbaik
                pred = predict_single(input_text, best['id'])
                if 'error' in pred:
                    error = pred['error']
                else:
                    # Simpan hasil prediksi ke database untuk analisis
                    insert_prediction(
                        input_text,
                        pred.get('preprocessed', ''),
                        pred['label'],
                        pred['confidence'],
                        best['id']
                    )
                    result = pred
            except Exception as e:
                error = f'Gagal melakukan klasifikasi: {str(e)}'

    return render_template('classify.html',
                           result=result, error=error,
                           input_text=input_text, best=best)


@app.route('/about')
def about():
    """
    Menampilkan halaman informasi tentang sistem.

    Berisi penjelasan tentang algoritma ID3 Modifikasi, metodologi penelitian,
    referensi jurnal, dan informasi pembuat sistem.

    Template: templates/about.html
    """
    return render_template('about.html')


# ============================================================
# AUTH ROUTES
# Route untuk proses login dan logout admin.
# ============================================================

@app.route('/admin/login', methods=['GET', 'POST'])
def login():
    """
    Menampilkan form login dan memproses autentikasi admin.

    GET  : Tampilkan halaman form login. Jika sudah login, redirect ke dashboard.
    POST : Validasi username dan password terhadap kredensial dari environment.
           Jika valid → set session 'admin_logged_in' = True dan redirect dashboard.
           Jika tidak valid → tampilkan pesan error di halaman yang sama.

    Autentikasi menggunakan perbandingan string sederhana (plain text password).
    Untuk production, pertimbangkan menggunakan hashing (werkzeug.security).

    Template: templates/login.html

    Context yang dikirim ke template:
        error (str|None): Pesan error jika login gagal.
    """
    # Jika sudah login, langsung redirect ke dashboard
    if session.get('admin_logged_in'):
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # Login berhasil: set flag di session Flask
            session['admin_logged_in'] = True
            session['admin_username'] = username
            flash('Login berhasil! Selamat datang, Admin.', 'success')
            return redirect(url_for('dashboard'))
        else:
            error = 'Username atau password salah.'

    return render_template('login.html', error=error)


@app.route('/admin/logout')
def logout():
    """
    Melakukan logout admin dengan menghapus seluruh data session.

    Menggunakan session.clear() untuk menghapus semua data session sekaligus,
    termasuk 'admin_logged_in' dan 'admin_username'. Setelah logout,
    pengguna diarahkan ke halaman beranda publik.
    """
    session.clear()
    flash('Berhasil logout.', 'info')
    return redirect(url_for('index'))


# ============================================================
# ADMIN ROUTES
# Route-route yang hanya dapat diakses setelah login admin.
# Semua route di sini menggunakan dekorator @login_required.
# ============================================================

@app.route('/admin')
@login_required
def dashboard():
    """
    Menampilkan halaman dashboard utama admin.

    Memuat statistik ringkasan sistem (jumlah dataset, eksperimen, prediksi,
    akurasi model terbaik) serta daftar dataset dan eksperimen terbaru.

    Template: templates/dashboard.html

    Context yang dikirim ke template:
        stats       (dict)      : Statistik ringkasan dari get_dashboard_stats().
        datasets    (list[dict]): Semua dataset yang tersimpan.
        experiments (list[dict]): Semua eksperimen yang selesai (status='done').
    """
    from database import get_dashboard_stats, get_all_datasets, get_all_done_experiments
    stats = get_dashboard_stats()
    datasets = get_all_datasets()
    experiments = get_all_done_experiments()
    return render_template('dashboard.html', stats=stats,
                           datasets=datasets, experiments=experiments)


@app.route('/admin/datasets')
@login_required
def datasets():
    """
    Menampilkan halaman daftar semua dataset yang telah diunggah.

    Template: templates/datasets.html

    Context yang dikirim ke template:
        datasets (list[dict]): Semua record dataset, diurutkan dari terbaru.
    """
    from database import get_all_datasets
    dsets = get_all_datasets()
    return render_template('datasets.html', datasets=dsets)


@app.route('/admin/datasets/upload', methods=['GET', 'POST'])
@login_required
def upload_dataset():
    """
    Menangani upload file CSV dataset baru dan proses awal preprocessing.

    GET  : Tampilkan form upload dataset.
    POST : Terima file CSV, validasi format, preproses semua teks, simpan ke database.

    Alur proses upload (POST):
        1. Terima file dari form (field 'file') dan catatan opsional ('notes').
        2. Validasi: file tidak kosong dan berekstensi .csv.
        3. Simpan file sementara ke UPLOAD_FOLDER.
        4. Baca CSV dengan mencoba beberapa encoding umum (utf-8, latin-1, dll.).
        5. Normalisasi nama kolom (lowercase, hapus BOM character).
        6. Validasi keberadaan kolom wajib: 'full_text' dan 'label'.
        7. Filter baris: hanya label 'positif'/'negatif' dan teks > 2 karakter.
        8. Hitung distribusi kelas (jumlah positif dan negatif).
        9. Jalankan preprocessing batch pada semua teks.
        10. Insert metadata dataset ke tabel 'datasets'.
        11. Insert semua baris data (dengan preprocessed_text) ke tabel 'raw_data'
            dalam batch 500 record.
        12. Redirect ke halaman daftar dataset dengan pesan sukses.

    Penanganan error:
        - File tidak dipilih atau bukan CSV → pesan flash danger, redirect form.
        - Gagal membaca CSV (encoding) → pesan flash danger.
        - Kolom tidak sesuai → pesan flash dengan informasi kolom yang ditemukan.
        - Dataset kosong setelah filter → pesan flash danger.
        - Exception umum → log traceback dan tampilkan pesan error.

    Format CSV yang diterima:
        Wajib memiliki dua kolom (nama case-insensitive):
            - full_text : Teks komentar asli dari platform X.
            - label     : Kelas komentar ('positif' atau 'negatif').

    Template: templates/upload_dataset.html
    """
    if request.method == 'POST':
        file = request.files.get('file')
        notes = request.form.get('notes', '')

        # Validasi: pastikan file dipilih
        if not file or file.filename == '':
            flash('Pilih file CSV terlebih dahulu.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)  # Sanitasi nama file
        if not filename.endswith('.csv'):
            flash('File harus berformat CSV.', 'danger')
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Coba beberapa encoding umum secara berurutan
            # untuk menangani file CSV dari berbagai sumber
            df = None
            for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                flash('Gagal membaca file CSV. Pastikan encoding UTF-8.', 'danger')
                return redirect(request.url)

            # Normalisasi nama kolom: lowercase dan hapus BOM character (\ufeff)
            # yang sering muncul pada file CSV yang disimpan Excel
            df.columns = [c.strip().replace('\ufeff', '').lower() for c in df.columns]

            # Validasi keberadaan kolom wajib
            if 'full_text' not in df.columns or 'label' not in df.columns:
                flash(
                    f'CSV harus memiliki kolom "full_text" dan "label". '
                    f'Kolom ditemukan: {list(df.columns)}',
                    'danger'
                )
                return redirect(request.url)

            # Ambil hanya kolom yang dibutuhkan dan bersihkan nilai null
            df = df[['full_text', 'label']].dropna()
            df['full_text'] = df['full_text'].astype(str).str.strip()
            df['label']     = df['label'].astype(str).str.strip().str.lower()

            # Filter: hanya simpan baris dengan label valid dan teks cukup panjang
            df = df[df['label'].isin(['positif', 'negatif'])]
            df = df[df['full_text'].str.len() > 2]

            # Hitung statistik distribusi kelas
            total   = len(df)
            positif = int((df['label'] == 'positif').sum())
            negatif = int((df['label'] == 'negatif').sum())

            if total == 0:
                flash('Dataset kosong atau format tidak sesuai.', 'danger')
                return redirect(request.url)

            # ── Preprocessing semua teks sebelum disimpan ke database ────────
            # Ini dilakukan saat upload agar tidak perlu diulang saat training.
            # Hasilnya disimpan di kolom 'preprocessed_text' tabel raw_data.
            from preprocessing import preprocess_batch
            texts = df['full_text'].tolist()
            print(f'[Upload] Preprocessing {total} teks...')
            preprocessed_texts = preprocess_batch(texts)
            print(f'[Upload] Preprocessing selesai.')

            # Simpan metadata dataset ke database
            from database import insert_dataset, insert_raw_data_batch
            ds = insert_dataset(filename, total, positif, negatif, notes)
            dataset_id = ds['id']

            # Siapkan record batch untuk insert ke raw_data
            # Setiap record menyertakan preprocessed_text yang sudah dihitung
            records = []
            for i, (_, row) in enumerate(df.iterrows()):
                prep = preprocessed_texts[i] if i < len(preprocessed_texts) else ''
                records.append({
                    'dataset_id':        dataset_id,
                    'full_text':         str(row['full_text'])[:2000],   # Batasi 2000 char
                    'label':             row['label'],
                    'preprocessed_text': prep[:1000] if prep else '',    # Batasi 1000 char
                })
            insert_raw_data_batch(records)

            flash(
                f'Dataset "{filename}" berhasil diupload! '
                f'({total} data: {positif} positif, {negatif} negatif) — '
                f'Preprocessing selesai ✓',
                'success'
            )
            return redirect(url_for('datasets'))

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            flash(f'Error memproses file: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('upload_dataset.html')


@app.route('/admin/datasets/<int:dataset_id>')
@login_required
def dataset_detail(dataset_id):
    """
    Menampilkan halaman detail satu dataset beserta preview data dan eksperimen.

    Menampilkan informasi metadata dataset, preview baris data dengan paginasi
    (20 baris per halaman), dan daftar eksperimen yang menggunakan dataset ini.

    Args (URL):
        dataset_id (int): ID dataset yang akan ditampilkan.

    Query parameters:
        page (int): Nomor halaman untuk paginasi data (default: 1).

    Template: templates/dataset_detail.html

    Context yang dikirim ke template:
        ds          (dict)      : Metadata dataset.
        raw_data    (list[dict]): Baris data untuk halaman saat ini.
        experiments (list[dict]): Eksperimen yang menggunakan dataset ini.
        page        (int)       : Nomor halaman saat ini.
        limit       (int)       : Jumlah baris per halaman (20).
    """
    from database import get_dataset_by_id, get_raw_data_by_dataset, get_experiments_by_dataset
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash('Dataset tidak ditemukan.', 'danger')
        return redirect(url_for('datasets'))

    # Paginasi: 20 baris per halaman
    page = int(request.args.get('page', 1))
    limit = 20
    offset = (page - 1) * limit
    raw_data = get_raw_data_by_dataset(dataset_id, limit=limit, offset=offset)
    experiments = get_experiments_by_dataset(dataset_id)

    return render_template('dataset_detail.html',
                           ds=ds, raw_data=raw_data,
                           experiments=experiments, page=page, limit=limit)


@app.route('/admin/datasets/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete_dataset_route(dataset_id):
    """
    Menghapus dataset beserta semua data terkait dari database.

    Menggunakan method POST (bukan DELETE) untuk kompatibilitas form HTML.
    Setelah penghapusan berhasil, redirect ke halaman daftar dataset.

    Args (URL):
        dataset_id (int): ID dataset yang akan dihapus.
    """
    from database import delete_dataset
    delete_dataset(dataset_id)
    flash('Dataset berhasil dihapus.', 'success')
    return redirect(url_for('datasets'))


@app.route('/admin/experiments')
@login_required
def experiments_list():
    """
    Menampilkan halaman daftar semua eksperimen yang telah selesai.

    Menandai eksperimen mana yang saat ini berstatus 'best' agar
    dapat dibedakan secara visual di tampilan tabel.

    Template: templates/experiments.html

    Context yang dikirim ke template:
        experiments (list[dict]): Semua eksperimen berstatus 'done'.
        best        (dict|None) : Eksperimen yang ditandai sebagai terbaik.
    """
    from database import get_all_done_experiments, get_best_experiment
    exps = get_all_done_experiments()
    best = get_best_experiment()
    return render_template('experiments.html', experiments=exps, best=best)


@app.route('/admin/experiments/run', methods=['GET', 'POST'])
@login_required
def run_experiment_route():
    """
    Menangani form konfigurasi dan menjalankan eksperimen training model.

    GET  : Tampilkan form pemilihan dataset, proporsi split, max_depth, dan opsi lain.
    POST : Validasi konfigurasi, buat task ID unik, dan jalankan training di background thread.

    Parameter form (POST):
        dataset_id       (int)       : ID dataset yang dipilih.
        proportions      (list[str]) : Daftar proporsi split, misal ['80:20', '70:30'].
                                       Bisa memilih lebih dari satu untuk multi-eksperimen.
        max_depth        (int)       : Nilai max_depth pohon keputusan (default: 3).
        use_smote        (str)       : 'on' jika SMOTE diaktifkan, kosong jika tidak.
        run_depth_sweep  (str)       : 'on' jika sweep depth 1-20 juga dijalankan.

    Mekanisme background task:
        Training dijalankan dalam threading.Thread terpisah (daemon=True) agar
        web server tidak blocking. Status task dipantau melalui _task_status dict
        yang diupdate oleh thread, dan dapat diquery melalui endpoint
        /api/task-status/<task_id> dengan polling dari frontend JavaScript.

    Alur background task (bg_task):
        1. Muat semua data dari database (get_all_raw_texts_labels).
        2. Cek apakah preprocessed_text sudah tersedia di database.
        3. Untuk setiap proporsi yang dipilih:
           a. Insert record eksperimen baru (status='pending').
           b. Update status ke 'running'.
           c. Jalankan run_experiment() dari pipeline.py.
           d. Simpan hasil ke database (update_experiment_result).
           e. Simpan metrik per kelas (insert_experiment_details).
           f. Simpan model ke file pkl (save_model).
           g. Jika run_depth_sweep=True: jalankan sweep depth 1-20 dan simpan hasilnya.
           h. Jika error: update status ke 'error' dengan pesan error.
        4. Update status task ke 'done' dan progress ke 100.

    Template: templates/run_experiment.html

    Context yang dikirim ke template:
        datasets (list[dict]): Daftar dataset yang tersedia untuk dipilih.
    """
    from database import get_all_datasets
    datasets_list = get_all_datasets()

    if request.method == 'POST':
        # Ambil dan parse parameter dari form
        dataset_id      = int(request.form.get('dataset_id'))
        proportions     = request.form.getlist('proportions')   # Bisa ['80:20', '70:30']
        max_depth       = int(request.form.get('max_depth', 3))
        use_smote       = request.form.get('use_smote') == 'on'
        run_depth_sweep = request.form.get('run_depth_sweep') == 'on'

        if not proportions:
            flash('Pilih minimal satu proporsi data.', 'danger')
            return redirect(request.url)

        # Buat task ID unik berdasarkan dataset_id dan timestamp Unix
        task_id = f"task_{dataset_id}_{int(datetime.now().timestamp())}"
        _task_status[task_id] = {
            'status':   'running',
            'progress': 0,
            'message':  'Memulai...'
        }

        def bg_task():
            """
            Fungsi yang dijalankan dalam background thread untuk proses training.

            Mengupdate _task_status secara berkala agar frontend dapat memantau
            kemajuan melalui polling. Semua exception ditangkap dan status
            diupdate ke 'error' agar tidak ada thread yang diam tanpa kejelasan.
            """
            try:
                from database import (
                    get_all_raw_texts_labels, insert_experiment,
                    update_experiment_result, update_experiment_error,
                    update_experiment_running, insert_experiment_details,
                    insert_depth_results
                )
                from pipeline import run_experiment, run_depth_sweep as rds, save_model

                _task_status[task_id]['message'] = 'Memuat data dari database...'

                # Muat semua data (teks, label, preprocessed_text) dari database
                rows   = get_all_raw_texts_labels(dataset_id)
                texts  = [r['full_text'] for r in rows]
                labels = [r['label'] for r in rows]

                # Optimalkan: gunakan preprocessed_text dari DB jika sudah ada
                # (dihitung saat upload) agar tidak perlu preprocessing ulang
                preprocessed_from_db = [r.get('preprocessed_text') or '' for r in rows]
                has_preprocessed = sum(1 for t in preprocessed_from_db if t and t.strip()) > 0

                if has_preprocessed:
                    _task_status[task_id]['message'] = (
                        f'Ditemukan {sum(1 for t in preprocessed_from_db if t)} '
                        f'teks sudah dipreproses, digunakan langsung...'
                    )
                    prep_texts = preprocessed_from_db
                else:
                    # Tidak ada preprocessed_text di DB → pipeline akan preprocess sendiri
                    _task_status[task_id]['message'] = 'Preprocessing teks...'
                    prep_texts = None

                total_tasks = len(proportions)

                # Iterasi setiap proporsi yang dipilih admin
                for idx, prop in enumerate(proportions):
                    _task_status[task_id]['message'] = (
                        f'Training proporsi {prop} ({idx+1}/{total_tasks})...'
                    )
                    # Progress bar: 0-80% untuk training, 80-100% untuk selesai
                    _task_status[task_id]['progress'] = int((idx / total_tasks) * 80)

                    # Parse proporsi: "80:20" → train_r = 0.8
                    parts   = prop.split(':')
                    train_r = int(parts[0]) / 100.0

                    # Buat record eksperimen dan tandai sebagai 'running'
                    exp    = insert_experiment(dataset_id, prop, train_r, 1-train_r, max_depth, use_smote)
                    exp_id = exp['id']
                    update_experiment_running(exp_id)

                    try:
                        # Jalankan pipeline training lengkap
                        result = run_experiment(
                            texts, labels, train_r, max_depth, use_smote,
                            preprocessed_texts=prep_texts
                        )

                        # Simpan semua hasil ke database
                        update_experiment_result(exp_id, result)
                        insert_experiment_details(exp_id, result['report'], result['class_names'])
                        save_model(exp_id, result['vectorizer'], result['model'], result['le'])

                        # Opsional: jalankan sweep depth 1-20 untuk grafik analisis
                        if run_depth_sweep:
                            _task_status[task_id]['message'] = f'Sweep max_depth untuk {prop}...'
                            depth_res = rds(
                                texts, labels, train_r, use_smote=use_smote,
                                preprocessed_texts=prep_texts
                            )
                            insert_depth_results(exp_id, depth_res)

                    except Exception as e:
                        # Catat error pada eksperimen ini dan lanjutkan ke proporsi berikutnya
                        update_experiment_error(exp_id, str(e))
                        _task_status[task_id]['message'] = f'Error pada {prop}: {str(e)}'

                # Semua proporsi selesai diproses
                _task_status[task_id]['status']   = 'done'
                _task_status[task_id]['progress'] = 100
                _task_status[task_id]['message']  = 'Selesai!'

            except Exception as e:
                # Error fatal di level atas (misal: gagal koneksi database)
                _task_status[task_id]['status']    = 'error'
                _task_status[task_id]['message']   = f'Error: {str(e)}'
                _task_status[task_id]['traceback'] = traceback.format_exc()

        # Jalankan bg_task dalam background thread (daemon=True agar
        # thread otomatis berhenti jika main thread berhenti)
        t = threading.Thread(target=bg_task)
        t.daemon = True
        t.start()

        flash(f'Eksperimen dimulai! Task ID: {task_id}', 'info')
        return redirect(url_for('experiment_progress', task_id=task_id))

    return render_template('run_experiment.html', datasets=datasets_list)


@app.route('/admin/experiments/progress/<task_id>')
@login_required
def experiment_progress(task_id):
    """
    Menampilkan halaman monitoring progress eksperimen yang sedang berjalan.

    Halaman ini menampilkan progress bar dan pesan status yang diperbarui
    secara real-time menggunakan JavaScript polling ke endpoint
    /api/task-status/<task_id> setiap beberapa detik.

    Args (URL):
        task_id (str): ID task yang dibuat saat memulai eksperimen.

    Template: templates/experiment_progress.html

    Context yang dikirim ke template:
        task_id (str): ID task untuk digunakan oleh JavaScript polling.
    """
    return render_template('experiment_progress.html', task_id=task_id)


@app.route('/api/task-status/<task_id>')
@login_required
def task_status_api(task_id):
    """
    Endpoint API untuk mengecek status background task training (JSON).

    Dipanggil oleh JavaScript di halaman experiment_progress.html secara
    periodik (polling) untuk mendapatkan pembaruan status, progress, dan
    pesan terbaru dari background training thread.

    Args (URL):
        task_id (str): ID task yang statusnya ingin dicek.

    Returns:
        JSON: Dictionary status task dengan format:
            {
                "status"  : "running" | "done" | "error" | "not_found",
                "progress": int (0-100),
                "message" : str (pesan status terakhir),
                "traceback": str (hanya jika status='error', opsional)
            }
    """
    status = _task_status.get(task_id, {'status': 'not_found'})
    return jsonify(status)


@app.route('/admin/experiments/<int:exp_id>')
@login_required
def experiment_detail(exp_id):
    """
    Menampilkan halaman detail lengkap satu eksperimen beserta semua metriknya.

    Memuat dan menampilkan: parameter eksperimen, metrik evaluasi keseluruhan,
    metrik per kelas, confusion matrix, dan data grafik akurasi vs. max_depth.

    Args (URL):
        exp_id (int): ID eksperimen yang akan ditampilkan.

    Template: templates/experiment_detail.html

    Context yang dikirim ke template:
        exp        (dict)      : Record eksperimen lengkap.
        details    (list[dict]): Metrik per kelas (precision, recall, F1).
        depth_results (list[dict]): Hasil sweep depth untuk grafik.
        ds         (dict|None) : Metadata dataset yang digunakan.
    """
    from database import (get_experiment_by_id, get_experiment_details,
                          get_depth_results, get_dataset_by_id)
    exp = get_experiment_by_id(exp_id)
    if not exp:
        flash('Eksperimen tidak ditemukan.', 'danger')
        return redirect(url_for('experiments_list'))

    details   = get_experiment_details(exp_id)
    depth_res = get_depth_results(exp_id)
    ds = get_dataset_by_id(exp['dataset_id']) if exp.get('dataset_id') else None

    return render_template('experiment_detail.html',
                           exp=exp, details=details,
                           depth_results=depth_res, ds=ds)


@app.route('/admin/experiments/<int:exp_id>/set-best', methods=['POST'])
@login_required
def set_best_route(exp_id):
    """
    Menetapkan eksperimen tertentu sebagai model terbaik yang aktif digunakan.

    Setelah route ini dipanggil, model dari eksperimen ini akan digunakan
    untuk semua prediksi publik di halaman /classify dan API /api/classify.
    Hanya satu eksperimen yang dapat berstatus 'best' pada satu waktu
    (semua yang lain di-reset oleh set_best_experiment()).

    Args (URL):
        exp_id (int): ID eksperimen yang akan dijadikan model terbaik.
    """
    from database import set_best_experiment, get_experiment_by_id
    set_best_experiment(exp_id)
    flash('Model terbaik berhasil ditentukan!', 'success')
    return redirect(url_for('experiment_detail', exp_id=exp_id))


@app.route('/admin/statistics')
@login_required
def statistics():
    """
    Menampilkan halaman statistik komprehensif sistem.

    Menampilkan perbandingan metrik antar eksperimen, distribusi label
    prediksi, tren akurasi, dan prediksi terbaru dari pengguna.

    Template: templates/statistics.html

    Context yang dikirim ke template:
        stats        (dict)      : Statistik ringkasan dari get_dashboard_stats().
        experiments  (list[dict]): Semua eksperimen selesai untuk perbandingan.
        recent_preds (list[dict]): 50 prediksi terbaru.
        pred_stats   (dict)      : Distribusi label prediksi {'total', 'positif', 'negatif'}.
    """
    from database import (get_dashboard_stats, get_all_done_experiments,
                          get_recent_predictions, get_prediction_stats)
    stats        = get_dashboard_stats()
    experiments  = get_all_done_experiments()
    recent_preds = get_recent_predictions(50)
    pred_stats   = get_prediction_stats()

    return render_template('statistics.html', stats=stats,
                           experiments=experiments,
                           recent_preds=recent_preds,
                           pred_stats=pred_stats)


@app.route('/admin/predictions')
@login_required
def predictions_list():
    """
    Menampilkan halaman riwayat prediksi yang dilakukan pengguna.

    Menampilkan 100 prediksi terbaru beserta statistik distribusi
    label (berapa banyak teks diprediksi positif vs. negatif).

    Template: templates/predictions.html

    Context yang dikirim ke template:
        predictions (list[dict]): 100 prediksi terbaru.
        stats       (dict)      : Distribusi label {'total', 'positif', 'negatif'}.
    """
    from database import get_recent_predictions, get_prediction_stats
    preds = get_recent_predictions(100)
    stats = get_prediction_stats()
    return render_template('predictions.html', predictions=preds, stats=stats)


# ============================================================
# API ENDPOINTS
# Endpoint yang mengembalikan data dalam format JSON.
# Digunakan oleh JavaScript frontend dan client eksternal.
# ============================================================

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """
    REST API endpoint untuk klasifikasi teks (JSON request/response).

    Memungkinkan integrasi sistem klasifikasi dengan aplikasi eksternal
    atau pengujian tanpa antarmuka web. Tidak memerlukan autentikasi.

    Request body (JSON):
        {
            "text": "teks komentar yang akan diklasifikasikan"
        }

    Response sukses (200):
        {
            "label"      : "positif" | "negatif",
            "confidence" : float (0.0-1.0),
            "preprocessed": "teks setelah preprocessing"
        }

    Response error:
        400: {"error": "Teks kosong"} — jika field text kosong.
        400: {"error": "..."} — jika prediksi gagal (misal teks kosong setelah preprocessing).
        500: {"error": "..."} — jika terjadi exception internal.
        503: {"error": "Model belum tersedia"} — jika belum ada model aktif.

    Catatan:
        Setiap prediksi berhasil otomatis disimpan ke tabel 'predictions' di database.
    """
    from database import get_best_experiment, insert_prediction
    from pipeline import predict_single

    data = request.get_json()
    text = data.get('text', '').strip() if data else ''

    if not text:
        return jsonify({'error': 'Teks kosong'}), 400

    best = get_best_experiment()
    if not best:
        return jsonify({'error': 'Model belum tersedia'}), 503

    try:
        pred = predict_single(text, best['id'])
        if 'error' in pred:
            return jsonify(pred), 400

        # Simpan hasil prediksi ke database untuk analisis
        insert_prediction(
            text,
            pred.get('preprocessed', ''),
            pred['label'],
            pred['confidence'],
            best['id']
        )
        return jsonify(pred)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments/<int:exp_id>/depth-results')
def api_depth_results(exp_id):
    """
    API endpoint untuk mengambil data hasil sweep max_depth (JSON).

    Digunakan oleh JavaScript di halaman detail eksperimen untuk merender
    grafik "Akurasi vs. Max Depth" dan "Waktu Eksekusi vs. Max Depth"
    menggunakan library chart (Chart.js atau sejenisnya).

    Tidak memerlukan autentikasi agar grafik dapat diakses dari berbagai konteks.

    Args (URL):
        exp_id (int): ID eksperimen yang data sweep depth-nya ingin diambil.

    Returns:
        JSON: List of dicts, masing-masing berisi max_depth, accuracy,
              precision, recall, f1, execution_time. List kosong [] jika
              sweep belum pernah dijalankan untuk eksperimen ini.
    """
    from database import get_depth_results
    data = get_depth_results(exp_id)
    return jsonify(data)


@app.route('/api/experiments/<int:exp_id>/model-detail')
@login_required
def api_model_detail(exp_id):
    """
    API endpoint untuk mengambil informasi detail dari file model pickle (JSON).

    Membaca file model_{exp_id}.pkl dari disk dan mengekstrak informasi
    yang berguna untuk ditampilkan ke admin, termasuk top-20 fitur kata
    dengan Information Gain tertinggi (feature importance).

    Membutuhkan autentikasi admin karena mengekspos detail internal model.

    Args (URL):
        exp_id (int): ID eksperimen yang detail modelnya ingin diambil.

    Returns:
        JSON sukses (200):
            {
                "max_depth"   : int    — Nilai max_depth yang digunakan model.
                "n_features"  : int    — Jumlah fitur input model.
                "vocab_size"  : int    — Ukuran vocabulary TF-IDF vectorizer.
                "class_labels": list   — Nama kelas ['negatif', 'positif'].
                "top_features": list   — Top-20 kata dengan importance tertinggi:
                    [{"rank": 1, "word": "leceh", "importance": 0.123}, ...]
            }
        JSON error (404): {"error": "File model tidak ditemukan"}
        JSON error (500): {"error": "pesan exception"}
    """
    from pipeline import load_model
    import numpy as np

    vectorizer, model, le = load_model(exp_id)
    if model is None:
        return jsonify({'error': 'File model tidak ditemukan'}), 404

    try:
        feature_names  = vectorizer.get_feature_names_out().tolist()
        vocab_size     = len(vectorizer.vocabulary_)
        class_labels   = le.classes_.tolist()
        n_features     = int(model.n_features) if model.n_features else len(feature_names)
        max_depth_used = int(model.max_depth)

        # Hitung top-N fitur berdasarkan feature_importances_ yang diakumulasi saat training
        importances = model.feature_importances_
        if importances is not None and len(importances) > 0:
            top_n = 20  # Tampilkan 20 kata terpenting
            # Urutkan descending berdasarkan nilai importance, ambil top_n indeks
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_features = [
                {
                    'rank':       int(i + 1),
                    'word':       feature_names[idx],
                    'importance': float(importances[idx])
                }
                for i, idx in enumerate(top_indices)
                if importances[idx] > 0  # Hanya tampilkan fitur yang berkontribusi
            ]
        else:
            top_features = []

        return jsonify({
            'max_depth':    max_depth_used,
            'n_features':   n_features,
            'vocab_size':   vocab_size,
            'class_labels': class_labels,
            'top_features': top_features,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """
    API endpoint publik untuk mengambil statistik ringkasan sistem (JSON).

    Mengembalikan data yang sama dengan yang digunakan di halaman dashboard,
    berguna untuk integrasi dengan sistem monitoring eksternal atau widget
    real-time di halaman publik.

    Tidak memerlukan autentikasi karena hanya mengekspos data statistik agregat
    yang tidak sensitif.

    Returns:
        JSON: Dictionary statistik dari get_dashboard_stats():
            {
                "total_datasets"    : int,
                "total_experiments" : int,
                "total_predictions" : int,
                "best_accuracy"     : float | null,
                "best_experiment"   : dict | null,
                "pred_positif"      : int,
                "pred_negatif"      : int
            }
    """
    from database import get_dashboard_stats
    return jsonify(get_dashboard_stats())


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    # Jalankan development server Flask pada port 5000 dengan mode debug aktif.
    # Mode debug menampilkan traceback di browser dan auto-reload saat kode berubah.
    # PENTING: Jangan gunakan debug=True di lingkungan production!
    app.run(debug=True, port=5000)