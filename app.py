"""
app.py — Flask Web Application
================================
Sistem Informasi Klasifikasi Respon Pelecehan Seksual Verbal
Platform X menggunakan ID3 Modifikasi

Author: Rizka Mardiah Putri Buyung Lubis (220170183)
Universitas Malikussaleh, 2026
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

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rizka_skripsi_2026')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'rizka2026')

# Task status tracking (in-memory)
_task_status = {}

# ============================================================
# AUTH DECORATOR
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# PUBLIC ROUTES
# ============================================================

@app.route('/')
def index():
    from database import get_best_experiment, get_prediction_stats
    best = get_best_experiment()
    stats = get_prediction_stats()
    return render_template('index.html', best=best, stats=stats)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Halaman klasifikasi untuk pengguna umum (tanpa auth)."""
    from database import get_best_experiment, insert_prediction
    from pipeline import predict_single

    result = None
    error = None
    input_text = ''

    best = get_best_experiment()
    if not best:
        error = 'Model belum tersedia. Hubungi administrator.'

    if request.method == 'POST' and best:
        input_text = request.form.get('text', '').strip()
        if not input_text:
            error = 'Teks tidak boleh kosong.'
        elif len(input_text) < 5:
            error = 'Teks terlalu pendek (min. 5 karakter).'
        else:
            try:
                pred = predict_single(input_text, best['id'])
                if 'error' in pred:
                    error = pred['error']
                else:
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
    return render_template('about.html')


# ============================================================
# AUTH ROUTES
# ============================================================

@app.route('/admin/login', methods=['GET', 'POST'])
def login():
    if session.get('admin_logged_in'):
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            flash('Login berhasil! Selamat datang, Admin.', 'success')
            return redirect(url_for('dashboard'))
        else:
            error = 'Username atau password salah.'
    return render_template('login.html', error=error)


@app.route('/admin/logout')
def logout():
    session.clear()
    flash('Berhasil logout.', 'info')
    return redirect(url_for('index'))


# ============================================================
# ADMIN ROUTES
# ============================================================

@app.route('/admin')
@login_required
def dashboard():
    from database import get_dashboard_stats, get_all_datasets, get_all_done_experiments
    stats = get_dashboard_stats()
    datasets = get_all_datasets()
    experiments = get_all_done_experiments()
    return render_template('dashboard.html', stats=stats,
                           datasets=datasets, experiments=experiments)


@app.route('/admin/datasets')
@login_required
def datasets():
    from database import get_all_datasets
    dsets = get_all_datasets()
    return render_template('datasets.html', datasets=dsets)


@app.route('/admin/datasets/upload', methods=['GET', 'POST'])
@login_required
def upload_dataset():
    if request.method == 'POST':
        file = request.files.get('file')
        notes = request.form.get('notes', '')

        if not file or file.filename == '':
            flash('Pilih file CSV terlebih dahulu.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        if not filename.endswith('.csv'):
            flash('File harus berformat CSV.', 'danger')
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Coba beberapa encoding umum
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

            # Normalisasi nama kolom
            df.columns = [c.strip().replace('\ufeff', '').lower() for c in df.columns]

            if 'full_text' not in df.columns or 'label' not in df.columns:
                flash(f'CSV harus memiliki kolom "full_text" dan "label". Kolom ditemukan: {list(df.columns)}', 'danger')
                return redirect(request.url)

            df = df[['full_text', 'label']].dropna()
            df['full_text'] = df['full_text'].astype(str).str.strip()
            df['label']     = df['label'].astype(str).str.strip().str.lower()
            df = df[df['label'].isin(['positif', 'negatif'])]
            df = df[df['full_text'].str.len() > 2]

            total   = len(df)
            positif = int((df['label'] == 'positif').sum())
            negatif = int((df['label'] == 'negatif').sum())

            if total == 0:
                flash('Dataset kosong atau format tidak sesuai.', 'danger')
                return redirect(request.url)

            # ── Preprocessing semua teks ─────────────────────────
            from preprocessing import preprocess_batch
            texts = df['full_text'].tolist()
            print(f'[Upload] Preprocessing {total} teks...')
            preprocessed_texts = preprocess_batch(texts)
            print(f'[Upload] Preprocessing selesai.')
            # ─────────────────────────────────────────────────────

            from database import insert_dataset, insert_raw_data_batch
            ds = insert_dataset(filename, total, positif, negatif, notes)
            dataset_id = ds['id']

            # Insert raw data DENGAN preprocessed_text terisi
            records = []
            for i, (_, row) in enumerate(df.iterrows()):
                prep = preprocessed_texts[i] if i < len(preprocessed_texts) else ''
                records.append({
                    'dataset_id':        dataset_id,
                    'full_text':         str(row['full_text'])[:2000],
                    'label':             row['label'],
                    'preprocessed_text': prep[:1000] if prep else '',
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
    from database import get_dataset_by_id, get_raw_data_by_dataset, get_experiments_by_dataset
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash('Dataset tidak ditemukan.', 'danger')
        return redirect(url_for('datasets'))
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
    from database import delete_dataset
    delete_dataset(dataset_id)
    flash('Dataset berhasil dihapus.', 'success')
    return redirect(url_for('datasets'))


@app.route('/admin/experiments')
@login_required
def experiments_list():
    from database import get_all_done_experiments, get_best_experiment
    exps = get_all_done_experiments()
    best = get_best_experiment()
    return render_template('experiments.html', experiments=exps, best=best)


@app.route('/admin/experiments/run', methods=['GET', 'POST'])
@login_required
def run_experiment_route():
    from database import get_all_datasets
    datasets_list = get_all_datasets()

    if request.method == 'POST':
        dataset_id = int(request.form.get('dataset_id'))
        proportions = request.form.getlist('proportions')  # e.g. ['80:20','70:30']
        max_depth = int(request.form.get('max_depth', 3))
        use_smote = request.form.get('use_smote') == 'on'
        run_depth_sweep = request.form.get('run_depth_sweep') == 'on'

        if not proportions:
            flash('Pilih minimal satu proporsi data.', 'danger')
            return redirect(request.url)

        # Jalankan di background thread
        task_id = f"task_{dataset_id}_{int(datetime.now().timestamp())}"
        _task_status[task_id] = {'status': 'running', 'progress': 0, 'message': 'Memulai...'}

        def bg_task():
            try:
                from database import (get_all_raw_texts_labels, insert_experiment,
                                      update_experiment_result, update_experiment_error,
                                      update_experiment_running, insert_experiment_details,
                                      insert_depth_results)
                from pipeline import run_experiment, run_depth_sweep as rds, save_model

                _task_status[task_id]['message'] = 'Memuat data dari database...'
                rows = get_all_raw_texts_labels(dataset_id)
                texts  = [r['full_text'] for r in rows]
                labels = [r['label'] for r in rows]

                # Gunakan preprocessed_text dari DB jika sudah ada (hasil upload)
                preprocessed_from_db = [r.get('preprocessed_text') or '' for r in rows]
                has_preprocessed = sum(1 for t in preprocessed_from_db if t and t.strip()) > 0
                if has_preprocessed:
                    _task_status[task_id]['message'] = f'Ditemukan {sum(1 for t in preprocessed_from_db if t)} teks sudah dipreproses, digunakan langsung...'
                    prep_texts = preprocessed_from_db
                else:
                    _task_status[task_id]['message'] = 'Preprocessing teks...'
                    prep_texts = None  # pipeline akan preprocess sendiri

                total_tasks = len(proportions)
                for idx, prop in enumerate(proportions):
                    _task_status[task_id]['message'] = f'Training proporsi {prop} ({idx+1}/{total_tasks})...'
                    _task_status[task_id]['progress'] = int((idx / total_tasks) * 80)

                    parts = prop.split(':')
                    train_r = int(parts[0]) / 100.0

                    # Insert experiment record
                    exp = insert_experiment(dataset_id, prop, train_r, 1-train_r, max_depth, use_smote)
                    exp_id = exp['id']
                    update_experiment_running(exp_id)

                    try:
                        result = run_experiment(
                            texts, labels, train_r, max_depth, use_smote,
                            preprocessed_texts=prep_texts
                        )
                        update_experiment_result(exp_id, result)
                        insert_experiment_details(exp_id, result['report'], result['class_names'])
                        save_model(exp_id, result['vectorizer'], result['model'], result['le'])

                        if run_depth_sweep:
                            _task_status[task_id]['message'] = f'Sweep max_depth untuk {prop}...'
                            depth_res = rds(
                                texts, labels, train_r, use_smote=use_smote,
                                preprocessed_texts=prep_texts
                            )
                            insert_depth_results(exp_id, depth_res)

                    except Exception as e:
                        update_experiment_error(exp_id, str(e))
                        _task_status[task_id]['message'] = f'Error pada {prop}: {str(e)}'

                _task_status[task_id]['status'] = 'done'
                _task_status[task_id]['progress'] = 100
                _task_status[task_id]['message'] = 'Selesai!'

            except Exception as e:
                _task_status[task_id]['status'] = 'error'
                _task_status[task_id]['message'] = f'Error: {str(e)}'
                _task_status[task_id]['traceback'] = traceback.format_exc()

        t = threading.Thread(target=bg_task)
        t.daemon = True
        t.start()

        flash(f'Eksperimen dimulai! Task ID: {task_id}', 'info')
        return redirect(url_for('experiment_progress', task_id=task_id))

    return render_template('run_experiment.html', datasets=datasets_list)


@app.route('/admin/experiments/progress/<task_id>')
@login_required
def experiment_progress(task_id):
    return render_template('experiment_progress.html', task_id=task_id)


@app.route('/api/task-status/<task_id>')
@login_required
def task_status_api(task_id):
    status = _task_status.get(task_id, {'status': 'not_found'})
    return jsonify(status)


@app.route('/admin/experiments/<int:exp_id>')
@login_required
def experiment_detail(exp_id):
    from database import (get_experiment_by_id, get_experiment_details,
                          get_depth_results, get_dataset_by_id)
    exp = get_experiment_by_id(exp_id)
    if not exp:
        flash('Eksperimen tidak ditemukan.', 'danger')
        return redirect(url_for('experiments_list'))
    details = get_experiment_details(exp_id)
    depth_res = get_depth_results(exp_id)
    ds = get_dataset_by_id(exp['dataset_id']) if exp.get('dataset_id') else None
    return render_template('experiment_detail.html',
                           exp=exp, details=details,
                           depth_results=depth_res, ds=ds)


@app.route('/admin/experiments/<int:exp_id>/set-best', methods=['POST'])
@login_required
def set_best_route(exp_id):
    from database import set_best_experiment, get_experiment_by_id
    set_best_experiment(exp_id)
    flash('Model terbaik berhasil ditentukan!', 'success')
    return redirect(url_for('experiment_detail', exp_id=exp_id))


@app.route('/admin/statistics')
@login_required
def statistics():
    from database import (get_dashboard_stats, get_all_done_experiments,
                          get_recent_predictions, get_prediction_stats,
                          get_experiment_details)
    stats = get_dashboard_stats()
    experiments = get_all_done_experiments()
    recent_preds = get_recent_predictions(50)
    pred_stats = get_prediction_stats()

    # Bangun data precision/recall/f1 per kelas (positif & negatif) untuk tiap eksperimen,
    # diambil dari experiment_details (bukan experiments, karena metrik ini per-kelas).
    exp_metrics = []
    for exp in experiments:
        details = get_experiment_details(exp['id'])
        pos = next((d for d in details if d.get('class_label') == 'positif'), {})
        neg = next((d for d in details if d.get('class_label') == 'negatif'), {})
        exp_metrics.append({
            'label': exp.get('proportion_label'),
            'precision_positif': pos.get('precision_val'),
            'precision_negatif': neg.get('precision_val'),
            'recall_positif': pos.get('recall_val'),
            'recall_negatif': neg.get('recall_val'),
            'f1_positif': pos.get('f1_val'),
            'f1_negatif': neg.get('f1_val'),
        })

    return render_template('statistics.html', stats=stats,
                           experiments=experiments,
                           exp_metrics=exp_metrics,
                           recent_preds=recent_preds,
                           pred_stats=pred_stats)


@app.route('/admin/predictions')
@login_required
def predictions_list():
    from database import get_recent_predictions, get_prediction_stats
    preds = get_recent_predictions(100)
    stats = get_prediction_stats()
    return render_template('predictions.html', predictions=preds, stats=stats)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/api/classify', methods=['POST'])
def api_classify():
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
        insert_prediction(text, pred.get('preprocessed', ''),
                          pred['label'], pred['confidence'], best['id'])
        return jsonify(pred)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments/<int:exp_id>/depth-results')
def api_depth_results(exp_id):
    from database import get_depth_results
    data = get_depth_results(exp_id)
    return jsonify(data)


@app.route('/api/experiments/<int:exp_id>/model-detail')
@login_required
def api_model_detail(exp_id):
    """Baca detail informatif dari file model_{exp_id}.pkl"""
    from pipeline import load_model
    import numpy as np

    vectorizer, model, le = load_model(exp_id)
    if model is None:
        return jsonify({'error': 'File model tidak ditemukan'}), 404

    try:
        feature_names = vectorizer.get_feature_names_out().tolist()
        vocab_size = len(vectorizer.vocabulary_)
        class_labels = le.classes_.tolist()
        n_features = int(model.n_features) if model.n_features else len(feature_names)
        max_depth_used = int(model.max_depth)

        # Top-N feature importance
        importances = model.feature_importances_
        if importances is not None and len(importances) > 0:
            top_n = 20
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_features = [
                {
                    'rank': int(i + 1),
                    'word': feature_names[idx],
                    'importance': float(importances[idx])
                }
                for i, idx in enumerate(top_indices)
                if importances[idx] > 0
            ]
        else:
            top_features = []

        return jsonify({
            'max_depth': max_depth_used,
            'n_features': n_features,
            'vocab_size': vocab_size,
            'class_labels': class_labels,
            'top_features': top_features,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    from database import get_dashboard_stats
    return jsonify(get_dashboard_stats())


if __name__ == '__main__':
    app.run(debug=True, port=5000)
