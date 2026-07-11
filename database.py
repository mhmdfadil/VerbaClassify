"""
database.py
===========
Interaksi dengan Supabase untuk semua operasi CRUD.
"""
import os
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_supabase: Client = None


def get_db() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        _supabase = create_client(url, key)
    return _supabase


# ============ DATASETS ============

def insert_dataset(filename, total, positif, negatif, notes=''):
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
    db = get_db()
    res = db.table('datasets').select('*').order('uploaded_at', desc=True).execute()
    return res.data or []


def get_dataset_by_id(dataset_id):
    db = get_db()
    res = db.table('datasets').select('*').eq('id', dataset_id).execute()
    return res.data[0] if res.data else None


def delete_dataset(dataset_id):
    db = get_db()
    db.table('datasets').delete().eq('id', dataset_id).execute()


# ============ RAW DATA ============

def insert_raw_data_batch(records: list):
    """Insert banyak record sekaligus (batch 500)."""
    db = get_db()
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        db.table('raw_data').insert(batch).execute()


def get_raw_data_by_dataset(dataset_id, limit=100, offset=0):
    db = get_db()
    res = (db.table('raw_data')
           .select('*')
           .eq('dataset_id', dataset_id)
           .range(offset, offset + limit - 1)
           .execute())
    return res.data or []


def get_all_raw_texts_labels(dataset_id):
    """Ambil semua teks dan label untuk training (streaming besar)."""
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
    db = get_db()
    db.table('raw_data').update({'preprocessed_text': preprocessed}).eq('id', row_id).execute()


# ============ EXPERIMENTS ============

def insert_experiment(dataset_id, proportion_label, train_ratio, test_ratio, max_depth, use_smote):
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
    db = get_db()
    db.table('experiments').update({
        'accuracy':        result['accuracy'],
        'precision_score': result['precision'],
        'recall_score':    result['recall'],
        'f1_score':        result['f1'],
        'macro_precision':    result.get('macro_precision'),
        'macro_recall':       result.get('macro_recall'),
        'macro_f1':           result.get('macro_f1'),
        'weighted_precision': result.get('weighted_precision'),
        'weighted_recall':    result.get('weighted_recall'),
        'weighted_f1':        result.get('weighted_f1'),
        'support':         result.get('support'),
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
    db = get_db()
    db.table('experiments').update({
        'status': 'error',
        'error_msg': str(msg)[:500],
        'finished_at': datetime.now(timezone.utc).isoformat()
    }).eq('id', exp_id).execute()


def update_experiment_running(exp_id):
    db = get_db()
    db.table('experiments').update({'status': 'running'}).eq('id', exp_id).execute()


def get_experiments_by_dataset(dataset_id):
    db = get_db()
    res = (db.table('experiments')
           .select('*')
           .eq('dataset_id', dataset_id)
           .order('created_at', desc=True)
           .execute())
    return res.data or []


def get_experiment_by_id(exp_id):
    db = get_db()
    res = db.table('experiments').select('*').eq('id', exp_id).execute()
    return res.data[0] if res.data else None


def get_best_experiment():
    """Ambil eksperimen terbaik yang sudah ditandai."""
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
    """Tandai satu eksperimen sebagai terbaik, reset yang lain."""
    db = get_db()
    db.table('experiments').update({'is_best': False}).neq('id', exp_id).execute()
    db.table('experiments').update({'is_best': True}).eq('id', exp_id).execute()


def get_all_done_experiments():
    db = get_db()
    res = (db.table('experiments')
           .select('*')
           .eq('status', 'done')
           .order('accuracy', desc=True)
           .execute())
    return res.data or []


# ============ EXPERIMENT DETAILS ============

def insert_experiment_details(exp_id, report: dict, class_names: list):
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
    db = get_db()
    res = db.table('experiment_details').select('*').eq('experiment_id', exp_id).execute()
    return res.data or []


# ============ DEPTH RESULTS ============

def insert_depth_results(exp_id, depth_results: list):
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
    db = get_db()
    res = (db.table('depth_results')
           .select('*')
           .eq('experiment_id', exp_id)
           .order('max_depth')
           .execute())
    return res.data or []


# ============ PREDICTIONS ============

def insert_prediction(input_text, preprocessed, label, confidence, exp_id):
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
    db = get_db()
    res = (db.table('predictions')
           .select('*')
           .order('created_at', desc=True)
           .limit(limit)
           .execute())
    return res.data or []


def get_prediction_stats():
    db = get_db()
    res = db.table('predictions').select('predicted_label').execute()
    data = res.data or []
    total = len(data)
    positif = sum(1 for d in data if d['predicted_label'] == 'positif')
    negatif = total - positif
    return {'total': total, 'positif': positif, 'negatif': negatif}


# ============ STATISTICS ============

def get_dashboard_stats():
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