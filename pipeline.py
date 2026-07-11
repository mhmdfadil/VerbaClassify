"""
pipeline.py
===========
Pipeline lengkap: Preprocessing → TF-IDF → Split → SMOTE → ID3 Modifikasi → Evaluasi
Sesuai alur sistem proposal Rizka Mardiah Putri Buyung Lubis (2026)
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

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


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
    Jalankan satu eksperimen klasifikasi lengkap.
    Jika preprocessed_texts sudah tersedia (dari DB), langsung digunakan —
    tidak perlu preprocessing ulang.

    Returns dict dengan semua metrics dan hasil.
    """
    # --- Preprocessing ---
    if preprocessed_texts and len(preprocessed_texts) == len(texts):
        print(f"  [1] Menggunakan preprocessed_text dari database ({len(texts)} teks)...")
        preprocessed = preprocessed_texts
    else:
        print(f"  [1] Preprocessing {len(texts)} teks...")
        preprocessed = preprocess_batch(texts)

    # --- Label Encoding ---
    le = LabelEncoder()
    y = le.fit_transform(labels)  # negatif=0, positif=1
    # Pastikan: 'negatif'=0, 'positif'=1
    class_names = list(le.classes_)

    # --- TF-IDF Feature Extraction ---
    print(f"  [2] TF-IDF extraction (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(preprocessed)

    # --- Split Data ---
    test_ratio = round(1.0 - train_ratio, 2)
    print(f"  [3] Split data {int(train_ratio*100)}:{int(test_ratio*100)}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=42,
        stratify=y
    )

    # --- SMOTE ---
    if use_smote:
        print(f"  [4] SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y_train).min() - 1))
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"  SMOTE failed ({e}), skip SMOTE")
            X_train_sm, y_train_sm = X_train, y_train
    else:
        X_train_sm, y_train_sm = X_train, y_train

    # --- ID3 Modifikasi + ukur waktu eksekusi ---
    print(f"  [5] Training ID3 Modifikasi (max_depth={max_depth})...")
    t_start = time.perf_counter()
    model = ID3ModifiedClassifier(max_depth=max_depth, min_samples_split=2)
    model.fit(X_train_sm, y_train_sm)
    t_train = time.perf_counter() - t_start

    # --- Prediksi & Evaluasi ---
    print(f"  [6] Evaluasi model...")
    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_pred = time.perf_counter() - t_pred_start
    execution_time = round(t_train + t_pred, 6)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)

    # Macro & weighted averages (langsung dari classification_report)
    macro = report.get('macro avg', {})
    weighted = report.get('weighted avg', {})

    # Total support = jumlah support dari seluruh class_name pada eksperimen ini
    total_support = sum(int(report[c]['support']) for c in class_names if c in report)

    # TP, TN, FP, FN (binary)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"  ✓ Akurasi: {acc:.4f} | Presisi: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Waktu: {execution_time:.4f}s")

    return {
        'preprocessed': preprocessed,
        'vectorizer': vectorizer,
        'model': model,
        'le': le,
        'class_names': class_names,
        'X_train': X_train_sm,
        'X_test': X_test,
        'y_train': y_train_sm,
        'y_test': y_test,
        'y_pred': y_pred,
        'train_count': X_train_sm.shape[0],
        'test_count': X_test.shape[0],
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'macro_precision': float(macro.get('precision', 0)),
        'macro_recall': float(macro.get('recall', 0)),
        'macro_f1': float(macro.get('f1-score', 0)),
        'weighted_precision': float(weighted.get('precision', 0)),
        'weighted_recall': float(weighted.get('recall', 0)),
        'weighted_f1': float(weighted.get('f1-score', 0)),
        'support': total_support,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'confusion_matrix': cm.tolist(),
        'report': report,
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'max_depth': max_depth,
        'use_smote': use_smote,
        'execution_time': execution_time,
        'train_time': round(t_train, 6),
        'predict_time': round(t_pred, 6),
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
    Menjalankan eksperimen untuk berbagai nilai max_depth (1-20).
    Returns list of dicts {max_depth, accuracy, precision, recall, f1}
    """
    if depths is None:
        depths = list(range(1, 21))

    if preprocessed_texts and len(preprocessed_texts) == len(texts):
        print(f"  [Sweep] Menggunakan preprocessed_text dari database...")
        preprocessed = preprocessed_texts
    else:
        preprocessed = preprocess_batch(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(preprocessed)

    test_ratio = round(1.0 - train_ratio, 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    if use_smote:
        smote = SMOTE(random_state=42,
                      k_neighbors=min(5, np.bincount(y_train).min() - 1))
        try:
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        except Exception:
            X_train_sm, y_train_sm = X_train, y_train
    else:
        X_train_sm, y_train_sm = X_train, y_train

    results = []
    for d in depths:
        t0 = time.perf_counter()
        model = ID3ModifiedClassifier(max_depth=d, min_samples_split=2)
        model.fit(X_train_sm, y_train_sm)
        t_train_d = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        t_pred_d = time.perf_counter() - t1

        exec_time = round(t_train_d + t_pred_d, 6)
        acc_val   = float(accuracy_score(y_test, y_pred))

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


def save_model(experiment_id: int, vectorizer, model, le) -> str:
    """Simpan model ke disk."""
    path = os.path.join(MODEL_DIR, f'model_{experiment_id}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': model, 'le': le}, f)
    return path


def load_model(experiment_id: int):
    """Load model dari disk."""
    path = os.path.join(MODEL_DIR, f'model_{experiment_id}.pkl')
    if not os.path.exists(path):
        return None, None, None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['vectorizer'], data['model'], data['le']


def predict_single(text: str, experiment_id: int) -> dict:
    """
    Prediksi single text menggunakan model terbaik.
    Returns dict {label, confidence, preprocessed}
    """
    vectorizer, model, le = load_model(experiment_id)
    if model is None:
        return {'error': 'Model tidak ditemukan'}

    preprocessed = preprocess_batch([text])[0]
    if not preprocessed.strip():
        return {'error': 'Teks kosong setelah preprocessing'}

    X = vectorizer.transform([preprocessed])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(proba[pred])
    label = le.inverse_transform([pred])[0]

    return {
        'label': label,
        'confidence': confidence,
        'preprocessed': preprocessed,
        'raw_pred': int(pred)
    }