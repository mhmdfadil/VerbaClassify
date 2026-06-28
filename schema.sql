-- ============================================================
-- SKRIPSI: Klasifikasi Respon Pelecehan Seksual Verbal
-- Platform: Supabase (PostgreSQL)
-- Author: Rizka Mardiah Putri Buyung Lubis
-- NIM: 220170183
-- ============================================================

-- 1. Table: datasets (uploaded CSV datasets)
CREATE TABLE IF NOT EXISTS datasets (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    total_data INTEGER NOT NULL,
    positif_count INTEGER NOT NULL,
    negatif_count INTEGER NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

-- 2. Table: raw_data (individual tweet records per dataset)
CREATE TABLE IF NOT EXISTS raw_data (
    id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT REFERENCES datasets(id) ON DELETE CASCADE,
    created_at_tweet TEXT,
    full_text TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('positif', 'negatif')),
    preprocessed_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Table: experiments (each training run / proportion experiment)
CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT REFERENCES datasets(id) ON DELETE CASCADE,
    proportion_label TEXT NOT NULL,  -- e.g. "80:20", "70:30"
    train_ratio FLOAT NOT NULL,
    test_ratio FLOAT NOT NULL,
    train_count INTEGER,
    test_count INTEGER,
    max_depth INTEGER DEFAULT 3,
    use_smote BOOLEAN DEFAULT TRUE,
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    tp INTEGER, tn INTEGER, fp INTEGER, fn INTEGER,
    execution_time FLOAT,          -- waktu training + prediksi (detik)
    train_time FLOAT,              -- waktu training saja
    predict_time FLOAT,            -- waktu prediksi saja
    is_best BOOLEAN DEFAULT FALSE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending','running','done','error')),
    error_msg TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- 4. Table: experiment_details (per-class metrics)
CREATE TABLE IF NOT EXISTS experiment_details (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES experiments(id) ON DELETE CASCADE,
    class_label TEXT NOT NULL,
    precision_val FLOAT,
    recall_val FLOAT,
    f1_val FLOAT,
    support INTEGER
);

-- 5. Table: depth_results (accuracy per max_depth for a given experiment's proportion)
CREATE TABLE IF NOT EXISTS depth_results (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES experiments(id) ON DELETE CASCADE,
    max_depth INTEGER NOT NULL,
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    execution_time FLOAT           -- waktu eksekusi untuk depth ini (detik)
);

-- 6. Table: predictions (user classification requests, no auth)
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,
    preprocessed_text TEXT,
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('positif', 'negatif')),
    confidence FLOAT,
    experiment_id BIGINT REFERENCES experiments(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Table: admin_sessions (simple session tracking)
CREATE TABLE IF NOT EXISTS admin_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_token TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_raw_data_dataset ON raw_data(dataset_id);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_id);
CREATE INDEX IF NOT EXISTS idx_experiments_best ON experiments(is_best);
CREATE INDEX IF NOT EXISTS idx_depth_results_exp ON depth_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);

-- ============================================================
-- ENABLE ROW LEVEL SECURITY (optional, for production)
-- ============================================================
-- ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE raw_data ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE experiments ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- MIGRATION: Tambah kolom execution_time (jalankan jika tabel
-- sudah ada dan belum punya kolom ini)
-- ============================================================
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS execution_time FLOAT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS train_time FLOAT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS predict_time FLOAT;
ALTER TABLE depth_results ADD COLUMN IF NOT EXISTS execution_time FLOAT;
