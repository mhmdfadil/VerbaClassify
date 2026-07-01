"""
id3_modified.py
===============
Implementasi algoritma ID3 Modifikasi berdasarkan jurnal ilmiah:

    Asrianda, A., Mawengkang, H., Sihombing, P., Nasution, M. K. M. (2025).
    "Optimization of Marketing Campaigns Using a Modified ID3 Decision Tree
    Algorithm." Eastern-European Journal of Enterprise Technologies, 2(13), 58–70.
    DOI: 10.15587/1729-4061.2025.327158

    Asrianda, A., et al. (2025). "Evaluating the Impact of Model Complexity on
    the Accuracy of ID3 and Modified ID3: A Case Study of the Max_Depth Parameter."
    Jurnal Teknik Informatika (JUTIF), 6(5), 3707–3718.

Inti modifikasi yang dilakukan:
    Mengganti formula Shannon Entropy standar yang menggunakan logaritma basis-2
    dengan formula entropy yang disederhanakan menggunakan logaritma natural (ln),
    lalu dinormalisasi kembali ke basis-2 dengan faktor (1/ln2). Perubahan ini
    bertujuan mengurangi bias pada atribut dengan banyak nilai kategori dan
    meningkatkan efisiensi komputasi pada data tidak seimbang (imbalanced).

Formula Entropy Modifikasi (Persamaan 4, Asrianda et al. 2025):
    Ent(D) = (1/ln2) * [ ln(p+n) - (p*ln(p) + n*ln(n)) / (p+n) ]

    di mana:
        p   = jumlah instance kelas positif (label=1)
        n   = jumlah instance kelas negatif (label=0)
        ln  = logaritma natural (basis e ≈ 2.718)

Formula Conditional Entropy (Persamaan 5, Asrianda et al. 2025):
    EntA(D) = Σ_j (|Dj|/|D|) * (1/ln2) * [ ln(|Dj|) - (pj/|Dj|)*ln(pj)
                                                       - (nj/|Dj|)*ln(nj) ]

Formula Information Gain (Persamaan 6, Asrianda et al. 2025):
    Gain(A) = Ent(D) - EntA(D)

Implementasi pada data TF-IDF:
    Karena fitur TF-IDF bersifat numerik kontinu, pemilihan split menggunakan
    strategi binary split: untuk setiap fitur, dicoba semua titik tengah
    (midpoint) antara nilai unik yang berurutan sebagai kandidat threshold.
    Threshold dengan Information Gain tertinggi dipilih sebagai titik pemisah.

Dependensi:
    - numpy  : Operasi array dan pencarian nilai unik (np.unique, np.sum, dll.)
    - math   : Fungsi logaritma natural dan konstanta (math.log, math.log(2))

Author   : Rizka Mardiah Putri Buyung Lubis (220170183)
Institusi: Universitas Malikussaleh, 2026
"""

import numpy as np
import math


# ============================================================
# FUNGSI-FUNGSI ENTROPY DAN INFORMATION GAIN
# Fungsi-fungsi di bawah ini mengimplementasikan formula
# matematika dari jurnal Asrianda et al. (2025) secara langsung.
# ============================================================

def modified_entropy(p, n):
    """
    Menghitung entropy modifikasi untuk dataset dengan p positif dan n negatif.

    Ini adalah implementasi langsung dari Formula Entropy Modifikasi
    (Persamaan 4) dalam jurnal Asrianda et al. (2025). Formula ini
    merupakan bentuk setara dari Shannon Entropy yang ditulis ulang
    menggunakan logaritma natural, lalu dinormalisasi ke basis-2
    dengan faktor (1/ln2) agar hasilnya konsisten dengan satuan 'bits'.

    Formula:
        Ent(D) = (1/ln2) * [ ln(p+n) - (p*ln(p) + n*ln(n)) / (p+n) ]

    Properti nilai entropy yang dihasilkan:
        - 0.0  : Dataset murni (semua positif ATAU semua negatif) → tidak ada ketidakpastian.
        - 1.0  : Dataset seimbang sempurna (p == n) → ketidakpastian maksimum.
        - Nilai antara 0.0–1.0 untuk kondisi di antara keduanya.

    Args:
        p (int): Jumlah instance kelas positif (label=1) dalam dataset.
        n (int): Jumlah instance kelas negatif (label=0) dalam dataset.

    Returns:
        float: Nilai entropy dalam satuan bits (rentang 0.0 hingga 1.0).
               Mengembalikan 0.0 jika total data adalah nol atau salah satu
               kelas tidak ada (dataset sudah murni).

    Contoh penggunaan:
        >>> modified_entropy(50, 50)   # Seimbang sempurna → ~1.0
        1.0
        >>> modified_entropy(100, 0)   # Murni positif → 0.0
        0.0
        >>> modified_entropy(80, 20)   # Mayoritas positif → ~0.72
        ~0.72
    """
    total = p + n
    if total == 0:
        # Tidak ada data sama sekali → entropy tidak terdefinisi, kembalikan 0
        return 0.0
    if p == 0 or n == 0:
        # Dataset sudah murni satu kelas → tidak ada ketidakpastian
        return 0.0

    ln2 = math.log(2)                          # ln(2) ≈ 0.6931, faktor normalisasi ke basis-2
    ln_total = math.log(total)                 # ln(p+n), informasi total dataset
    weighted = (p * math.log(p) + n * math.log(n)) / total  # rata-rata berbobot ln per kelas
    entropy = (1 / ln2) * (ln_total - weighted)              # formula inti modifikasi
    return max(0.0, entropy)                   # pastikan tidak negatif akibat floating point


def modified_conditional_entropy(y_parent, X_col, y):
    """
    Menghitung Conditional Entropy (Entropy Atribut) untuk fitur biner tertentu.

    Mengimplementasikan formula EntA(D) dari Persamaan 5 dalam jurnal
    Asrianda et al. (2025). Fungsi ini menghitung rata-rata entropy berbobot
    dari semua subset yang terbentuk ketika dataset dipartisi berdasarkan
    nilai unik dari atribut/fitur yang diberikan.

    Formula:
        EntA(D) = Σ_j (|Dj|/|D|) * (1/ln2) * [ ln(|Dj|) - (pj/|Dj|)*ln(pj)
                                                            - (nj/|Dj|)*ln(nj) ]
    di mana:
        j      = indeks subset (setiap nilai unik atribut A)
        |Dj|   = jumlah data dalam subset ke-j
        |D|    = jumlah total data
        pj     = jumlah data positif dalam subset ke-j
        nj     = jumlah data negatif dalam subset ke-j

    Catatan implementasi:
        Karena fungsi ini dipanggil pada konteks binary split (X_col berisi
        nilai 0 atau 1 hasil threshold), biasanya hanya ada dua nilai unik:
        0 (data ≤ threshold) dan 1 (data > threshold).

    Args:
        y_parent (array-like): Parameter ini tidak digunakan secara langsung
                               dalam perhitungan (kompatibilitas antarmuka).
                               Nilai y aktual diambil dari parameter y.
        X_col    (np.ndarray): Array 1D nilai atribut/fitur untuk setiap sampel.
                               Dalam konteks binary split, berisi nilai 0 atau 1.
        y        (np.ndarray): Array 1D label kelas aktual (0 atau 1) untuk
                               setiap sampel, sejajar dengan X_col.

    Returns:
        float: Nilai conditional entropy dalam bits. Semakin rendah nilainya,
               semakin baik atribut ini dalam memisahkan kelas.
               Mengembalikan 0.0 jika dataset kosong.
    """
    total = len(y)
    if total == 0:
        return 0.0

    unique_vals = np.unique(X_col)   # Dapatkan semua nilai unik dalam kolom fitur
    cond_entropy = 0.0               # Akumulator conditional entropy

    for val in unique_vals:
        # Buat mask untuk memilih sampel yang memiliki nilai fitur == val
        mask = X_col == val
        subset_y = y[mask]           # Label kelas untuk subset ini
        dj = len(subset_y)           # |Dj|: ukuran subset
        if dj == 0:
            continue                 # Lewati subset kosong

        pj = np.sum(subset_y == 1)  # Jumlah positif dalam subset
        nj = np.sum(subset_y == 0)  # Jumlah negatif dalam subset

        if pj == 0 or nj == 0:
            # Subset sudah murni → entropy subset = 0, tidak perlu dihitung
            subset_ent = 0.0
        else:
            # Hitung entropy subset menggunakan formula modifikasi
            ln2 = math.log(2)
            subset_ent = (1 / ln2) * (
                math.log(dj)                    # ln(|Dj|)
                - (pj / dj) * math.log(pj)     # - (pj/|Dj|) * ln(pj)
                - (nj / dj) * math.log(nj)     # - (nj/|Dj|) * ln(nj)
            )
            subset_ent = max(0.0, subset_ent)  # Koreksi negatif akibat floating point

        # Bobot subset: proporsinya terhadap total dataset
        weight = dj / total
        cond_entropy += weight * subset_ent    # Tambahkan kontribusi subset ke akumulator

    return cond_entropy


def information_gain(y, X_col):
    """
    Menghitung Information Gain dari satu fitur terhadap label kelas.

    Mengimplementasikan formula Gain(A) dari Persamaan 6 dalam jurnal
    Asrianda et al. (2025). Information Gain mengukur seberapa besar
    pengurangan entropy (ketidakpastian) yang dicapai ketika dataset
    dipartisi berdasarkan atribut/fitur tertentu.

    Formula:
        Gain(A) = Ent(D) - EntA(D)

    Interpretasi nilai:
        - Gain tinggi → Fitur sangat informatif, mampu memisahkan kelas dengan baik.
        - Gain = 0.0  → Fitur tidak memberikan informasi sama sekali.
        - Gain negatif (jarang) → Dapat terjadi akibat floating point; dianggap 0.

    Dalam ID3ModifiedClassifier, fungsi ini dipanggil untuk setiap fitur
    pada setiap node saat pembangunan pohon, dan fitur dengan Gain tertinggi
    dipilih sebagai node pemisah (splitting node).

    Args:
        y     (np.ndarray): Array 1D label kelas aktual (0 atau 1) untuk
                            seluruh dataset/subset saat ini.
        X_col (np.ndarray): Array 1D nilai fitur hasil binary split (0 atau 1)
                            yang merepresentasikan apakah nilai asli fitur
                            tersebut ≤ threshold (0) atau > threshold (1).

    Returns:
        float: Nilai Information Gain (dalam bits). Semakin tinggi nilainya,
               semakin baik fitur ini untuk digunakan sebagai splitting node.
    """
    p = np.sum(y == 1)               # Jumlah positif dalam dataset/subset saat ini
    n = np.sum(y == 0)               # Jumlah negatif dalam dataset/subset saat ini
    ent_d = modified_entropy(p, n)   # Entropy dataset sebelum dipartisi: Ent(D)
    ent_a = modified_conditional_entropy(y, X_col, y)  # Entropy setelah partisi: EntA(D)
    gain = ent_d - ent_a             # Gain(A) = Ent(D) - EntA(D)
    return gain


# ============================================================
# STRUKTUR DATA: NODE POHON KEPUTUSAN
# ============================================================

class ID3ModifiedNode:
    """
    Representasi satu node dalam pohon keputusan ID3 Modifikasi.

    Pohon keputusan dibangun secara rekursif dari root hingga leaf.
    Setiap node menyimpan informasi tentang:
        - Fitur mana yang digunakan sebagai pemisah (feature_index + threshold).
        - Referensi ke node anak kiri dan kanan.
        - Apakah node ini adalah daun (leaf) dan prediksi kelasnya.
        - Kedalaman node ini dalam pohon.

    Atribut:
        feature_index (int | None)  : Indeks fitur (kolom) yang digunakan sebagai
                                      splitting criterion pada node ini.
                                      None jika node adalah leaf.
        threshold     (float | None): Nilai ambang batas untuk binary split.
                                      Sampel dengan nilai fitur ≤ threshold
                                      diarahkan ke node.left; yang > threshold
                                      ke node.right. None jika node adalah leaf.
        children      (dict)        : Dictionary untuk split kategorikal
                                      (tidak digunakan pada implementasi ini
                                      yang menggunakan binary split numerik).
        left          (ID3ModifiedNode | None): Node anak kiri (sampel ≤ threshold).
        right         (ID3ModifiedNode | None): Node anak kanan (sampel > threshold).
        is_leaf       (bool)        : True jika node ini adalah daun (tidak memiliki
                                      node anak dan langsung memberikan prediksi).
        prediction    (int | None)  : Prediksi kelas (0 atau 1) jika is_leaf=True.
                                      Ditentukan oleh kelas mayoritas pada subset.
        depth         (int)         : Kedalaman node ini dalam pohon (root = 0).
    """
    def __init__(self):
        self.feature_index = None   # Indeks fitur splitting (None untuk leaf)
        self.threshold = None       # Nilai threshold binary split (None untuk leaf)
        self.children = {}          # Anak untuk split kategorikal (placeholder)
        self.left = None            # Node anak kiri: sampel dengan nilai ≤ threshold
        self.right = None           # Node anak kanan: sampel dengan nilai > threshold
        self.is_leaf = False        # Flag apakah ini node daun
        self.prediction = None      # Prediksi kelas jika is_leaf=True (0 atau 1)
        self.depth = 0              # Kedalaman node dalam pohon


# ============================================================
# CLASSIFIER UTAMA: ID3 MODIFIKASI
# ============================================================

class ID3ModifiedClassifier:
    """
    Classifier pohon keputusan ID3 Modifikasi untuk data TF-IDF.

    Mengimplementasikan algoritma ID3 dengan modifikasi formula entropy
    dari jurnal Asrianda et al. (2025). Classifier ini dirancang untuk
    data fitur numerik (seperti output TF-IDF), menggunakan strategi
    binary split pada setiap node.

    Perbedaan utama dengan ID3 klasik (Shannon):
        1. Formula entropy menggunakan logaritma natural yang dinormalisasi
           ke basis-2, bukan langsung log2.
        2. Lebih responsif terhadap distribusi data tidak seimbang
           (imbalanced dataset).
        3. Binary split pada fitur numerik: threshold dipilih sebagai
           midpoint antara dua nilai unik yang berdekatan.

    Alur kerja (fit → predict):
        1. fit(X, y)     : Bangun pohon keputusan dari data training.
        2. predict(X)    : Traversal pohon untuk setiap sampel baru.
        3. predict_proba : Estimasi probabilitas (simplified, confidence tetap).

    Atribut yang diset setelah fit():
        root                 (ID3ModifiedNode): Node akar pohon keputusan.
        n_features           (int)            : Jumlah fitur dalam data training.
        classes_             (np.ndarray)     : Array kelas yang dikenali [0, 1].
        feature_importances_ (np.ndarray)     : Array bobot kepentingan setiap fitur,
                                                dihitung dari akumulasi Information Gain.

    Args (Konstruktor):
        max_depth         (int): Kedalaman maksimum pohon yang diizinkan.
                                 Sesuai rekomendasi jurnal, nilai optimal
                                 umumnya ada di rentang 3–6. Default: 3.
        min_samples_split (int): Jumlah minimum sampel yang harus ada di
                                 suatu node agar node tersebut dapat dipecah
                                 (split) lebih lanjut. Default: 2.

    Referensi:
        Asrianda et al. (2025), JUTIF & EJET.
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth                  # Batas kedalaman pohon
        self.min_samples_split = min_samples_split  # Minimum sampel untuk melakukan split
        self.root = None                            # Root node, diisi setelah fit()
        self.n_features = None                      # Jumlah fitur, diisi setelah fit()
        self.classes_ = np.array([0, 1])            # Kelas yang dikenali: negatif=0, positif=1
        self.feature_importances_ = None            # Bobot fitur, diisi setelah fit()

    def fit(self, X, y):
        """
        Melatih model dengan membangun pohon keputusan dari data training.

        Proses pembangunan pohon dilakukan secara rekursif menggunakan
        metode _build_tree(). Setelah pohon selesai dibangun, nilai
        feature_importances_ dinormalisasi sehingga totalnya menjadi 1.0,
        memudahkan interpretasi relatif kepentingan setiap fitur.

        Args:
            X (array-like atau sparse matrix): Data fitur training dengan bentuk
                                               (n_samples, n_features). Menerima
                                               output TF-IDF (scipy sparse matrix)
                                               atau numpy array biasa.
            y (array-like)                   : Array label kelas training dengan
                                               bentuk (n_samples,). Nilai yang
                                               diharapkan: 0 (negatif) dan 1 (positif).

        Returns:
            self (ID3ModifiedClassifier): Mengembalikan instance classifier itu sendiri,
                                          memungkinkan method chaining.

        Catatan:
            Jika X berupa scipy sparse matrix (misalnya hasil TfidfVectorizer),
            secara otomatis dikonversi ke dense array menggunakan .toarray()
            sebelum diproses. Ini dapat memakan memori besar untuk dataset
            dengan banyak fitur.
        """
        # Konversi sparse matrix ke dense array jika diperlukan
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X)
        y = np.array(y)

        self.n_features = X.shape[1]                          # Simpan jumlah fitur
        self.feature_importances_ = np.zeros(self.n_features) # Inisialisasi array importance

        # Bangun pohon secara rekursif dari root (depth=0)
        self.root = self._build_tree(X, y, depth=0)

        # Normalisasi feature importance agar total = 1.0
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def _build_tree(self, X, y, depth):
        """
        Membangun pohon keputusan secara rekursif (fungsi internal).

        Ini adalah jantung dari algoritma ID3 Modifikasi. Fungsi ini
        dipanggil secara rekursif untuk membangun setiap node dalam pohon.
        Pada setiap pemanggilan, fungsi mencari fitur dan threshold terbaik
        berdasarkan Information Gain dari entropy modifikasi, kemudian
        membagi data dan memanggil dirinya sendiri untuk node anak.

        Kondisi berhenti (membuat leaf node):
            1. Kedalaman saat ini sudah mencapai max_depth.
            2. Jumlah sampel kurang dari min_samples_split.
            3. Semua sampel sudah termasuk satu kelas (dataset murni).
            4. Tidak ada fitur yang memberikan Information Gain positif.
            5. Salah satu sisi split menghasilkan subset kosong.

        Prediksi pada leaf node ditentukan oleh kelas mayoritas (majority voting)
        menggunakan np.bincount().argmax().

        Args:
            X     (np.ndarray): Data fitur subset saat ini, bentuk (n_samples, n_features).
            y     (np.ndarray): Label kelas subset saat ini, bentuk (n_samples,).
            depth (int)       : Kedalaman node saat ini (root dimulai dari 0).

        Returns:
            ID3ModifiedNode: Node yang sudah dikonfigurasi, bisa berupa:
                - Internal node: memiliki feature_index, threshold, left, right.
                - Leaf node   : memiliki is_leaf=True dan prediction terisi.
        """
        node = ID3ModifiedNode()
        node.depth = depth

        # Hitung statistik subset saat ini
        n_samples = len(y)
        p = np.sum(y == 1)   # Jumlah positif
        n = np.sum(y == 0)   # Jumlah negatif

        # ── Cek kondisi berhenti ─────────────────────────────────────────
        if (depth >= self.max_depth or          # Batas kedalaman tercapai
                n_samples < self.min_samples_split or  # Terlalu sedikit sampel
                p == 0 or n == 0):              # Dataset sudah murni satu kelas
            node.is_leaf = True
            node.prediction = int(np.bincount(y).argmax())  # Kelas mayoritas
            return node

        # ── Cari splitting criterion terbaik ────────────────────────────
        # Iterasi semua fitur dan semua kandidat threshold untuk mencari
        # kombinasi (feature_index, threshold) dengan Information Gain tertinggi.
        best_gain = -1          # Gain terbaik yang ditemukan sejauh ini
        best_feature = None     # Indeks fitur terbaik
        best_threshold = None   # Threshold terbaik untuk fitur tersebut

        for feat_idx in range(self.n_features):
            col = X[:, feat_idx]                  # Ambil nilai fitur ke-feat_idx
            unique_vals = np.unique(col)          # Nilai unik yang ada
            if len(unique_vals) < 2:
                # Fitur ini bernilai sama untuk semua sampel → tidak berguna
                continue

            # Kandidat threshold: titik tengah antara setiap pasang nilai unik berurutan
            # Contoh: unique=[0.0, 0.3, 0.7] → thresholds=[0.15, 0.5]
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for thresh in thresholds:
                # Binary split: 1 jika nilai > threshold, 0 jika ≤ threshold
                y_binary = (col > thresh).astype(int)
                gain = information_gain(y, y_binary)  # Hitung Gain untuk split ini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thresh

        # ── Jika tidak ada split yang menguntungkan → buat leaf ─────────
        if best_feature is None or best_gain <= 0:
            node.is_leaf = True
            node.prediction = int(np.bincount(y).argmax())
            return node

        # ── Catat kontribusi fitur terpilih ke feature importance ───────
        # Akumulasi Information Gain dari fitur ini (akan dinormalisasi setelah fit())
        self.feature_importances_[best_feature] += best_gain

        # Simpan informasi splitting pada node saat ini
        node.feature_index = best_feature
        node.threshold = best_threshold

        # ── Bagi data berdasarkan threshold terbaik ──────────────────────
        mask_left  = X[:, best_feature] <= best_threshold  # Sampel yang ke kiri
        mask_right = ~mask_left                            # Sampel yang ke kanan

        # Jika salah satu sisi kosong → tidak bisa split → buat leaf
        if np.sum(mask_left) == 0 or np.sum(mask_right) == 0:
            node.is_leaf = True
            node.prediction = int(np.bincount(y).argmax())
            return node

        # ── Rekursi: bangun subtree kiri dan kanan ───────────────────────
        node.left  = self._build_tree(X[mask_left],  y[mask_left],  depth + 1)
        node.right = self._build_tree(X[mask_right], y[mask_right], depth + 1)
        return node

    def _predict_one(self, x, node):
        """
        Melakukan prediksi untuk satu sampel dengan traversal pohon secara rekursif.

        Memulai dari root node dan terus bergerak ke node anak kiri atau kanan
        berdasarkan nilai fitur sampel dibandingkan threshold setiap node,
        sampai mencapai leaf node yang menyimpan prediksi kelas.

        Aturan traversal:
            - Jika x[node.feature_index] ≤ node.threshold → ke node.left
            - Jika x[node.feature_index] > node.threshold → ke node.right

        Args:
            x    (np.ndarray)    : Array 1D nilai fitur untuk satu sampel,
                                   panjangnya sama dengan n_features.
            node (ID3ModifiedNode): Node yang sedang dikunjungi. Pemanggilan
                                   pertama selalu dengan node=self.root.

        Returns:
            int: Prediksi kelas untuk sampel x (0 untuk negatif, 1 untuk positif).
        """
        if node.is_leaf:
            return node.prediction  # Sudah sampai leaf → kembalikan prediksi

        # Bandingkan nilai fitur dengan threshold node ini
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)   # Ke subtree kiri
        else:
            return self._predict_one(x, node.right)  # Ke subtree kanan

    def predict(self, X):
        """
        Melakukan prediksi kelas untuk seluruh dataset (batch prediction).

        Memanggil _predict_one() untuk setiap baris dalam X secara berurutan,
        menghasilkan array prediksi kelas dengan panjang sama dengan jumlah
        baris input.

        Args:
            X (array-like atau sparse matrix): Data fitur yang akan diprediksi,
                                               bentuk (n_samples, n_features).
                                               Harus memiliki jumlah fitur yang
                                               sama dengan data training.

        Returns:
            np.ndarray: Array prediksi kelas dengan bentuk (n_samples,).
                        Setiap elemen bernilai 0 (negatif) atau 1 (positif).

        Catatan:
            Sparse matrix akan dikonversi ke dense array secara otomatis.
        """
        # Konversi sparse matrix ke dense array jika diperlukan
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        """
        Mengestimasi probabilitas kelas untuk seluruh dataset (simplified).

        Implementasi ini menggunakan pendekatan sederhana (hard-coded confidence):
        kelas yang diprediksi mendapat probabilitas 0.85 dan kelas lainnya 0.15.
        Ini bukan estimasi probabilitas sejati (seperti Platt Scaling atau
        Laplace Smoothing), melainkan representasi kepercayaan yang seragam.

        Nilai 0.85/0.15 dipilih sebagai nilai kepercayaan konservatif yang
        mencerminkan bahwa model cukup yakin namun tidak 100% pasti.

        Args:
            X (array-like atau sparse matrix): Data fitur yang akan diprediksi,
                                               bentuk (n_samples, n_features).

        Returns:
            np.ndarray: Array probabilitas dengan bentuk (n_samples, 2).
                        Kolom 0 = probabilitas kelas 0 (negatif).
                        Kolom 1 = probabilitas kelas 1 (positif).
                        Contoh: [[0.15, 0.85], [0.85, 0.15], ...]

        Catatan untuk pengembangan lebih lanjut:
            Implementasi yang lebih akurat dapat menggunakan distribusi label
            pada leaf node yang dicapai oleh setiap sampel, sehingga
            probabilitas mencerminkan komposisi kelas sesungguhnya pada leaf tersebut.
        """
        preds = self.predict(X)                          # Dapatkan prediksi kelas dulu
        proba = np.zeros((len(preds), 2))                # Inisialisasi array probabilitas
        for i, p in enumerate(preds):
            proba[i][p] = 0.85       # Kelas yang diprediksi mendapat confidence 0.85
            proba[i][1 - p] = 0.15  # Kelas alternatif mendapat sisa 0.15
        return proba