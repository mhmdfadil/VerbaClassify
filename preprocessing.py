"""
preprocessing.py
================
Modul ini mengimplementasikan pipeline preprocessing teks Bahasa Indonesia
yang dirancang khusus untuk klasifikasi respon masyarakat terhadap isu
pelecehan seksual verbal pada platform X (Twitter).

Pipeline preprocessing mengikuti tahapan yang tercantum dalam proposal
skripsi Rizka Mardiah Putri Buyung Lubis (2026):
    1. Case Folding   : Mengubah semua huruf ke huruf kecil (lowercase).
    2. Cleansing      : Menghapus noise (URL, mention, hashtag, simbol, angka).
    3. Normalisasi    : Mengubah kata slang/alay ke bentuk standar.
    4. Tokenizing     : Memecah teks menjadi token (kata) individual.
    5. Stopword Removal: Menghapus kata-kata umum yang tidak bermakna.
    6. Stemming       : Mengubah kata berimbuhan ke bentuk kata dasarnya.

Library utama yang digunakan:
    - Sastrawi : Library NLP Bahasa Indonesia untuk stemming (algoritma
                 Nazief-Adriani) dan daftar stopwords bawaan.
    - NLTK     : Tokenizer teks (word_tokenize dengan mode Indonesian).
    - re       : Ekspresi reguler untuk cleansing berbasis pola.

Strategi Fallback:
    Jika library Sastrawi tidak tersedia (ImportError), sistem akan tetap
    berjalan tanpa stemming (token dikembalikan apa adanya) dan hanya
    menggunakan stopwords dari _extra_stopwords yang sudah didefinisikan
    secara manual di modul ini.

Penggunaan:
    from preprocessing import preprocess, preprocess_batch

    # Preproses satu teks
    hasil = preprocess("@user Ini pelecehan verbal banget!!")

    # Preproses banyak teks sekaligus
    hasil_batch = preprocess_batch(["teks1", "teks2", ...])

Author   : Rizka Mardiah Putri Buyung Lubis (220170183)
Institusi: Universitas Malikussaleh, 2026
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize

# ── Import Sastrawi dengan fallback ──────────────────────────────────────────
# Sastrawi adalah library NLP Bahasa Indonesia yang menyediakan stemmer
# berbasis algoritma Nazief-Adriani dan daftar stopwords bahasa Indonesia.
# Jika tidak terinstal, modul tetap bisa berjalan tanpa fitur stemming.
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _sastrawi_available = True
except ImportError:
    _sastrawi_available = False

# ── Download NLTK data yang dibutuhkan ───────────────────────────────────────
# Menggunakan try/except untuk menghindari download ulang jika sudah ada.
# 'punkt' dan 'punkt_tab' dibutuhkan oleh word_tokenize().
# 'stopwords' dibutuhkan sebagai referensi (tidak langsung digunakan di sini).
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ── Inisialisasi Sastrawi (hanya jika tersedia) ───────────────────────────────
# Stemmer Sastrawi menggunakan algoritma Nazief-Adriani untuk menghapus
# imbuhan bahasa Indonesia (awalan, akhiran, konfiks) dari kata.
# StopWordRemoverFactory menyediakan daftar stopwords baku bahasa Indonesia.
if _sastrawi_available:
    _stemmer_factory = StemmerFactory()
    _stemmer = _stemmer_factory.create_stemmer()           # Instance stemmer siap pakai
    _sw_factory = StopWordRemoverFactory()
    _sastrawi_stopwords = set(_sw_factory.get_stop_words()) # Konversi ke set untuk O(1) lookup
else:
    # Fallback: gunakan stemmer None (stemming dinonaktifkan)
    _stemmer = None
    _sastrawi_stopwords = set()

# ── Stopwords tambahan ────────────────────────────────────────────────────────
# Daftar kata tambahan yang tidak ada di kamus Sastrawi tetapi sering muncul
# dalam komentar media sosial Bahasa Indonesia informal, termasuk:
#   - Singkatan umum media sosial (yg, dgn, bgt, dll.)
#   - Kata ganti informal (gw, lu, lo, dll.)
#   - Ekspresi tawa/tidak bermakna (haha, wkwk, dll.)
#   - Kata hubung dan partikel umum
#   - Kata yang tidak relevan untuk analisis sentimen pelecehan verbal
_extra_stopwords = {
    'yg', 'dgn', 'gak', 'ga', 'gw', 'gue', 'lo', 'lu', 'aja', 'doang',
    'deh', 'dong', 'sih', 'lah', 'nih', 'kah', 'tuh', 'mah', 'kok',
    'bgt', 'bget', 'bgt', 'bngt', 'banget', 'amat', 'kali', 'juga',
    'udah', 'udh', 'sudah', 'sdh', 'udh', 'jadi', 'jd', 'sama', 'sm',
    'kalo', 'klo', 'klu', 'kalau', 'emg', 'emang', 'memang',
    'amp', 'rt', 'via', 'http', 'https', 'co', 'bit', 'ly',
    'ya', 'yaa', 'iya', 'iyaa', 'ok', 'oke', 'oks',
    'haha', 'hahaha', 'wkwk', 'wkwkwk', 'wkwkwkwk', 'xixi',
    'anjir', 'anjg', 'anjing', 'bgst', 'kontol', 'babi',
    'dr', 'ke', 'di', 'dan', 'atau', 'juga', 'tapi', 'tp',
    'ada', 'tdk', 'tidak', 'bukan', 'belum', 'akan', 'sdg', 'lagi',
    'untuk', 'utk', 'dari', 'pd', 'pada', 'oleh', 'dlm', 'dalam',
    'paling', 'sangat', 'lebih', 'kurang', 'cuma', 'hanya',
    'mau', 'mau', 'bisa', 'bs', 'harus', 'perlu',
    'org', 'orang', 'itu', 'ini', 'tsb', 'sbg', 'kyk', 'kayak',
    'gmn', 'gimana', 'gini', 'gitu', 'bgini', 'bgtu',
    'yuk', 'ayo', 'mari', 'tolong', 'tlong',
    'pake', 'pakai', 'pkai', 'trus', 'terus',
    'nah', 'nih', 'nah', 'kan', 'kan', 'tp', 'kyk',
    'cewe', 'cowo', 'cewe', 'cowok', 'cewek',
}

# ── Gabungan stopwords final ──────────────────────────────────────────────────
# Menggabungkan stopwords dari Sastrawi (baku) dengan stopwords tambahan
# informal media sosial menggunakan operasi union set (|).
STOPWORDS = _sastrawi_stopwords | _extra_stopwords

# ── Kamus normalisasi kata slang/alay ─────────────────────────────────────────
# Kamus ini digunakan pada tahap normalisasi untuk mengganti kata-kata
# tidak baku yang sering digunakan di media sosial ke bentuk bakunya.
# Key   = kata slang/alay/singkatan yang akan dicari
# Value = bentuk baku pengganti
# Kamus ini dirancang khusus untuk konteks komentar pelecehan seksual verbal
# di platform X (Twitter), mencakup kata-kata yang relevan dengan domain ini.
SLANG_DICT = {
    # ── Kata tugas dan konjungsi ──────────────────────────────────────────
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    # ── Kata negasi ───────────────────────────────────────────────────────
    'tdk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'engga': 'tidak',
    'enggak': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    # ── Kata keterangan waktu ─────────────────────────────────────────────
    'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
    # ── Kata intensifier ─────────────────────────────────────────────────
    'bgt': 'banget', 'bngt': 'banget', 'bget': 'banget',
    # ── Konjungsi kondisional ─────────────────────────────────────────────
    'klo': 'kalau', 'klu': 'kalau', 'kalo': 'kalau',
    # ── Kata penegas ──────────────────────────────────────────────────────
    'emg': 'memang', 'emang': 'memang', 'mmg': 'memang',
    # ── Kata hubung dan partikel ──────────────────────────────────────────
    'sm': 'sama', 'jd': 'jadi', 'tp': 'tapi', 'cuma': 'hanya',
    'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
    # ── Kata ganti orang pertama ──────────────────────────────────────────
    'gw': 'saya', 'gue': 'saya', 'aku': 'saya', 'sy': 'saya',
    # ── Kata ganti orang kedua ────────────────────────────────────────────
    'lo': 'kamu', 'lu': 'kamu', 'elo': 'kamu', 'elu': 'kamu',
    # ── Kata benda umum ───────────────────────────────────────────────────
    'org': 'orang', 'bnyk': 'banyak', 'sdkt': 'sedikit',
    # ── Kata keterangan keadaan ───────────────────────────────────────────
    'blm': 'belum', 'blom': 'belum', 'belom': 'belum',
    'msh': 'masih', 'msih': 'masih',
    'bs': 'bisa', 'bsa': 'bisa',
    'lg': 'lagi', 'lgi': 'lagi',
    # ── Preposisi ─────────────────────────────────────────────────────────
    'dr': 'dari', 'dlm': 'dalam', 'pd': 'pada',
    # ── Kata tambahan ─────────────────────────────────────────────────────
    'jg': 'juga', 'jga': 'juga',
    'kyk': 'seperti', 'kayak': 'seperti',
    'sbg': 'sebagai', 'sbgai': 'sebagai',
    'gmn': 'bagaimana', 'gimana': 'bagaimana',
    'krn': 'karena', 'karna': 'karena',
    # ── Istilah gender (normalisasi ke bentuk baku) ───────────────────────
    'cewek': 'perempuan', 'cewe': 'perempuan',
    'cowok': 'laki-laki', 'cowo': 'laki-laki',
    # ── Istilah domain pelecehan seksual (normalisasi variasi ejaan) ──────
    'pelecehan': 'pelecehan', 'lécéhan': 'pelecehan',
    'sexual': 'seksual', 'sexualiti': 'seksual',
    'seksi': 'seksi', 'sexi': 'seksi', 'sexy': 'seksi',
    'menggoda': 'menggoda', 'goda': 'goda',
    'korban': 'korban', 'pelaku': 'pelaku',
    'hukum': 'hukum', 'lapor': 'lapor', 'proses': 'proses',
}


# ============================================================
# FUNGSI-FUNGSI PREPROCESSING PER TAHAP
# Setiap fungsi di bawah merepresentasikan satu tahap dalam
# pipeline preprocessing yang dijelaskan pada proposal Rizka (2026).
# ============================================================

def case_folding(text: str) -> str:
    """
    Tahap 1: Mengubah semua karakter teks ke huruf kecil (lowercase).

    Case folding memastikan konsistensi representasi teks sehingga
    kata yang sama dengan penulisan berbeda (misal "Pelecehan" vs
    "pelecehan" vs "PELECEHAN") diperlakukan sebagai token yang sama
    oleh semua tahap selanjutnya.

    Selain lowercase, fungsi ini juga menghapus spasi berlebih di
    awal dan akhir string menggunakan .strip().

    Args:
        text (str): Teks asli yang akan diproses.

    Returns:
        str: Teks dalam huruf kecil semua, tanpa spasi leading/trailing.

    Contoh:
        >>> case_folding("PELECEHAN Seksual Verbal")
        'pelecehan seksual verbal'
        >>> case_folding("  Halo Dunia  ")
        'halo dunia'
    """
    return text.lower().strip()


def cleansing(text: str) -> str:
    """
    Tahap 2: Membersihkan teks dari elemen-elemen noise yang tidak relevan.

    Menghapus berbagai jenis noise yang umum ditemukan pada komentar
    media sosial secara berurutan menggunakan ekspresi reguler (regex).
    Urutan penghapusan penting: URL dan mention dihapus dulu sebelum
    simbol agar tidak meninggalkan karakter sisa.

    Urutan pembersihan yang dilakukan:
        1. URL (http://, https://, www.)    → Hapus
        2. Mention (@username)              → Hapus
        3. Hashtag (#topik)                 → Hapus
        4. Emoji dan karakter non-ASCII     → Hapus (encode-decode ascii)
        5. Angka (digit 0-9)                → Hapus
        6. Tanda baca dan simbol lainnya    → Ganti dengan spasi
        7. Underscore (_)                   → Ganti dengan spasi
        8. Spasi berlebih (multiple spaces) → Normalisasi ke satu spasi

    Args:
        text (str): Teks yang sudah melalui case_folding.

    Returns:
        str: Teks yang sudah dibersihkan dari semua elemen noise.
             Hanya mengandung huruf alfanumerik dan spasi tunggal.

    Contoh:
        >>> cleansing("@user123 Ini #pelecehan banget!! https://t.co/xxx 😡")
        'ini banget'
        >>> cleansing("halo_dunia 2024 bro...")
        'halo dunia bro'
    """
    # Langkah 1: Hapus semua URL (format http/https dan www)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Langkah 2: Hapus semua mention (@username)
    text = re.sub(r'@\w+', '', text)

    # Langkah 3: Hapus semua hashtag (#topik)
    text = re.sub(r'#\w+', '', text)

    # Langkah 4: Hapus emoji dan semua karakter non-ASCII
    # encode('ascii', 'ignore') → buang karakter yang tidak bisa direpresentasikan
    # dalam ASCII (termasuk emoji, karakter unicode khusus, dll.)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Langkah 5: Hapus semua karakter angka (digit 0-9)
    text = re.sub(r'\d+', '', text)

    # Langkah 6: Hapus tanda baca dan simbol, ganti dengan spasi
    # [^\w\s] berarti: hapus semua karakter yang bukan word character (\w)
    # dan bukan whitespace (\s)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Langkah 7: Ganti underscore dengan spasi
    # (karena \w mencakup underscore, perlu dihapus secara terpisah)
    text = re.sub(r'_', ' ', text)

    # Langkah 8: Normalisasi spasi berlebih menjadi satu spasi
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_slang(text: str) -> str:
    """
    Tahap 3: Mengganti kata slang dan singkatan informal ke bentuk bakunya.

    Menggunakan kamus SLANG_DICT yang sudah didefinisikan untuk melakukan
    substitusi kata per kata. Setiap token dalam teks dicari di kamus,
    dan jika ditemukan, diganti dengan padanan bakunya. Token yang tidak
    ada di kamus dikembalikan apa adanya (tidak berubah).

    Normalisasi ini penting karena komentar media sosial sangat sering
    menggunakan variasi ejaan yang tidak baku. Misalnya, "gak bisa" dan
    "tidak bisa" harus diperlakukan sebagai ekspresi yang sama oleh model.

    Args:
        text (str): Teks yang sudah melalui case_folding dan cleansing.

    Returns:
        str: Teks dengan kata-kata slang/alay yang sudah diganti ke
             bentuk bakunya, disatukan kembali menjadi string.

    Contoh:
        >>> normalize_slang("gw gak suka pelecehan bgt")
        'saya tidak suka pelecehan banget'
        >>> normalize_slang("lo udah lapor blm")
        'kamu sudah lapor belum'
    """
    tokens = text.split()   # Pecah teks menjadi kata-kata
    # Untuk setiap token, cek di kamus slang. Jika ada → ganti; jika tidak → tetap
    normalized = [SLANG_DICT.get(t, t) for t in tokens]
    return ' '.join(normalized)  # Gabungkan kembali menjadi string


def tokenizing(text: str) -> list:
    """
    Tahap 4: Memecah teks menjadi daftar token (kata-kata individual).

    Menggunakan word_tokenize dari NLTK dengan mode bahasa Indonesia.
    Tokenisasi lebih akurat dibanding sekadar text.split() karena
    word_tokenize dapat menangani tanda baca yang melekat pada kata
    (walaupun setelah cleansing seharusnya tidak banyak tersisa).

    Jika tokenisasi NLTK gagal (misal karena data punkt tidak tersedia),
    fungsi akan fallback ke pemisahan berdasarkan spasi sederhana.

    Args:
        text (str): Teks yang sudah melewati tahap normalisasi slang.

    Returns:
        list[str]: Daftar token (kata-kata) dari teks yang diinput.
                   Bisa berupa list kosong [] jika teks input adalah string kosong.

    Contoh:
        >>> tokenizing("pelecehan seksual verbal sangat merugikan korban")
        ['pelecehan', 'seksual', 'verbal', 'sangat', 'merugikan', 'korban']
    """
    try:
        tokens = word_tokenize(text, language='indonesian')
    except Exception:
        # Fallback: pisahkan berdasarkan spasi jika NLTK gagal
        tokens = text.split()
    return tokens


def stopword_removal(tokens: list) -> list:
    """
    Tahap 5: Menghapus token yang termasuk dalam daftar stopwords.

    Menyaring daftar token dan hanya mempertahankan kata-kata yang:
        1. Tidak ada dalam himpunan STOPWORDS (gabungan Sastrawi + extra).
        2. Memiliki panjang lebih dari 2 karakter (menghindari token
           terlalu pendek yang umumnya tidak bermakna).

    Menghapus stopwords mengurangi dimensi fitur TF-IDF karena kata-kata
    umum yang tidak membawa makna diskriminatif (seperti "dan", "yang",
    "di") tidak akan masuk ke dalam vocabulary vectorizer.

    Args:
        tokens (list[str]): Daftar token hasil tahap tokenizing.

    Returns:
        list[str]: Daftar token yang sudah difilter — hanya kata bermakna
                   yang panjangnya > 2 karakter dan tidak ada di STOPWORDS.

    Contoh:
        >>> stopword_removal(['pelecehan', 'seksual', 'dan', 'di', 'korban', 'ya'])
        ['pelecehan', 'seksual', 'korban']
    """
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def stemming(tokens: list) -> list:
    """
    Tahap 6: Mengubah token ke bentuk kata dasarnya (root word).

    Menggunakan Sastrawi Stemmer yang mengimplementasikan algoritma
    Nazief-Adriani untuk bahasa Indonesia. Algoritma ini menghapus
    imbuhan (awalan/prefiks, akhiran/sufiks, dan konfiks) untuk
    mendapatkan kata dasar.

    Contoh transformasi stemming:
        "melaporkan" → "lapor"
        "pelecehan"  → "leceh"
        "korbannya"  → "korban"
        "menggoda"   → "goda"

    Manfaat stemming dalam konteks ini:
        Mengurangi variasi morfologi sehingga kata-kata yang memiliki
        makna sama tetapi berbeda bentuk infleksi diperlakukan sebagai
        fitur yang sama oleh TF-IDF. Ini meningkatkan generalisasi model.

    Args:
        tokens (list[str]): Daftar token setelah stopword removal.

    Returns:
        list[str]: Daftar token dalam bentuk kata dasar. Jika Sastrawi
                   tidak tersedia (_stemmer is None), token dikembalikan
                   tanpa perubahan (identity transform) sebagai fallback.

    Catatan:
        Stemming Sastrawi cukup lambat untuk dataset besar karena
        memproses setiap token secara individual. Untuk 1000 teks,
        proses ini bisa memakan beberapa detik.
    """
    if _stemmer is not None:
        # Sastrawi tersedia: stem setiap token
        return [_stemmer.stem(t) for t in tokens]
    # Sastrawi tidak tersedia: kembalikan token apa adanya (tanpa stemming)
    return tokens


# ============================================================
# PIPELINE PREPROCESSING UTAMA
# ============================================================

def preprocess(text: str) -> str:
    """
    Menjalankan pipeline preprocessing lengkap untuk satu teks.

    Memanggil keenam tahap preprocessing secara berurutan sesuai
    alur sistem yang dirancang dalam proposal Rizka (2026):
        Case Folding → Cleansing → Normalisasi → Tokenizing
        → Stopword Removal → Stemming

    Setelah stemming, dilakukan filter tambahan untuk menghapus token
    yang panjangnya ≤ 1 karakter (token satu huruf yang mungkin tersisa
    setelah stemming agresif).

    Args:
        text (str): Teks komentar asli dari platform X yang akan dipreproses.
                    Bisa berupa string apapun, termasuk yang mengandung
                    mention, hashtag, URL, emoji, dan kata-kata informal.

    Returns:
        str: Teks yang sudah dipreproses, siap digunakan sebagai input
             untuk TF-IDF vectorizer. Berupa string kata-kata dasar yang
             dipisahkan spasi. Mengembalikan string kosong '' jika input
             tidak valid atau seluruh token terhapus.

    Contoh:
        >>> preprocess("@user2024 Ini pelecehan seksual banget!! Stop!!")
        'leceh seksual'

        >>> preprocess("Pelaku harus dihukum seberat-beratnya #JusticeForVictim")
        'pelaku hukum berat'

        >>> preprocess("")
        ''

    Catatan:
        Hasil preprocessing sangat bergantung pada kualitas kamus SLANG_DICT
        dan daftar STOPWORDS. Teks yang sangat singkat atau penuh noise
        bisa menghasilkan string kosong setelah preprocessing.
    """
    # Validasi: kembalikan string kosong jika input tidak valid
    if not text or not isinstance(text, str):
        return ''

    # Tahap 1: Case Folding — ubah semua huruf ke lowercase
    text = case_folding(text)

    # Tahap 2: Cleansing — hapus URL, mention, hashtag, emoji, angka, simbol
    text = cleansing(text)

    # Tahap 3: Normalisasi — ganti kata slang/alay ke bentuk baku
    text = normalize_slang(text)

    # Tahap 4: Tokenizing — pecah teks menjadi daftar token kata
    tokens = tokenizing(text)

    # Tahap 5: Stopword Removal — hapus kata yang tidak bermakna
    tokens = stopword_removal(tokens)

    # Tahap 6: Stemming — ubah kata berimbuhan ke kata dasar
    tokens = stemming(tokens)

    # Filter tambahan: hapus token yang panjangnya ≤ 1 karakter
    # (bisa terjadi setelah stemming yang sangat agresif)
    tokens = [t for t in tokens if len(t) > 1]

    return ' '.join(tokens)  # Gabungkan kembali menjadi string


def preprocess_batch(texts: list) -> list:
    """
    Menjalankan pipeline preprocessing untuk banyak teks sekaligus (batch).

    Memanggil fungsi preprocess() secara berurutan untuk setiap elemen
    dalam daftar input. Hasil akhirnya adalah daftar dengan panjang yang
    sama, di mana setiap elemen adalah hasil preprocessing dari teks
    pada indeks yang bersesuaian.

    Fungsi ini digunakan oleh:
        - Proses upload dataset: mempreproses semua teks sebelum disimpan ke DB.
        - Pipeline training: mempreproses ulang jika preprocessed_text di DB kosong.
        - Prediksi tunggal: dipanggil dengan list berisi satu elemen.

    Args:
        texts (list[str]): Daftar teks komentar asli yang akan dipreproses.
                           Bisa berisi ratusan hingga ribuan string.

    Returns:
        list[str]: Daftar hasil preprocessing, panjangnya sama dengan
                   panjang input. Elemen kosong '' dihasilkan untuk teks
                   yang tidak valid atau kosong setelah preprocessing.

    Catatan performa:
        Untuk dataset besar (>1000 teks), proses ini bisa memakan waktu
        beberapa menit karena stemming Sastrawi memproses setiap token
        secara individual. Hasil preprocessing disimpan ke database
        agar tidak perlu diulang saat eksperimen berikutnya.

    Contoh:
        >>> preprocess_batch(["pelecehan itu salah", "ini candaan biasa"])
        ['leceh salah', 'canda biasa']
    """
    return [preprocess(t) for t in texts]