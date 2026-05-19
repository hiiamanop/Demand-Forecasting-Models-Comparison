# Demand Forecasting pada Dataset E-Commerce Berdurasi Pendek

**Penelitian:** Mata Kuliah Data Science — S2 ITB  
**Dataset:** Olist Brazilian E-Commerce Public Dataset (Kaggle, 2016–2018)

---

## Pertanyaan Penelitian

> **Algoritma forecasting apa yang paling cocok untuk data permintaan (demand) e-commerce yang berdurasi pendek, tidak stasioner, dan memiliki variabilitas tinggi?**

---

## Karakteristik Dataset

Sebelum memilih model, penting memahami sifat-sifat dataset ini karena karakteristik inilah yang menentukan model mana yang bisa atau tidak bisa bekerja dengan baik.

**Sumber:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

| File | Baris | Kolom |
|---|---|---|
| `olist_orders_dataset.csv` | 99.441 | 8 |
| `olist_order_items_dataset.csv` | 112.650 | 7 |
| `olist_products_dataset.csv` | 32.951 | 9 |
| `product_category_name_translation.csv` | 71 | 2 |

Setelah merge, filter `order_status = 'delivered'`, dan agregasi mingguan:

| Karakteristik | Nilai | Implikasi |
|---|---|---|
| Rentang waktu | Sep 2016 – Ags 2018 (~2 tahun) | Dataset pendek |
| Panjang training set | ~78 minggu per kategori | Terlalu pendek untuk seasonal 52-minggu |
| Granularitas | Mingguan (W-MON) | Cukup untuk lag features |
| Stasioneritas (ADF) | p-value = 0.36 (tidak stasioner) | Ada tren → ARIMA perlu differencing |
| Variabilitas (XYZ) | Semua kategori kelas Y atau Z | Tidak ada kategori stabil (CV < 0.5) |
| Pola platform | Tumbuh 2017 → melambat 2018 | Bias positif sistemik pada semua model |

**Kesimpulan karakteristik:** Dataset ini tergolong *challenging* secara forecasting karena tidak memenuhi asumsi dasar sebagian besar model statistik klasik.

---

## Klasifikasi ABC/XYZ

### ABC (berdasarkan volume kumulatif)

| Kelas | Jumlah Kategori | Keterangan |
|---|---|---|
| A | 15 | Top 80% volume |
| B | 17 | 80–95% volume |
| C | 40 | Sisanya |

### XYZ (berdasarkan Coefficient of Variation)

| Kelas | Jumlah Kategori | CV |
|---|---|---|
| X | **0** | CV < 0.5 — stabil |
| Y | 36 | CV 0.5–1.0 — moderat |
| Z | 36 | CV > 1.0 — sangat fluktuatif |

Tidak ada satu pun kategori yang masuk kelas X. Ini adalah sinyal bahwa seluruh platform masih dalam fase pertumbuhan dan belum mencapai pola demand yang stabil — kondisi yang tidak menguntungkan bagi model statistik berbasis asumsi stasioneritas.

---

## Metode & Pendekatan

Penelitian ini menguji 5 pendekatan dari yang paling sederhana hingga paling kompleks, dibagi ke dalam 12 section notebook:

| Section | Metode | Kategori Pendekatan |
|---|---|---|
| 4 | Moving Average (window 4 minggu) | Naive baseline |
| 5 | Holt-Winters ETS | Statistik klasik |
| 6 | SARIMA (1,1,1)(1,0,1,4) | Statistik klasik |
| 7–10 | LightGBM + lag/rolling features | Machine learning |
| 11 | LightGBM + Optuna hyperparameter tuning | ML + optimasi |
| 12 | LightGBM + semua improvement + Prophet | ML vs dedicated tool |

**Feature engineering LightGBM (baseline):**  
Lag 1–12 minggu, rolling mean/std window 4/8/12, fitur kalender (week of year, month, quarter).

**Feature engineering LightGBM (improved):**  
Ditambah lag 52 (year-over-year), platform demand, flag hari libur Brasil, log transform target, bias correction.

---

## Hasil Eksperimen

### Perbandingan 4 Model Dasar — 6 Kategori A-class (Section 10)

| Kategori | CV | Mean/Minggu | MA | ETS | SARIMA | LightGBM |
|---|---|---|---|---|---|---|
| bed_bath_table | 0.66 | 97.7 | 16.8% | 41.1% | 17.5% | **16.2%** |
| sports_leisure | 0.63 | 79.1 | 40.4% | 60.1% | 45.0% | **20.9%** |
| furniture_decor | 0.65 | 74.4 | 27.1% | 39.9% | 27.3% | **22.1%** |
| computers_accessories | 0.80 | 71.6 | 53.7% | 69.6% | 43.5% | **22.1%** |
| health_beauty | 0.75 | 69.6 | 28.4% | 24.5% | 24.7% | **24.0%** |
| cool_stuff | 0.61 | 39.0 | 61.6% | 83.1% | 62.1% | **36.8%** |
| **Rata-rata** | | | 38.0% | 53.1% | 36.7% | **23.7%** |

LightGBM unggul di **6 dari 6 kategori (100%)**.

### Grand Comparison — Semua Versi Model (Section 12)

| Kategori | Baseline LGB | Tuned LGB | Improved LGB | Walk-Forward | Prophet |
|---|---|---|---|---|---|
| bed_bath_table | 16.2% ✓ | **14.2%** ✓ | 17.5% ✓ | 19.2% ✓ | 47.4% ✗ |
| furniture_decor | 22.1% ✓ | **22.1%** ✓ | 22.1% ✓ | 25.1% ~ | 42.0% ✗ |
| sports_leisure | 20.9% ✓ | **18.2%** ✓ | 26.9% ~ | 25.1% ~ | 77.4% ✗ |
| health_beauty | 24.0% ✓ | **22.0%** ✓ | 27.3% ~ | 22.3% ✓ | 20.2% ✓ |
| computers_accessories | 22.1% ✓ | **21.6%** ✓ | 37.4% ✗ | 25.3% ~ | 94.3% ✗ |
| cool_stuff | 36.8% ✗ | **30.6%** ~ | 77.4% ✗ | 81.6% ✗ | 190.2% ✗ |
| **Rata-rata** | 23.7% | **21.4%** | 34.8% | 33.1% | 78.6% |

Keterangan WMAPE: ✓ < 25% (Baik) · ~ 25–35% (Acceptable) · ✗ > 35% (Red flag)

---

## Kesimpulan Penelitian

### Jawaban atas Pertanyaan Penelitian

Untuk dataset e-commerce berdurasi pendek (~2 tahun), tidak stasioner, dan variabilitas tinggi:

> **LightGBM dengan lag features dan hyperparameter tuning adalah pendekatan yang paling sesuai**, dengan rata-rata WMAPE 21.4% — mengungguli semua model statistik klasik maupun Prophet.

### Mengapa Tiap Model Berhasil atau Gagal

**Moving Average — rata-rata WMAPE 38.0%**  
Gagal karena tidak memiliki komponen tren. Ketika demand sedang tumbuh, MA selalu tertinggal (lag effect). Namun pada kategori dengan tren yang sudah stabil, MA bisa menyaingi SARIMA.

**Holt-Winters ETS — rata-rata WMAPE 53.1%**  
Model terburuk. ETS membutuhkan minimal 2 siklus penuh untuk mengestimasi komponen seasonal 52-minggu. Dengan hanya ~78 minggu training, ETS terpaksa fallback ke Holt double (trend only) — kehilangan seluruh komponen musiman. Ini adalah contoh **model yang salah asumsi** terhadap panjang data.

**SARIMA — rata-rata WMAPE 36.7%**  
Mendekati LightGBM baseline (23.7% vs 36.7%), tapi tetap kalah. SARIMA memerlukan stasioneritas yang dipaksakan melalui differencing, dan parameter (p,d,q) yang ditetapkan secara manual tidak optimal untuk semua kategori sekaligus.

**LightGBM Baseline — rata-rata WMAPE 23.7%**  
Langsung kompetitif tanpa asumsi distribusi apapun. Lag features secara implisit mengajarkan model pola autokorelasi seperti yang dilakukan AR dalam ARIMA, tetapi lebih fleksibel karena bersifat non-parametrik.

**LightGBM Tuned (Optuna) — rata-rata WMAPE 21.4% ← TERBAIK**  
Perbaikan kecil tapi konsisten di semua kategori. Bayesian optimization menemukan kombinasi `n_estimators`, `learning_rate`, dan `num_leaves` yang optimal per kategori tanpa overfitting.

**LightGBM Improved — rata-rata WMAPE 34.8%**  
Lebih buruk dari baseline. Penambahan `lag_52` membutuhkan 52 minggu lookback — dengan training set 78 minggu, baris efektif yang bisa digunakan untuk training menjadi sangat sedikit. Ini adalah contoh **feature engineering yang melebihi kapasitas data**.

**Prophet — rata-rata WMAPE 78.6%**  
Gagal total. Meskipun dirancang khusus untuk time series bisnis, Prophet memerlukan data multi-tahun untuk mengestimasi `yearly_seasonality` dengan andal. Dengan data ~1.8 tahun, Prophet overfit pada pola musiman yang tidak cukup terwakilkan. Satu-satunya pengecualian: health_beauty (20.2%) yang memiliki pola paling smooth dan konsisten.

### Tiga Prinsip yang Ditemukan

**1. Kesesuaian asumsi model dengan sifat data lebih penting dari kompleksitas model**  
ETS dan Prophet dirancang untuk data multi-tahun. Ketika diterapkan pada data ~2 tahun, keduanya underperform bukan karena modelnya buruk, tapi karena asumsinya tidak terpenuhi. LightGBM tidak memiliki asumsi distribusi sehingga lebih robust terhadap kondisi ini.

**2. Optimasi hyperparameter lebih efektif daripada penambahan fitur pada dataset pendek**  
Tuned LGB (21.4%) > Improved LGB (34.8%), padahal Improved LGB memiliki fitur yang lebih kaya. Ketika jumlah data terbatas, menambah dimensi fitur justru meningkatkan risiko overfitting dan mengurangi baris training efektif.

**3. Volume demand lebih menentukan akurasi daripada variabilitas (CV)**  
cool_stuff memiliki CV terendah (0.61) tapi WMAPE terburuk (36.8%). computers_accessories memiliki CV tertinggi (0.80) tapi WMAPE hanya 21.6%. Kategori dengan mean demand ≥ 70 order/minggu lebih mudah diprediksi karena noise proporsional terhadap sinyal lebih kecil.

### Anomali: cool_stuff dan Lifecycle Pattern

cool_stuff tidak bisa diprediksi dengan baik oleh model apapun (WMAPE terbaik 30.6%). Ini bukan kegagalan model — melainkan pola demand-nya memang tidak bisa dimodelkan sebagai time series stasioner:

```
Minggu 1–12   : demand = 0 (produk baru, belum terdistribusi)
Minggu 13–78  : demand naik drastis → puncak  (fase growth)
Minggu 79–98  : demand turun tajam             (fase decline — ini periode test)
```

Model yang dilatih hanya pada fase growth akan selalu over-forecast ketika demand mulai turun. Kasus ini memerlukan **model lifecycle** (Bass Diffusion Model, kurva S) yang secara eksplisit memodelkan fase introduction → growth → maturity → decline, bukan model time series generik.

---

## Safety Stock (Tambahan — cool_stuff)

Sebagai contoh aplikasi praktis dari output forecasting:

*Asumsi lead time: 2 minggu. Parameter: mean = 39.0 unit/minggu, std = 24.6 unit/minggu.*

| Service Level | Z | Safety Stock | Reorder Point |
|---|---|---|---|
| 95% (standar A-class) | 1.65 | 70 unit | 148 unit |
| 99% (item kritis) | 2.33 | 99 unit | 177 unit |

Formula: **SS = Z × σ_demand × √(Lead Time + Review Period)**

---

## Referensi Metrik Evaluasi

| Metrik | Formula | Threshold |
|---|---|---|
| WMAPE | Σ\|aktual−pred\| / Σ\|aktual\| | < 25% Baik · 25–35% Acceptable · > 35% Red flag |
| Bias | mean(pred−aktual) / mean(aktual) | ±5% Sehat · > ±10% Ada masalah struktural |
| MAE | mean(\|aktual−pred\|) | Satuan unit order (absolut, bukan persentase) |

WMAPE dipilih sebagai metrik utama (bukan MAPE) karena robust terhadap nilai aktual mendekati 0 — kondisi yang umum pada kategori baru seperti cool_stuff.

---

## Setup & Cara Menjalankan

**Requirements:** Python 3.13+

```bash
# 1. Masuk ke folder proyek
cd demand_forecasting

# 2. Buat virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# 3. Install dependensi
pip install pandas numpy statsmodels lightgbm scikit-learn \
            matplotlib seaborn jupyter optuna prophet holidays

# 4. Jalankan notebook
jupyter notebook demand_forecasting.ipynb
```

| Library | Versi | Kegunaan |
|---|---|---|
| pandas | 3.0.3 | Data processing |
| numpy | 2.4.6 | Komputasi numerik |
| statsmodels | 0.14.6 | ETS, SARIMA, ADF test, ACF/PACF |
| lightgbm | 4.6.0 | Gradient boosting (model utama) |
| scikit-learn | 1.8.0 | Metrik evaluasi |
| optuna | — | Hyperparameter tuning Bayesian |
| prophet | — | Facebook/Meta time series model |
| holidays | — | Flag hari libur nasional Brasil |

---

## Struktur Direktori

```
demand_forecasting/
├── demand_forecasting.ipynb   # Notebook utama (54 sel, 12 section)
├── notebook.txt               # Catatan sesi & dokumentasi hasil lengkap
├── README.md                  # Dokumen ini
├── dataset/                   # CSV Olist (tidak di-commit ke git)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   └── product_category_name_translation.csv
└── venv/                      # Virtual environment (tidak di-commit)
```
