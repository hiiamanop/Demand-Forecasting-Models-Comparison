# Churn-Aware Demand Forecasting 📈

**Integrasi Prediksi Customer Churn sebagai Fitur Eksogen dalam Peramalan Permintaan pada Platform E-Commerce menggunakan Explainable AI (SHAP)**

Proyek ini merupakan implementasi penelitian tingkat pascasarjana (S2) yang bertujuan untuk membuktikan bahwa pemahaman terhadap risiko *churn* pelanggan dapat meningkatkan akurasi peramalan permintaan (*demand forecasting*) pada platform E-Commerce.

## 1. Ikhtisar Penelitian

Penelitian ini menggunakan dataset **Olist (Brazilian E-Commerce)** untuk menjawab tantangan ketidakpastian permintaan. Dengan mengintegrasikan probabilitas churn individu yang diolah menjadi indeks risiko bulanan, model peramalan dapat menangkap sinyal penurunan permintaan lebih dini dibandingkan model deret waktu (*time-series*) tradisional.

### Metodologi (5 Tahap):

1. **EDA & RFM Engineering:** Penggabungan data relasional dan ekstraksi perilaku pelanggan.
2. **Churn Prediction:** Klasifikasi biner menggunakan XGBoost dan CatBoost (dengan SMOTE).
3. **Explainable AI (SHAP):** Identifikasi pendorong utama churn (Logistik, Review, dll).
4. **Churn-Aware Forecasting:** Implementasi model Prophet dan XGBoost Regressor dengan fitur eksogen risiko churn.
5. **Prescriptive Analytics:** Transformasi hasil AI menjadi rekomendasi bisnis strategis.

## 2. Struktur Proyek

- `Churn-Aware Demand Forecasting.ipynb`: Notebook utama berisi seluruh alur riset.
- `dataset/`: (Direktori) Berisi file CSV asli dari Olist.
- `research_dashboard_final.png`: Visualisasi ringkasan hasil penelitian.
- `df_rfm_final.csv`: Ekspor data RFM akhir untuk analisis statistik lanjut.

## 3. Cara Penggunaan

1. Pastikan dataset Olist sudah berada di folder `./dataset`.
2. Buka VSCode dan pilih kernel `tesis_env`.
3. Jalankan seluruh *cell* pada notebook `Churn-Aware Demand Forecasting.ipynb`.
4. Hasil analisis strategi bisnis dapat dilihat di bagian akhir (Tahap 6).
