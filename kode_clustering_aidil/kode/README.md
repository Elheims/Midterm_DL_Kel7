# Tugas UTS Deep Learning - Task 1: Multimodal Defect Classification

Repositori ini memuat kode lengkap, pipeline eksperimen, dan model untuk **Task 1: Multimodal Classification (Product Defects)**.

---

## 1. Struktur Direktori
```text
UTS/
├── data_uts_dl/
│   ├── klasifikasi_train.npz
│   ├── klasifikasi_train_meta.csv
│   ├── klasifikasi_test.npz
│   └── klasifikasi_test_meta.csv
├── checkpoints/              # Model weights (.pt)
├── figures/                  # Visualisasi EDA & learning curves
├── results/                  # Ringkasan metrik eksperimen (JSON & CSV)
├── kode/
│   ├── dataset.py            # Dataset multimodal, sensor stats, & modality dropout
│   ├── models.py             # Image CNN, Signal 1D-CNN, Late Fusion, Cross-Attention
│   ├── train_eval.py         # Training loop, evaluation, & sanity check
│   ├── run_experiments.py   # One-command runner untuk semua eksperimen & prediksi
│   └── README.md             # Dokumentasi reproduktibilitas
├── generate_eda_figures.py   # Generator grafik eksplorasi data
└── prediksi_klasifikasi.csv  # Output prediksi akhir untuk test set
```

---

## 2. Persyaratan Lingkungan (Environment)
- Python 3.10+ (atau Python 3.15)
- PyTorch >= 2.0
- torchvision
- pandas, numpy, scikit-learn, matplotlib, seaborn, tqdm

Instalasi dependensi:
```bash
pip install torch torchvision pandas scikit-learn seaborn matplotlib tqdm
```

---

## 3. Menjalankan Kode (Reproducibility: One-Command)

### A. Menghasilkan Visualisasi EDA
```bash
python generate_eda_figures.py
```
Menghasilkan 4 grafik di folder `figures/`:
1. `eda_class_distribution.png`: Distribusi 7 kelas cacat pada train set (imbalanced).
2. `eda_domain_shift.png`: Perbandingan proporsi lini produksi (Lini A vs B) antara train dan test.
3. `eda_multimodal_samples.png`: Contoh sampel citra kamera + 3 sinyal sensor untuk setiap kelas.
4. `eda_missing_sensor.png`: Rasio kelengkapan vs kehilangan rekaman sensor.

### B. Sanity Check (Overfitting 1 Batch)
```bash
python kode/train_eval.py --sanity-check
```
Memverifikasi kapasitas model dan implementasi loss function dengan melatih 32 sampel hingga loss < 0.1 dan akurasi 100%.

### C. Menjalankan Seluruh Eksperimen & Menghasilkan Prediksi Akhir (One Command)
```bash
python kode/run_experiments.py
```
Perintah ini akan secara otomatis:
1. Melatih dan mengevaluasi **Image-Only CNN** dan **Sensor-Only 1D-CNN** (Poin c).
2. Melatih dan membandingkan **Late Fusion** vs **Cross-Attention Fusion** (Poin b).
3. Menguji ketahanan terhadap label noise dengan **Label Smoothing** (Poin d).
4. Menguji mitigasi domain shift dengan **Per-Line Normalization** (Poin e).
5. Menguji ketahanan terhadap **Missing Sensor** (Poin f).
6. Melatih model final pada **3 Random Seeds** (42, 123, 999) dan menghitung mean ± std (Poin g).
7. Menghasilkan file submission **`prediksi_klasifikasi.csv`** dengan ensemble probabilitas 3 seed.

---

## 4. Format Output Prediksi
File `prediksi_klasifikasi.csv` memiliki format persis sesuai spesifikasi:
- Kolom: `id_unit,kelas`
- Nilai kelas: salah satu dari `normal`, `gores`, `noda`, `retak`, `gelembung`, `las_dingin`, `kontaminasi`.
