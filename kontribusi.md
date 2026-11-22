# Kontribusi

## Peran anggota

| Anggota | Peran | Pull request |
|---|---|---|
| Elheims | Task 2 (RUL regression): EDA & preprocessing, pelabelan, sliding windows, baseline, 1D-CNN, GRU, validasi & 5-fold CV, uncertainty & aturan servis, model final (`prediksi_rul.csv`), bagian laporan Task 2, dan skrip demo lisan (`demo.py`) | [#1](https://github.com/Elheims/Midterm_DL_Kel7/pull/1), [#2](https://github.com/Elheims/Midterm_DL_Kel7/pull/2), [#3](https://github.com/Elheims/Midterm_DL_Kel7/pull/3), [#4](https://github.com/Elheims/Midterm_DL_Kel7/pull/4), [#5](https://github.com/Elheims/Midterm_DL_Kel7/pull/5), [#6](https://github.com/Elheims/Midterm_DL_Kel7/pull/6) |
| sandyxd18 | Task 1 (classification) / Task 3 (clustering) — TBD sesuai pembagian | TBD |
| aidilfadhli | Task 1 (classification) / Task 3 (clustering) — TBD sesuai pembagian | TBD |

> Catatan workflow GitHub: repositori kelompok berada di akun `Elheims`, dan GitHub tidak mengizinkan pemilik repositori mem-fork repositorinya sendiri. Karena itu `Elheims` bekerja melalui branch dan membuka pull request ke `main`, sedangkan anggota lain bekerja melalui fork masing-masing lalu membuka pull request ke `main`.

## Penggunaan AI tools

> Penggunaan AI harus disetujui seluruh anggota kelompok. Setiap penggunaan mencantumkan: tool, bagian yang dibantu, dan cara verifikasi.

| Tool | Bagian yang dibantu | Cara verifikasi |
|---|---|---|
| Claude Code (Anthropic) | **Task 2 (RUL):** penulisan kode pipeline (`eda.py`, `data.py`, `labels.py`, `windows.py`, `evaluate.py`, `baseline.py`, `models.py`, `train_nn.py`, `deep.py`, `error_analysis.py`, `validation.py`, `uncertainty.py`, `maintenance.py`, `final.py`, `predict.py`, `demo.py`), pembuatan grafik, dan penyusunan draf bagian laporan Task 2 (`task2_rul.tex`, `lampiran_task2.tex`). | Menjalankan uji kausalitas & bebas NaN (`check_preprocess.py`), uji label (`labels.py`), uji kebocoran jendela termasuk kasus bocor buatan (`windows.py`), uji overfit satu batch pada semua model deep learning, reproduksi penuh `python run_all.py --task 2` (dua kali run menghasilkan angka & `prediksi_rul.csv` identik), pengecekan angka laporan terhadap JSON via `python -m task2_rul.demo`, serta pemeriksaan visual semua grafik dan log keputusan di `catatan/keputusan.md`. |
| Antigravity / Gemini (Google) | **Review Task 2:** audit kesesuaian Task 2 terhadap rubrik UTS, pemeriksaan format `prediksi_rul.csv`, pembaruan `kontribusi.md` (bagian Task 2), dan penyusunan panduan penjelasan presentasi lisan (`catatan/panduan_presentasi_task2.md`). | Seluruh temuan dan angka diverifikasi langsung terhadap file kode, file JSON hasil di `kode/results/task2/`, dan `prediksi_rul.csv`. |
| *(Tambahkan AI tools untuk Task 1 & Task 3 jika digunakan)* | TBD | TBD |

## Persetujuan anggota

Centang sebelum merge final ke `main` (menandakan seluruh anggota menyetujui pembagian peran dan catatan penggunaan AI di atas):

- [x] Elheims
- [ ] sandyxd18
- [ ] aidilfadhli
