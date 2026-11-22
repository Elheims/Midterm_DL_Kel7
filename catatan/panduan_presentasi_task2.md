# Panduan Presentasi & Ujian Lisan — Task 2 (Prediksi RUL)

Dokumen ini merangkum **cara menjelaskan setiap langkah Task 2 secara sederhana dan jujur** di ujian lisan (5 menit demo + 15 menit tanya jawab). Tujuannya bukan memakai istilah rumit, tapi bisa menjelaskan **apa yang kita lakukan, kenapa, dan buktinya apa**.

---

## 1. Gambaran Besar (Penjelasan 30 Detik)

- **Masalah:** Memprediksi sisa umur pakai (**RUL**, dipotong maksimal 150 siklus) pada siklus terakhir dari 300 mesin uji, berdasarkan riwayat 900 mesin latih sampai rusak.
- **Alur kerja:**
  1. Bersihkan data sensor secara **kausal** (hanya pakai data masa lalu).
  2. Buat label RUL (dipotong di 150) dan potong riwayat menjadi jendela **50 siklus** (`50 × 14`).
  3. Bandingkan 2 baseline klasik (Ridge, Boosted Trees) vs 2 model deep learning (1D-CNN, GRU) dengan split **per mesin**.
  4. **GRU ($W=50$)** menang paling stabil di semua 5 fold CV (`RMSE 19.20 ± 0.83`).
  5. Buat interval ketidakpastian 90% (ensemble + MC dropout + kalibrasi conformal) dan uji aturan kapan mesin harus diservis.
  6. Model final: rata-rata **5 GRU** yang dilatih ulang di 900 mesin, lalu dikurangi **3 siklus** supaya lebih aman terhadap penalti telat → `prediksi_rul.csv`.

---

## 2. Penjelasan Sederhana per Bagian (a – f)

### (a) Eksplorasi & Preprocessing
- **Kenapa normalisasi per rezim (KMeans $k=6$)?**
  Tiga kolom pengaturan (`setel_1..3`) membentuk 6 mode operasi yang jelas. Perpindahan mode ini menyebabkan 37–100% variasi sensor, sehingga sinyal keausan tertutup lompatan mode. Setelah di-z-score **di dalam masing-masing mode**, tren keausan langsung terlihat (contoh: korelasi `s04` dengan RUL naik dari `0.28` ke `0.71`).
- **Kenapa 7 sensor dibuang?**
  Sensor `s01, s05, s06, s10, s16, s18, s19` tidak berubah dari awal sampai mesin rusak (perubahan cuma `0.23–0.25 z` vs `1.22–2.66 z` pada 14 sensor lainnya). Dibuang supaya model tidak belajar noise.
- **Kenapa pakai forward-fill untuk data hilang (NaN)?**
  Sekitar 2% bacaan per sensor hilang (35% baris punya minimal 1 NaN, jadi tidak mungkin menghapus baris). Kita pakai nilai siklus sebelumnya (forward-fill) karena **kausal** — tidak boleh interpolasi dari masa depan karena di data uji kita tidak punya siklus setelah siklus terakhir. Kalau NaN terjadi di paling awal sebelum ada bacaan, diisi rata-rata kondisi mesin sehat dari data latih.
  - *Catatan jujur bila ditanya:* ~98% data hilang hanya 1 siklus sendirian, tapi ada juga kasus langka sensor putus beruntun 5–21 siklus (terbesar 18 siklus di `s04`, 21 di `s11`). Forward-fill tetap aman dan kausal untuk kasus ini.
- **Kenapa filter spike (Hampel kausal) dilakukan setelah z-score?**
  Kalau sebelum z-score, lompatan rezim operasi akan dikira spike. Setelah z-score, spike kelihatan jelas (~0.2% bacaan, muncul di atas/bawah median 10 siklus sebelumnya lebih dari `5 × noise sd`). Semua statistik (`mean`, `std`, `noise_sd`, sensor terpilih) **hanya dihitung dari mesin latih**.
- **Kenapa ada 2 mode kerusakan?**
  342 mesin rusak di satu kelompok sensor, 558 mesin di kelompok lain, bahkan `s11` bergerak berlawanan arah. Artinya model harus menggabungkan banyak sensor sekaligus, tidak bisa mengandalkan tren linier tunggal.

### (b) Label RUL (Cap 150)
- Di data latih, mesin berjalan sampai rusak di siklus $T$, jadi $\text{RUL}(t) = \min(150, T - t)$.
- **Kenapa dipotong 150?** Di awal pemakaian mesin masih sehat sehingga sensor datar (mesin sisa 300 siklus terlihat sama persis dengan sisa 200 siklus). Ini juga sesuai aturan soal (47% baris latih bernilai 150).

### (c) Sliding Windows ($W = 50$)
- Setiap sampel mengambil $W$ siklus terakhir sampai siklus $t$, dengan target $\text{RUL}(t)$.
- **Kenapa aman dari kebocoran masa depan?** Jendela hanya melihat siklus $\le t$ dari satu mesin saja — sama persis dengan kondisi data uji.
- **Bagaimana riwayat yang lebih pendek dari $W$?** Di-pad di bagian depan dengan mengulang bacaan pertama. Karena mesin uji terpendek panjangnya **50 siklus**, untuk $W \le 50$ tidak ada mesin uji yang perlu padding.

### (d) Model & Kenapa GRU Menang
- **Protokol adil:** Semua model (Ridge, Boosted Trees, 1D-CNN, GRU) dievaluasi di **180 mesin held-out yang sama** (5 potongan acak per mesin = 900 evaluasi, panjang riwayatnya mirip data uji: median 160 vs 162 siklus) dan memakai jendela yang sama ($W=50$).
- **Sanity check:** Semua model deep learning diuji dulu untuk menghafal 1 batch berisi 64 jendela → RMSE turun di bawah `0.1` siklus (membuktikan arsitektur & gradien tidak ada bug).
- **Kenapa 1D-CNN cepat overfit (puncak di epoch 1–2)?**
  Lapisan `Flatten → Linear(3200, 64)` punya ~205 ribu parameter (dari total 238 ribu), jauh terlalu besar untuk 648 mesin latih sehingga cepat menghafal mesin latih. Mengganti ke global average pooling merusak akurasi karena *urutan waktu* (siklus terbaru) jadi hilang.
- **Kenapa GRU lebih baik?**
  GRU membaca urutan siklus satu per satu, parameternya 5× lebih kecil (45 ribu), lebih stabil antar-seed (`20.70 ± 0.10` vs CNN `21.45 ± 0.63`), dan menang di **semua 5 fold CV** (`19.20 ± 0.83`).
- **Kenapa error paling besar di RUL 51–100 (analisis error)?**
  Mesin mulai menunjukkan tanda aus di waktu yang berbeda-beda (kuartil: 59–102 siklus sebelum rusak). Mesin yang telat menunjukkan tanda aus akan terlihat masih sehat di RUL 50 sehingga diprediksi terlalu tinggi; mesin yang cepat aus diprediksi terlalu rendah. Karena sensor hanya menunjukkan kondisi *sekarang*, sebagian error di rentang tengah ini memang tidak bisa dihilangkan oleh model berbasis jendela.
  - *Catatan bila ditanya tentang GRU:* Grafik `t2_error_analysis` dibuat dari 1D-CNN, tapi Tabel 2 menunjukkan GRU punya pola persis sama di RUL 51–100 (RMSE `29.6`, bias `+12.6`).

### (e) Validasi per Mesin & 5-Fold CV
- **Kenapa split per mesin, bukan per baris?**
  Baris berdekatan dari satu mesin hampir identik. Kalau dipecah per baris, model tinggal menghafal identitas mesin sehingga skor tampak bagus padahal bohong.
- **Kenapa 5-fold CV RMSE (19.20) lebih bagus dari split utama (20.70)?**
  Kebetulan Fold 1 (= split utama 180 mesin) adalah fold yang paling sulit bagi *semua* model (Ridge, Trees, CNN, dan GRU semuanya paling buruk di Fold 1). Jadi split utama kita bukan hasil pilih-pilih split yang mudah.
- **Hasil per rentang RUL:** Dekat kerusakan (`0–25` siklus, paling penting untuk servis), GRU paling akurat (`RMSE 8.5`). Rentang `26–100` menyumbang 91–95% dari total skor asimetris.

### (f) Ketidakpastian & Kapan Servis Mesin
- **Interval 90%:** Kita gabungkan 5 GRU × 20 kali prediksi dengan dropout tetap menyala (ensemble + MC dropout).
- **Kenapa harus dikalibrasi (split-conformal)?**
  Interval mentahnya terlalu sempit (hanya mencakup 63% nilai asli, bukan 90%) karena ensemble/dropout hanya mengukur keraguan *model*, bukan ketidakpastian *mesin* (misalnya mesin yang telat aus). Dengan split-conformal pada 72 mesin early-stop, lebar interval diskalakan (`×3.54`) sehingga cakupan menjadi **90%** di 180 mesin held-out (rata-rata lebar 50 siklus; sempit `31` di dekat rusak, lebar `58` di paruh umur).
  - *Bila ditanya kenapa cakupan di RUL 51–100 cuma 74%:* Split-conformal menjamin cakupan 90% secara *keseluruhan (marginal)*, bukan di setiap sub-rentang; di 51–100 banyak mesin telat aus sehingga error besar sementara model masih yakin.
- **Aturan servis mesin (temuan jujur):**
  Kita simulasikan 180 mesin held-out siklus demi siklus dengan asumsi butuh 10 siklus untuk menyiapkan servis.
  - Aturan titik ("servis jika prediksi $\le 34$"): tidak ada mesin telat, rata-rata sisa umur terbuang **36.4 siklus**.
  - Aturan hati-hati ("servis jika batas bawah interval 90% $\le 14$"): tidak ada mesin telat, terbuang **38.1 siklus**.
  - **Kesimpulan jujur:** Interval 90% **tidak** membuat aturan servis lebih hemat dibanding prediksi titik, karena di sekitar ambang servis lebar interval hampir sama untuk semua mesin (~40 siklus) dan korelasinya dengan error kecil (`0.12`). Jadi rekomendasi praktis kita sederhana: **servis saat prediksi RUL turun ke sekitar 35 siklus**.

---

## 3. Penjelasan Sederhana untuk Bagian yang Paling Sering Ditanya

### Pertanyaan A: "Kenapa model final dikurangi 3 siklus (shift = 3)? Bagaimana milih angka 3?"
**Versi sederhana (cukup 4 kalimat):**
1. Skor asimetris menghukum prediksi **telat** (terlalu tinggi) jauh lebih berat daripada prediksi terlalu rendah, jadi mengurangi prediksi sedikit akan menurunkan skor asimetris dengan sedikit kenaikan RMSE.
2. Kalau cuma mengejar skor asimetris, pengurangan terbaik adalah **13 siklus**, tapi itu membuat RMSE melonjak dari `20.5` ke `24.1` (+3.6 siklus).
3. Kita uji beberapa nilai shift (`0–13`) di data validasi (simulasi 2.000 set berisi 300 mesin), lalu membandingkan penurunan skor vs kenaikan RMSE dalam satuan variasi alami masing-masing metrik (bobot sama 50/50 karena di soal RMSE = 3 poin dan skor = 3 poin).
4. **Shift = 3** adalah titik paling untung: skor asimetris turun **22%**, sedangkan RMSE cuma naik **0.10 siklus** (hampir tidak terasa). Data uji asli sama sekali tidak dipakai untuk memilih angka ini.

### Pertanyaan B: "Kenapa dilatih pakai MSE, bukan langsung pakai loss skor asimetris?"
- Loss asimetris ($\exp(d/10) - 1$) eksponensial: di awal training, satu error besar bisa bikin gradien meledak dan merusak RMSE (padahal RMSE juga bernilai 3 poin, sama dengan skor asimetris).
- Melatih dengan MSE stabil dan mengoptimalkan RMSE, lalu pengurangan kecil `-3` siklus di akhir sudah memangkas 22% skor asimetris tanpa merusak RMSE.

### Pertanyaan C: "Di `uncertainty.json`, rata-rata `ensemble+mc` (RMSE 20.33) sedikit lebih baik dari `ensemble` deterministik (20.49). Kenapa `prediksi_rul.csv` pakai yang deterministik?"
- Bedanya sangat kecil (~0.15 siklus RMSE), sedangkan ensemble deterministik 5 GRU (dropout off) jauh lebih cepat, tidak bergantung pada random dropout mask saat prediksi, dan 100% deterministik untuk demo reproduksi lisan.

### Pertanyaan D: "Kenapa tidak memasukkan nomor siklus (`siklus`) atau mode operasi sebagai fitur?"
- Mode operasi sudah tidak membawa sinyal keausan setelah normalisasi per rezim (dan berganti acak di 83% siklus).
- Nomor siklus (`siklus`) memang legal dipakai, tapi kita sengaja mengandalkan sinyal sensor murni agar model belajar dari **kondisi fisik mesin**, bukan menghafal rata-rata umur mesin latih.

### Pertanyaan E: "Kenapa `final_weights/` dan 6 checkpoint validasi disimpan di GitHub?"
- Ukurannya kecil (total ~3.3 MB), supaya siapapun bisa langsung menjalankan demo lisan (`python -m task2_rul.demo`) dalam ~15 detik tanpa perlu melatih ulang 14 menit.

---

## 4. Checklist Demo Lisan 5 Menit

Jalankan dari folder `kode/` (dengan dataset di `../data/raw/`):

```bash
python -m task2_rul.demo
```

Yang diperiksa otomatis (tanpa training, ~12–15 detik):
1. Membangun ulang split validasi 180 mesin held-out.
2. Memuat 3 GRU + 3 CNN validasi yang tersimpan → menghitung ulang 11 angka utama di laporan dan mencocokkannya (`Match: yes`).
3. Memuat 5 GRU final → menghitung ulang 300 prediksi uji dan mencocokkannya dengan `prediksi_rul.csv` (`identical: yes`).

> **Tips sebelum ujian lisan:** Tes juga sekali dengan CPU-only (`CUDA_VISIBLE_DEVICES=""`) di laptop anggota lain supaya yakin tidak ada error library/device kalau dosen menunjuk anggota yang tidak pakai GPU.
