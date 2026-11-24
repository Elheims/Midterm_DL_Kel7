# Kontribusi

Berkas ini baru memuat bagian Task 3. Bagian Task 1 dan Task 2 diisi anggota yang mengerjakannya (tanda `TODO`).

## Pembagian peran

| Anggota | Bagian | PR |
|---|---|---|
| Sandy | Task 3 (clustering multimodal) | TODO: tautan PR |
| TODO | Task 1 (klasifikasi) | TODO |
| TODO | Task 2 (regresi RUL) | TODO |

## Task 3: apa yang dikerjakan

- Satu notebook `kode/task3_clustering/task3_clustering.ipynb`: eksplorasi distorsi citra dan spektrum, SimCLR terpisah untuk citra dan spektrum, fusi dua modalitas (`concat` dan `joint`), KMeans / GMM / Agglomerative / pembelahan graf kNN / DEC, pemilihan K dengan beberapa kriteria, UMAP dan t-SNE, serta analisis pasangan cluster yang sulit dipisah.
- Keluaran: `cluster.csv` (36.000 baris, 7 cluster), `kode/results/task3/results.json`, dan gambar di `laporan/figures/` (awalan `t3_`).
- Run penuh dijalankan Sandy di Google Colab (GPU T4, lewat ekstensi Colab di VS Code). `cluster.csv` berasal dari run itu. Di komputer lokal hanya ada smoke test kecil (`FAST = True`) untuk memeriksa kode.
- Keputusan penting: jumlah cluster dipaksa 7 (`K_OVERRIDE = 7`) walau kriteria silhouette, Davies-Bouldin, Calinski-Harabasz, dan stabilitas memilih 5. Alasannya struktur graf kNN (dua dari lima komponen terhubung terbelah menjadi dua pulau yang berbeda hanya pada satu modalitas). Ini bukan hasil pengecekan terhadap referensi tersembunyi.
- Aturan pemilihan metode akhir (`knn_cut` terendah, lalu konsistensi) dibuat setelah melihat hasil run, bukan ditetapkan sebelumnya.

## Penggunaan alat AI

- **Alat:** Claude Code (model Claude Sonnet 5.5, Anthropic), dipakai di terminal pada repo ini untuk Task 3.
- **Dikerjakan dengan bantuan AI:** pembacaan spesifikasi tugas dan dataset, penulisan seluruh kode notebook Task 3 (pra-pemrosesan, augmentasi, SimCLR, fusi, metode clustering, kriteria K, diagnostik), skrip `extract_outputs.py`, perbaikan bug (broadcasting augmentasi spektrum, detektor spike, overflow fp16 pada loss, presisi campuran untuk T4), pembacaan output run dan gambar, serta draf catatan (README Task 3 dan berkas ini).
- **Dikerjakan manusia:** menentukan cakupan dan keluaran, menjalankan notebook di Colab dan mengulangnya setelah tiap perbaikan, memutuskan K = 7, dan memutuskan apa yang masuk repo. TODO: Sandy memeriksa dan menyesuaikan pernyataan ini sebelum dikumpulkan.
- **Verifikasi:** angka di laporan diambil dari output notebook yang dijalankan di Colab, bukan dari ingatan model. Tidak ada label untuk clustering, jadi kebenaran tidak bisa diperiksa langsung. Hanya diagnostik internal (silhouette, `knn_cut`, d' per modalitas, kebocoran mikroskop) yang tersedia.
- TODO: lengkapi pernyataan penggunaan AI untuk Task 1 dan Task 2 oleh anggota masing-masing.
