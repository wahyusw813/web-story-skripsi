# Memprediksi Saham Energi di Tengah Transisi

**Web story interaktif untuk skripsi** *Prediksi Harga Saham Sektor Energi dalam Transisi Energi Baru Terbarukan Menggunakan Temporal Fusion Transformer Berbasis Technical Indicator dan Sentimen Berita*.

[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-003060)](https://wahyusw813.github.io/web-story-skripsi/)
![D3.js](https://img.shields.io/badge/D3.js-v7-F9A03C)
![Skripsi](https://img.shields.io/badge/skripsi-Politeknik%20Statistika%20STIS%202026-FF914D)

Web story ini menyajikan hasil penelitian tentang prediksi harga penutupan saham PT Alamtri Resources Indonesia Tbk (ADRO) dalam bentuk narasi interaktif, dari latar belakang hingga kesimpulan. Seluruh grafik prediksi dihitung langsung dari model terbaik penelitian dan diverifikasi terhadap metadata model.

**Demo daring:** https://wahyusw813.github.io/web-story-skripsi/

---

## Daftar isi

- [Fitur](#fitur)
- [Isi repositori](#isi-repositori)
- [Ringkasan penelitian](#ringkasan-penelitian)
- [Hasil utama](#hasil-utama)
- [Verifikasi model](#verifikasi-model)
- [Alur kerja](#alur-kerja)
- [Menjalankan secara lokal](#menjalankan-secara-lokal)
- [Keterbatasan dan penafian](#keterbatasan-dan-penafian)
- [Sitasi](#sitasi)
- [Penulis](#penulis)

---

## Fitur

- **Scrollytelling tujuh bagian** yang mengikuti struktur skripsi: latar belakang, rumusan masalah dan tujuan, analisis sentimen, pemodelan TFT, validasi walk-forward, hasil prediksi, serta kesimpulan dan saran.
- **Penjelajah prediksi**: 2.404 prediksi harga penutupan satu hari ke depan. Pengguna dapat memilih rentang waktu, dan RMSE, MAPE, MASE, serta akurasi arah dihitung otomatis. Zona latih, validasi, dan uji ditandai pada grafik.
- **Bedah satu prediksi**: bobot attention model untuk 30 hari masukan dan bobot Variable Selection Network untuk setiap variabel.
- **Matriks 12 konfigurasi** yang dapat diklik dan **diagram walk-forward 10 fold**.
- **Satu file HTML mandiri**: tanpa server dan tanpa basis data, responsif untuk desktop dan ponsel, serta mendukung mode terang dan gelap.

## Isi repositori

| File | Keterangan |
|---|---|
| [`index.html`](index.html) | Web story dalam satu file: HTML, CSS, JavaScript (D3.js v7), dan seluruh data prediksi tertanam di dalamnya. File inilah yang ditayangkan oleh GitHub Pages. |
| [`prediksi_model_ckpt.csv`](prediksi_model_ckpt.csv) | 2.404 prediksi harga penutupan satu hari ke depan dari model terbaik, dengan kolom `time_idx`, `tanggal`, `aktual`, `prediksi`, dan `zona` (`latih`, `validasi`, `uji`, `setelah_uji`). |
| `README.md` | Dokumen ini. |

Kode pelatihan model dan skrip pembangun halaman tidak disertakan dalam repositori ini.

## Ringkasan penelitian

| Komponen | Keterangan |
|---|---|
| Objek | Harga penutupan harian ADRO.JK, 2016–2025 (2.464 hari perdagangan) |
| Berita | 303.257 artikel detikFinance, disaring menjadi 36.960 artikel relevan melalui validasi 18 kata kunci |
| Sentimen | Perbandingan dua alur kerja: VADER (teks terjemahan) dan FinBERT-Indonesia (teks sumber), enam uji keterkaitan |
| Model | Temporal Fusion Transformer, desain faktorial 3 aktivasi × 2 inisialisasi × 2 strategi pruning = 12 konfigurasi |
| Optimasi | Optuna (TPE), 240 trial per konfigurasi, objektif rata-rata RMSE validasi |
| Validasi | Walk-forward jendela ekspansif, 10 fold, horizon satu hari |
| Evaluasi | RMSE, MAPE, sMAPE, MASE, akurasi arah, uji residu, dan kontribusi variabel |

## Hasil utama

| Aspek | Hasil |
|---|---|
| Alur sentimen terpilih | VADER, unggul pada 4 dari 6 pengujian |
| Konfigurasi terbaik | K5: ReLU + Kaiming, tanpa pruning |
| RMSE / MAPE uji (rata-rata 10 fold) | 102,70 / 3,15% |
| Fold terbaik (model pada web story) | Fold 6: RMSE 44,77; MAPE 1,48% |
| Kontribusi variabel | Harga penutupan 48,10%; sentimen 8,56% |
| Akurasi arah | Rata-rata 47,97%, setara tebakan acak |

Aktivasi dan inisialisasi terbukti berinteraksi: Kaiming selalu lebih baik, tetapi selisihnya hanya 1,29 poin RMSE pada ELU dan 21,11 poin pada GELU. Strategi pruning tidak berpengaruh sistematis terhadap akurasi.

## Verifikasi model

Prediksi pada web story dihitung ulang dari checkpoint model terbaik (K5, fold 6) dengan implementasi NumPy, lalu dicocokkan dengan metadata model pada data uji fold 6:

| Metrik uji fold 6 | Metadata model | Hasil web story |
|---|---|---|
| RMSE | 44,7711 | 44,8036 |
| MAPE (%) | 1,4825 | 1,4842 |
| sMAPE (%) | 1,4723 | 1,4763 |
| MASE | 0,9911 | 0,9917 |

Selisih di bawah 0,1% berasal dari perbedaan presisi float32 (PyTorch) dan float64 (NumPy).

## Alur kerja

```mermaid
flowchart LR
    A[Data harga, indikator teknikal, dan sentimen] --> C[Inferensi offline model TFT K5]
    B[Checkpoint model terbaik] --> C
    C --> D[Prediksi, attention, dan bobot VSN]
    C --> G[prediksi_model_ckpt.csv]
    D --> E[index.html satu file]
    E --> F[GitHub Pages]
```

Inferensi model dijalankan sekali secara offline. Hasilnya ditanam di dalam `index.html`, sehingga situs dapat ditayangkan di layanan statis seperti GitHub Pages tanpa server.

## Menjalankan secara lokal

Unduh atau clone repositori, lalu buka `index.html` di peramban. Koneksi internet diperlukan untuk memuat pustaka D3.js dan font.

```bash
git clone https://github.com/wahyusw813/web-story-skripsi.git
cd web-story-skripsi
python -m http.server 8000   # buka http://localhost:8000
```

Data prediksi juga dapat dianalisis langsung, misalnya dengan pandas:

```python
import pandas as pd

df = pd.read_csv("prediksi_model_ckpt.csv")
uji = df[df["zona"] == "uji"]   # 20 prediksi data uji fold 6
print(uji[["tanggal", "aktual", "prediksi"]])
```

Situs daring diperbarui otomatis oleh GitHub Pages setiap kali `index.html` di branch `main` diganti.

## Keterbatasan dan penafian

- **Bukan rekomendasi investasi.** Web story ini adalah luaran penelitian akademik.
- Model memprediksi **level harga satu hari ke depan**. Akurasi arahnya setara tebakan acak, dan MASE rata-rata 10 fold (2,4455) belum mengungguli tebakan naif.
- Model fold 6 dilatih dengan data hingga Januari 2024. Prediksi setelah periode uji dibuat tanpa pelatihan ulang, sehingga galatnya membesar, terutama setelah penyesuaian harga akibat aksi korporasi pada November 2024.
- Bobot attention dan Variable Selection Network menunjukkan alokasi perhatian model, bukan hubungan sebab-akibat.

## Sitasi

Jika merujuk proyek ini, mohon kutip skripsinya:

> Widodo, W. S. (2026). *Prediksi Harga Saham Sektor Energi dalam Transisi Energi Baru Terbarukan Menggunakan Temporal Fusion Transformer Berbasis Technical Indicator dan Sentimen Berita* [Skripsi]. Politeknik Statistika STIS.

```bibtex
@thesis{widodo2026prediksi,
  author      = {Widodo, Wahyu Satrio},
  title       = {Prediksi Harga Saham Sektor Energi dalam Transisi Energi Baru Terbarukan
                 Menggunakan Temporal Fusion Transformer Berbasis Technical Indicator
                 dan Sentimen Berita},
  type        = {Skripsi},
  institution = {Politeknik Statistika STIS},
  year        = {2026}
}
```

## Penulis

**Wahyu Satrio Widodo** (NIM 222212911)
D-IV Komputasi Statistik, peminatan Sains Data, Politeknik Statistika STIS

Dosen pembimbing: Dr. Robert Kurniawan, SST, M.Si.

Data harga bersumber dari Yahoo Finance dan skor sentimen diturunkan dari artikel detikFinance; hak atas sumber data tetap pada pemiliknya masing-masing. Pemodelan memakai [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting) dan visualisasi memakai [D3.js](https://d3js.org/).

© 2026 Wahyu Satrio Widodo. Hak cipta dilindungi. Untuk penggunaan ulang kode atau data, silakan hubungi penulis.
