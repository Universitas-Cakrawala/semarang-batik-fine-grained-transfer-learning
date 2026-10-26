# Evaluasi From-Scratch CNN dan Fine-Tuned EfficientNet-B0 untuk Fine-Grained Classification Motif Batik Semarang

**Kelompok 5, Track A** · Tita Noviana, Titanio Yudista, Suci Fransisca Sisilia R., Fajar Dwiharjo

> **Status draf.** Bagian 1–3 dan 8 sudah ditulis dari artefak yang ada.
> Setiap bagian bertanda **[HASIL GPU]** diisi setelah `batik matrix`,
> `aggregate`, `robustness`, dan `errors` selesai. Seluruh angka harus disalin
> dari `artifacts/metrics/*` dan tidak boleh diketik ulang dari ingatan. Target
> panjang akhir: 8–12 halaman.

---

## 1. Problem statement dan kesesuaian deep learning

**Keputusan yang didukung.** Model mengklasifikasikan foto kain ke salah satu dari
10 motif batik Semarang: Asem Arang, Asem Sinom, Asem Warak, Blekok, Blekok Warak,
Gambang Semarangan, Kembang Sepatu, Semarangan, Tugu Muda, dan Warak Beras Utah.
Tujuannya membantu kurasi dan pengarsipan. Output model adalah **saran top-3 untuk
diverifikasi kurator**, bukan katalogisasi otomatis.

**Pengguna dan biaya kesalahan.** Pengguna adalah kurator, peneliti tekstil, dan
pengelola platform kriya. False positive mencemari arsip dengan label yang salah.
False negative membuat motif asli terlewat dan menambah beban verifikasi manual.
Karena kedua jenis kesalahan sama-sama penting dan kelas hampir seimbang, metrik
utama yang dipakai adalah Macro F1.

**Mengapa deep learning.** Foto kain mengalami lipatan non-rigid, perubahan skala,
rotasi, dan pencahayaan. Isen-isen antar motif juga mirip. Fitur manual seperti
SIFT/HOG, atau fitur warna/tekstur pada paper asal dataset (Winarno dkk., 2024),
harus dirancang ulang untuk setiap variasi tersebut. Sebaliknya, CNN mempelajari
fitur hierarkis langsung dari data. Klaim ini **diuji**, bukan diasumsikan:
laporan ini membandingkan CNN dengan baseline non-deep pada split yang sama
(Bagian 4).

**Ruang lingkup.** Klasifikasi closed-set dengan 10 kelas, tanpa deployment.

## 2. Data

| Item | Nilai | Sumber artefak |
|---|---|---|
| Dataset | Semarang Batik Dataset, Kaggle `edywinarno/semarang-batik-dataset`, versi 2 (diperbarui 2023-10-16) | `data/manifests/dataset_snapshot.json` |
| Lisensi | CC BY-NC-SA 4.0 (proposal keliru menulis “Academic Research Use”) | `DATA_LICENSE.md` |
| Sitasi | Winarno, Septiarini, Hadikurniawati & Hamdani (2024), *ACM JOCCH* 17(2), doi:10.1145/3631131 | Metadata Kaggle |
| Arsip | 12.922.229.647 byte, SHA-256 `65710841…4fba93c` | `dataset_snapshot.json` |
| File | 3.020 JPEG, iPhone 6s, mayoritas 3024×4032; 650 file (21,7%) memiliki rotasi EXIF | `audit_summary.json` |
| Setelah audit | **2.993** gambar: 0 corrupt dan 27 exact duplicate dikeluarkan | `exclusions.csv` |
| Class balance | 295–308 per kelas, rasio imbalance 1,04 | `audit_summary.json`, `figures/class_distribution.png` |
| Split | 2.090 / 453 / 450 (69,8 / 15,1 / 15,0%), seed 42 | `split.csv`, `split_meta.json` (SHA-256 `32a0d9e2…`) |

**Preprocessing.** Setiap foto dirotasi sesuai EXIF, diubah ke RGB, dan di-resize
dengan LANCZOS hingga shorter side 320 px. Cache ini deterministik dan tidak
bergantung pada label. Input model adalah 224×224: untuk evaluasi, Resize 256 lalu
CenterCrop 224, kemudian normalisasi ImageNet.

**Leakage check eksplisit.** Audit menemukan dua sifat snapshot yang menentukan
desain split:

1. **Setiap jepretan disimpan sebagai sekitar 4 file hasil rotasi/flip.** Rata-rata
   ada 4 file per detik capture EXIF, maksimum 12. pHash biasa tidak mendeteksi
   pasangan ini (jarak sekitar 30). pHash invarian terhadap 8 transformasi dihedral
   mendeteksinya (jarak 0–4), sedangkan pasangan dari jepretan berbeda berjarak ≥ 18.
2. **Setiap kelas berasal dari satu lembar kain** yang difoto dalam satu sesi
   selama 5–18 menit.

Split dibuat per **grup foto sumber**. Dua gambar dalam kelas yang sama masuk grup
yang sama jika memenuhi salah satu syarat berikut:
- detik capture EXIF sama;
- jarak pHash dihedral ≤ 8;
- cosine embedding ImageNet yang dipusatkan ≥ 0,95.

Threshold embedding dikalibrasi. Pada 0,90 grup mulai berantai hingga 156 gambar,
sedangkan pada 0,95 hanya ada 129 edge tambahan, dan semuanya diverifikasi visual
sebagai foto ulang dari view yang sama.

Hasilnya 668 grup, dengan grup terbesar 28 gambar. Grup dibagi dengan
StratifiedGroupKFold (20 fold: 14 train, 3 val, 3 test).

Verifikasi (`split_meta.json`, `cross_split_nearest_train.csv`):
- tidak ada grup yang lintas split;
- jarak pHash dihedral minimum dari val/test ke train adalah 10;
- tidak ada detik capture yang sama antara val/test dan train;
- cosine maksimum ke train adalah 0,949.

Sebagai pembanding, split acak biasa seperti di proposal akan menempatkan salinan
rotasi dari **98,9%** gambar test (444 dari 449) ke dalam train
(`dataset_diagnostics.json`).

**Shortcut warna.** Karena satu kelas sama dengan satu kain, warna dan kecerahan
saja sudah memisahkan kelas. Tabel berikut menunjukkan baseline non-deep
(logistic regression; C dipilih pada validation):

| Fitur (non-deep) | Input | Test Macro F1 |
|---|---|---:|
| Histogram warna HSV 16×8×8 | RGB | 100,00% |
| Histogram intensitas 32 bin | RGB | 99,09% |
| Histogram LBP 8-neighbour (tekstur) | RGB | 96,72% |
| Histogram intensitas | grayscale + equalization | 55,74% |
| Histogram LBP | grayscale + equalization | 96,26% |

Karena itu, selain protokol RGB sesuai proposal, kami menjalankan **protokol
grayscale + histogram equalization** (G1g–G3g). Protokol ini menghilangkan shortcut
warna dan kecerahan global, sementara informasi tekstur motif tetap ada.

## 3. Baseline dan pendekatan yang diusulkan

| Group | Model | Parameter | Init | Input | Augmentation | Peran |
|---|---|---:|---|---|---|---|
| G1 | Custom CNN 4 blok (Conv3×3–BN–ReLU–MaxPool; 32/64/128/256 channel; GAP; Dropout 0,3; Linear) | 391.466 | Kaiming | RGB | ON | Baseline from-scratch |
| G2 | EfficientNet-B0 | 4.020.358 | Random | RGB | ON | Kontrol arsitektur |
| G3 | EfficientNet-B0 | 4.020.358 | ImageNet (`IMAGENET1K_V1`) | RGB | ON | Model transfer utama |
| G4 | EfficientNet-B0 | 4.020.358 | ImageNet | RGB | **OFF** | Ablation |
| G1g–G3g | Sama dengan G1–G3 | | | grayscale + eq. | ON | Protokol tanpa shortcut |
| G5 | EfficientNet-B0: linear probe 5 epoch (lr 1e-3), lalu unfreeze `features[5:]` 25 epoch (lr 1e-4) | | ImageNet | RGB | ON | Strategi training (suplemen) |

**Alasan pemilihan.**
- G1 adalah baseline sederhana yang dapat dijelaskan dan diterima engineer sebagai
  pembanding.
- G2 memisahkan efek arsitektur dari efek pretraining. Tanpa G2, selisih G1→G3
  mencampur kedua efek tersebut.
- G4 mengisolasi satu keputusan desain, yaitu augmentation; semua komponen lain
  identik, dan hal ini diverifikasi oleh unit test.

**Konfigurasi training (identik untuk semua run kecuali G5):**
- AdamW, lr 3e-4, weight decay 1e-4;
- batch 32, 30 epoch, cosine annealing per step;
- cross-entropy tanpa class weight;
- AMP fp16 di CUDA;
- tanpa early stopping;
- checkpoint dipilih berdasarkan validation Macro F1 (tie-break validation loss);
- test set dievaluasi sekali setelah training;
- seed 42, 43, dan 44.

**Augmentation ON:**
- RandomResizedCrop 224 dengan scale 0,8–1,0 dan ratio 0,75–1,33;
- HorizontalFlip dengan p=0,5;
- RandomAffine: rotasi ±15°, translate 5%, scale 0,9–1,1, shear 5°, area kosong
  diisi warna mean;
- ColorJitter: brightness, contrast, dan saturation 0,2; hue 0,02.

Inspeksi visual augmentation ada di `figures/augmentation_grid.png`.

**Deviasi dari proposal** (lihat juga README):
- Two-stage fine-tuning dipindahkan ke G5 supaya tidak mengacaukan perbandingan
  efek pretraining.
- ResNet-50 tidak dijalankan.
- Threshold keberhasilan diperjelas menjadi +10 *percentage points*.

## 4. Hasil **[HASIL GPU]**

Salin `artifacts/metrics/tables.md`:
- tabel semua group dengan mean ± std Macro F1, accuracy, dan top-3 dari 3 seed,
  jumlah parameter, serta waktu training per run;
- tabel per run;
- `figures/group_macro_f1.png` dan `figures/training_curves.png`.

Bandingkan juga dengan baseline non-deep di Bagian 2. Jika model deep tidak
melampaui baseline LBP (96,3% pada grayscale), tuliskan hal itu secara eksplisit.

Status target proposal (`summary.json` → `headline`): Macro F1 ≥ 85% dan gain
≥ 10 pp atas G1. Interpretasikan dengan mempertimbangkan ceiling effect pada RGB.

## 5. Ablation **[HASIL GPU]**

Isi tabel perbandingan berpasangan dari `tables.md`. Untuk setiap baris
perbandingan, cantumkan: mean Δ ± std dalam percentage points, 95% CI paired
bootstrap per seed, dan **interpretasi**:

| Perbandingan | Pertanyaan yang dijawab |
|---|---|
| G3 vs G1 | Headline: transfer learning vs baseline from-scratch |
| G2 vs G1 | Efek arsitektur |
| G3 vs G2 | Efek pretraining |
| G3 vs G4 | Efek augmentation |
| G3g vs G1g, G3g vs G2g | Efek yang sama tanpa shortcut warna |
| G3 vs G3g | Kontribusi warna |
| G5 vs G3 | Strategi training |

Selisih yang CI-nya mencakup 0 tidak boleh disebut sebagai perbedaan.

Robustness (`artifacts/metrics/robustness.md`): tujuh corruption × 3 severity, ditambah
probe `grayscale` dan `grayscale_equalized`. Penurunan besar pada model RGB di
bawah probe grayscale menunjukkan bahwa model bergantung pada warna.

## 6. Error analysis **[HASIL GPU]**

Sumber: `artifacts/error_analysis/<run_id>/cases.json` dan
`figures/errors/<run_id>/case_*.png`. `batik errors` otomatis beralih ke model
final G3g dan G1 bila model RGB G3 memiliki kurang dari 3 error.

Untuk setiap minimal 3 kasus, tulis:
- ID file, label sebenarnya, prediksi, confidence, dan top-3;
- bukti visual: Grad-CAM, nearest training neighbour di kelas benar dan kelas
  prediksi;
- **intervensi yang membalik prediksi**: full frame, grayscale, flip, normalisasi
  brightness, 5-crop;
- penjelasan kausal yang didukung oleh intervensi tersebut. Contohnya: “crop tengah
  kehilangan border motif; full frame memperbaiki prediksi → penyebabnya konteks
  yang hilang akibat crop.”

Penjelasan ditulis anggota setelah inspeksi visual, bukan dibuat oleh AI.

## 7. Limitations, failure modes, dan 10× compute

**Limitations (sudah pasti):**
- Setiap kelas hanya berasal dari satu kain dan satu sesi foto. Hasil mengukur
  generalisasi ke view baru dari kain yang sama, bukan ke kain lain dengan motif
  yang sama.
- Pada protokol RGB, warna adalah shortcut sempurna.
- Test set hanya berisi 450 gambar dari 668 grup, sehingga CI cukup lebar.
- Resume tidak bit-exact jika `num_workers > 0`.
- Beberapa kernel CUDA tidak deterministik (`warn_only`).

**Failure modes:** isi dari error analysis dan robustness **[HASIL GPU]**.

**Dengan 10× compute/data:**
- Kumpulkan beberapa kain per motif dari pengrajin dan warna berbeda, lalu uji
  dengan *leave-one-cloth-out*. Ini perbaikan terpenting.
- Jalankan learning curve (10/25/50/100% data train) untuk mengukur efisiensi data
  transfer learning.
- Tambahkan backbone lain (ResNet-50, ConvNeXt-T, ViT-S) dan resolusi lebih tinggi
  (384 px) agar isen-isen halus terlihat.
- Evaluasi pada foto kain yang dikenakan atau terlipat sungguhan, bukan hanya
  corruption sintetis.

## 8. Reproducibility statement

| Item | Nilai |
|---|---|
| Seed | Split 42 (beku, checksum); training 42, 43, 44; worker seed diturunkan dari seed training |
| Environment | Python 3.12, PyTorch 2.5.1, torchvision 0.20.1 (`requirements.txt`, `pyproject.toml`) |
| Hardware | **[HASIL GPU]**: nama GPU dari `artifacts/runs/*/env.json` |
| Runtime | **[HASIL GPU]**: `train_seconds` per run dari `summary.json` / `all_runs.csv` |
| Command headline | `make data && make reproduce` (evaluasi checkpoint rilis); `make reproduce-retrain` (training ulang G3 × 3 seed) |
| Model artefact | `artifacts/release/batik_effb0_final.pt` (GitHub Release) |

## 9. Individual contribution statement

Lihat `reports/contribution_statement.md` (wajib ditandatangani seluruh anggota).

## 10. AI use declaration

Lihat `reports/ai_use_declaration.md` (per anggota).

## Referensi

- Winarno, E., Septiarini, A., Hadikurniawati, W., & Hamdani, H. (2024). The Hybrid Features and Supervised Learning for Batik Pattern Classification. *ACM JOCCH* 17(2). https://doi.org/10.1145/3631131
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Chollet, F. (2021). *Deep Learning with Python*, 2nd ed. Manning.
- Géron, A. (2022). *Hands-On Machine Learning*, 3rd ed. O'Reilly.
- Selvaraju, R. R. et al. (2017). Grad-CAM. *ICCV*.
