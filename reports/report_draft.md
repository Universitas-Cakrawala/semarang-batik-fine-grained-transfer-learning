# Evaluasi From-Scratch CNN dan Fine-Tuned EfficientNet-B0 untuk Fine-Grained Classification Motif Batik Semarang

**Kelompok 5, Track A** · Tita Noviana, Titanio Yudista, Suci Fransisca Sisilia R., Fajar Dwiharjo

> **Status draf.** Semua bagian sudah diisi dari artefak di `artifacts/`. Sebelum
> final: bagian bertanda **[VERIFIKASI ANGGOTA]** diperiksa dan ditulis ulang oleh
> anggota, lalu draf dipadatkan ke 8–12 halaman. Setiap angka harus tetap dapat
> ditelusuri ke `artifacts/metrics/*` atau `artifacts/runs/*`.

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

## 4. Hasil

Sumber: `artifacts/metrics/tables.md` (dihasilkan `batik aggregate`). Semua angka
adalah Macro F1 pada test split beku (450 gambar), mean ± std dari seed 42/43/44.

| Group | Konfigurasi | Parameter | Test Macro F1 (%) | Test Acc (%) | Test Top-3 (%) | Val Macro F1 (%) | Waktu / run (menit) |
|---|---|---:|---:|---:|---:|---:|---:|
| G1 | Custom CNN, scratch, aug ON | 391.466 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 18,7 |
| G2 | EfficientNet-B0, scratch, aug ON | 4.020.358 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 14,6 |
| G3 | EfficientNet-B0, ImageNet, aug ON | 4.020.358 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 17,8 |
| G4 | EfficientNet-B0, ImageNet, aug OFF | 4.020.358 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 9,2 |
| G5 | EfficientNet-B0, ImageNet, two-stage | 4.020.358 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 15,0 |
| G1g | Custom CNN, scratch, grayscale-eq | 391.466 | 84,41 ± 1,45 | 85,04 ± 1,28 | 98,67 ± 1,60 | 87,72 ± 2,74 | 12,0 |
| G2g | EfficientNet-B0, scratch, grayscale-eq | 4.020.358 | 75,66 ± 2,38 | 75,70 ± 1,67 | 98,30 ± 0,26 | 73,02 ± 4,39 | 14,1 |
| G3g | EfficientNet-B0, ImageNet, grayscale-eq | 4.020.358 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 100,00 ± 0,00 | 13,8 |

Tabel per run ada di `tables.md`; kurva training di `figures/training_curves.png`
dan ringkasan group di `figures/group_macro_f1.png`.

**Protokol RGB mencapai ceiling.** Semua model RGB, termasuk baseline Custom CNN
dengan 0,39 M parameter, mencapai 100% Macro F1 di ketiga seed. Hal ini konsisten
dengan baseline non-deep di Bagian 2: histogram warna HSV saja sudah 100%. Pada
RGB, model deep **tidak** lebih baik daripada logistic regression di atas histogram
warna, sehingga protokol ini tidak dapat membedakan model.

**Protokol grayscale membedakan model.** Setelah warna dan kecerahan global
dihilangkan:
- G3g (transfer) tetap 100,00%, melampaui baseline non-deep terbaik pada input
  yang sama (LBP + logistic regression, 96,26%).
- G1g (84,41%) dan G2g (75,66%) **berada di bawah** baseline LBP non-deep. Kedua
  model from-scratch tidak mampu mempelajari tekstur motif sebaik fitur LBP
  rancangan manual dari 2.090 gambar train.

**Kecepatan konvergensi** (`artifacts/runs/*_s42/history.csv`): G3 dan G3g sudah
mencapai val Macro F1 ≥ 0,99 pada epoch 1. G1 (RGB) mencapainya pada epoch 4 dan
G2 pada epoch 7, sedangkan G1g dan G2g tidak pernah mencapainya dalam 30 epoch.

**Status target proposal** (`summary.json` → `headline`):

| Target | Hasil | Status |
|---|---|---|
| Macro F1 G3 ≥ 85% | 100,00 ± 0,00% | Tercapai |
| Gain G3 atas G1 ≥ +10 pp (RGB) | +0,00 pp | **Tidak tercapai** (kedua model di ceiling) |
| Gain yang sama pada protokol grayscale (G3g vs G1g) | +15,59 ± 1,45 pp | Tercapai, CI semua seed > 0 |

Hipotesis proposal tidak dapat diuji pada protokol RGB karena ceiling effect.
Hipotesis tersebut didukung pada protokol grayscale, yaitu kondisi di mana model
harus mengandalkan tekstur motif.

## 5. Ablation

Sumber: tabel *paired comparisons* di `tables.md`. Δ = Macro F1 A − B dalam
percentage points, dipasangkan per seed. CI adalah 95% paired bootstrap per seed
(2.000 resample test set). Selisih yang CI-nya mencakup 0 tidak disebut sebagai
perbedaan.

| A vs B | Pertanyaan | Mean Δ ± std (pp) | 95% CI per seed (pp) | Interpretasi |
|---|---|---:|---|---|
| G3 vs G1 | Headline: transfer vs from-scratch | 0,00 ± 0,00 | [0,0; 0,0] × 3 | Tidak ada perbedaan (ceiling) |
| G2 vs G1 | Efek arsitektur | 0,00 ± 0,00 | [0,0; 0,0] × 3 | Tidak ada perbedaan (ceiling) |
| G3 vs G2 | Efek pretraining | 0,00 ± 0,00 | [0,0; 0,0] × 3 | Tidak ada perbedaan (ceiling) |
| G3 vs G4 | Efek augmentation | 0,00 ± 0,00 | [0,0; 0,0] × 3 | Tidak ada perbedaan pada test bersih; lihat robustness |
| G5 vs G3 | Two-stage vs full fine-tuning | 0,00 ± 0,00 | [0,0; 0,0] × 3 | Tidak ada perbedaan (ceiling) |
| G3g vs G1g | Transfer vs from-scratch (grayscale) | **+15,59 ± 1,45** | s42 [11,5; 17,8]; s43 [13,9; 21,1]; s44 [12,2; 18,1] | Transfer lebih baik; CI semua seed > 0 |
| G2g vs G1g | Efek arsitektur (grayscale) | **−8,75 ± 2,78** | s42 [−16,0; −7,7]; s43 [−11,9; −3,2]; s44 [−10,7; −2,9] | EfficientNet-B0 scratch **lebih buruk** daripada CNN kecil |
| G3g vs G2g | Efek pretraining (grayscale) | **+24,34 ± 2,38** | s42 [23,3; 29,8]; s43 [21,7; 28,5]; s44 [18,5; 25,2] | Efek terbesar; CI semua seed > 0 |
| G3 vs G3g | Kontribusi warna untuk model transfer | 0,00 ± 0,00 | [0,0; 0,0] × 3 | Model transfer yang dilatih tanpa warna tetap 100%; warna tidak diperlukan jika fitur pretrained dipakai |

**Interpretasi.** Pada protokol grayscale, keuntungan G3g atas G1g (+15,6 pp) bukan
berasal dari arsitektur. Dengan inisialisasi acak, EfficientNet-B0 justru 8,8 pp
di bawah CNN kecil, kemungkinan karena 4,0 M parameter terlalu banyak untuk 2.090
gambar. Seluruh keuntungan berasal dari bobot ImageNet (+24,3 pp atas G2g).

**Robustness** (`artifacts/metrics/robustness.md`; satu run final per group, sehingga
tanpa CI). Corruption hanya diterapkan saat evaluasi, dengan 3 tingkat severity:

| Kondisi (Macro F1 %) | G1 | G2 | G3 | G4 (aug OFF) | G1g | G2g | G3g |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 100,00 | 100,00 | 100,00 | 100,00 | 85,49 | 78,26 | 100,00 |
| Brightness s3 | 66,10 | 67,04 | 99,32 | 94,92 | 84,13 | 77,71 | 100,00 |
| Contrast s3 | 99,19 | 73,38 | 100,00 | 90,82 | 78,99 | 77,89 | 100,00 |
| Occlusion s3 | 100,00 | 45,74 | 100,00 | 56,34 | 18,71 | 32,33 | 94,86 |
| Blur s1 | 100,00 | 99,55 | 99,77 | 100,00 | 26,09 | 35,12 | 99,28 |
| Blur s3 | 61,07 | 46,19 | 98,63 | 72,50 | 8,93 | 6,13 | 35,15 |
| Probe grayscale | 7,32 | 21,89 | 67,42 | 34,66 | 85,49 | 78,26 | 100,00 |
| Probe grayscale-eq | 5,96 | 12,38 | 32,87 | 31,90 | 85,49 | 77,97 | 100,00 |

Temuan robustness:
1. **Ketergantungan warna.** Di bawah probe grayscale, G1 jatuh ke 7,3% (hampir
   acak untuk 10 kelas), G2 ke 21,9%, dan G3 ke 67,4%. Ketiganya 100% pada test
   bersih, tetapi hanya model transfer yang mempertahankan sebagian informasi
   tekstur.
2. **Augmentation berpengaruh di luar test bersih.** G3 dan G4 sama-sama 100% pada
   test bersih, tetapi pada occlusion s3 G3 tetap 100,00% sedangkan G4 turun ke
   56,34%. Pada blur s3 hasilnya 98,63% vs 72,50%, dan pada contrast s3 100,00% vs
   90,82%. Ablation augmentation baru terlihat di bawah distribution shift.
3. **Model from-scratch sangat sensitif terhadap blur.** Blur tingkat 1 saja
   menurunkan G1g dari 85,5% ke 26,1% dan G2g dari 78,3% ke 35,1%. Model grayscale
   from-scratch tampak bergantung pada detail frekuensi tinggi, bukan bentuk motif.
4. Rotasi, perspektif, dan deformasi elastis hampir tidak berpengaruh pada model
   RGB, termasuk G4 tanpa augmentation. Kemungkinan penyebabnya adalah data train
   sendiri sudah berisi salinan rotasi/flip dari setiap jepretan (Bagian 2).

## 6. Error analysis

Sumber: `artifacts/error_analysis/G1g_customcnn-scratch_augon_s42/cases.json` dan
`figures/errors/G1g_customcnn-scratch_augon_s42/case_01–10.png`.

**Pemilihan model.** Model final G3, G3g, dan G1 tidak memiliki satu pun error pada
test set. Aturan fallback awal (G3 → G3g → G1) karena itu tidak menghasilkan kasus.
Setelah hasil ini terlihat, aturan diperluas ke G1g lalu G2g. Model yang dipakai
adalah run final **G1g seed 42** (val Macro F1 tertinggi dalam G1g): **64 error
dari 450 gambar**. Sepuluh kasus dipilih bergiliran per pasangan kebingungan (yang
paling yakin dari setiap pasangan lebih dulu).

**Distribusi error.** Dua pasangan mendominasi: Asem Warak → Kembang Sepatu
(23 dari 64 error, 36%) dan Gambang Semarangan → Asem Arang (11 error).

**Intervensi.** Setiap gambar yang salah diprediksi ulang dengan lima perubahan
input: *full frame* (seluruh foto di-resize ke 224 tanpa crop), *grayscale*,
*horizontal flip*, *normalisasi brightness* ke rata-rata train, dan *5-crop*.
Pada model G1g, intervensi grayscale tidak berpengaruh karena input sudah grayscale.

| # | File | Label → Prediksi (conf.) | Top-3 | Diperbaiki oleh | Sharpness / median kelas |
|---:|---|---|---|---|---:|
| 1 | `asem warak (61).JPG` | Asem Warak → Kembang Sepatu (0,94) | KS, AW, Blekok | — | 3.868 / 3.455 |
| 2 | `Gambang Semarangan (123).JPG` | Gambang Semarangan → Asem Arang (0,83) | AA, Semarangan, GS | — | 14.092 / 21.261 |
| 3 | `Kembang Sepatu (99).JPG` | Kembang Sepatu → Asem Arang (0,67) | AA, KS, Semarangan | full frame (p=0,99), 5-crop | 5.389 / 8.484 |
| 4 | `blekok (274).JPG` | Blekok → Asem Sinom (0,65) | AS, Blekok, AW | — | 1.761 / 6.130 |
| 5 | `Asem Sinom (82).JPG` | Asem Sinom → Warak Beras Utah (0,60) | WBU, AS, Blekok Warak | hflip | 1.319 / 967 |
| 6 | `blekok (258).JPG` | Blekok → Kembang Sepatu (0,59) | KS, AW, Blekok | full frame | 1.811 / 6.130 |
| 7 | `Asem Sinom (149).JPG` | Asem Sinom → Kembang Sepatu (0,53) | KS, AS, Blekok | full frame (p=0,71), 5-crop | 618 / 967 |
| 8 | `Tugu Muda (122).JPG` | Tugu Muda → Warak Beras Utah (0,35) | WBU, TM, Semarangan | full frame | 7.091 / 15.874 |
| 9 | `Tugu Muda (123).JPG` | Tugu Muda → Semarangan (0,31) | Semarangan, WBU, TM | full frame | 7.091 / 15.874 |
| 10 | `Asem Sinom (272).JPG` | Asem Sinom → Asem Arang (0,48) | AA, AS, WBU | full frame, hflip, 5-crop | 670 / 967 |

Normalisasi brightness tidak memperbaiki satu pun kasus. Karena itu, pencahayaan
**bukan** penyebab error.

**Mekanisme kegagalan yang didukung intervensi:**

1. **Konteks hilang akibat center crop** (kasus 3, 6, 7, 8, 9, 10). Prediksi berbalik
   menjadi benar ketika model melihat seluruh foto (full frame) atau beberapa crop
   (5-crop). Pada kasus 3, p(kelas benar) naik dari 0,33 ke 0,99. Penyebabnya adalah
   CenterCrop 224 dari sisi pendek 256 yang membuang bagian motif pembeda di tepi
   foto. Penyebab ini terletak pada protokol evaluasi, bukan pada bobot model.
2. **Tekstur mirip tanpa warna, tidak dapat diperbaiki input** (kasus 1, 2, 4).
   Tidak satu pun intervensi membalik prediksi, dan confidence tinggi (0,65–0,94).
   Kasus 1 mewakili pasangan error terbesar (23 error). Informasinya sebenarnya
   ada di gambar grayscale, karena G3g mengklasifikasikan seluruh test set dengan
   benar. Jadi penyebabnya adalah fitur yang dipelajari from scratch dari 2.090
   gambar tidak cukup membedakan isen-isen kedua motif.
3. **Foto kurang tajam** (8 dari 10 kasus memiliki sharpness di bawah median
   kelasnya; kasus 4 dan 6 hanya ±29% dari median). Ini konsisten dengan uji
   robustness: blur tingkat 1 menurunkan G1g dari 85,5% ke 26,1%.
4. **Sensitivitas orientasi** (kasus 5, 10 diperbaiki oleh flip; kasus 8 dan 9).
   Kasus 8 dan 9 adalah dua salinan dari jepretan yang sama (brightness dan
   sharpness identik) dan berada dalam grup split yang sama. Model memberi dua
   prediksi salah yang **berbeda** untuk keduanya. Fitur G1g belum invarian
   terhadap transformasi yang tidak mengubah identitas motif.

Catatan: pada seluruh 10 kasus, nearest training neighbour di ruang embedding G1g
berasal dari kelas yang benar (cosine 0,957–0,998), lebih dekat daripada neighbour
kelas prediksi (0,860–0,956). Ini mengindikasikan bahwa informasi pembeda masih
ada di representasi dan kesalahan terjadi di classifier head. Indikasi ini belum
diuji pada seluruh test set (misalnya dengan k-NN di ruang embedding), dan
kedekatan tersebut dapat dipengaruhi oleh view lain dari kain yang sama.

> **[VERIFIKASI ANGGOTA]** Bagian ini ditulis dari data intervensi di
> `cases.json`. Sebelum laporan final, anggota yang memegang error analysis wajib
> memeriksa `figures/errors/G1g_customcnn-scratch_augon_s42/case_*.png` (Grad-CAM
> dan nearest neighbour), memastikan penjelasan di atas sesuai dengan yang terlihat,
> lalu menulis ulang dengan kata-kata sendiri.

## 7. Limitations, failure modes, dan 10× compute

**Limitations (sudah pasti):**
- Setiap kelas hanya berasal dari satu kain dan satu sesi foto. Hasil mengukur
  generalisasi ke view baru dari kain yang sama, bukan ke kain lain dengan motif
  yang sama.
- Pada protokol RGB, warna adalah shortcut sempurna.
- Test set hanya berisi 450 gambar dari 668 grup, sehingga CI cukup lebar.
- Resume tidak bit-exact jika `num_workers > 0`.
- Beberapa kernel CUDA tidak deterministik (`warn_only`).

- Robustness dievaluasi pada satu run final per group, sehingga tidak ada CI antar
  seed untuk tabel robustness.
- Training dijalankan di Colab dengan torch 2.11 / Python 3.13, bukan versi yang
  di-pin di `requirements.txt` (lihat Bagian 8).

**Failure modes:**
- **Model RGB bergantung pada warna.** Kain dengan motif sama tetapi warna berbeda
  kemungkinan besar akan salah diklasifikasikan. Probe grayscale menurunkan G1
  dari 100% ke 7,3%, dan G3 ke 67,4%.
- **Foto buram.** Blur s3 menurunkan G1 ke 61,1%. Model grayscale from-scratch
  sudah gagal pada blur ringan (G1g 26,1% pada s1).
- **Oklusi tanpa augmentation.** G4 turun ke 56,3% pada occlusion s3, sementara G3
  tetap 100%.
- **Crop yang memotong motif.** Enam dari sepuluh kasus error G1g menjadi benar
  dengan full frame atau 5-crop.
- **Pasangan motif mirip** saat warna tidak tersedia: Asem Warak ↔ Kembang Sepatu
  dan Gambang Semarangan ↔ Asem Arang.

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
| Environment (pin repo) | Python 3.12, PyTorch 2.5.1, torchvision 0.20.1 (`requirements.txt`, `pyproject.toml`) |
| Environment (training aktual) | Google Colab: Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0, CUDA 13.0, AMP fp16; kode commit `d4f6709` (`artifacts/runs/*/env.json`) |
| Hardware | NVIDIA Tesla T4 16 GB, 2 vCPU, `num_workers=2`, untuk ke-24 run |
| Runtime | Total 5,76 jam GPU untuk 24 run. Per run 9,2–23,1 menit (±26–30 detik/epoch untuk EfficientNet-B0 dengan augmentation). G1 s42 tercatat 31,0 menit karena epoch pertama membaca cache gambar dari Google Drive (1.102 detik). Peak memory GPU: 0,7 GB (Custom CNN), 1,5 GB (EfficientNet-B0) (`all_runs.csv`) |
| Verifikasi lintas lingkungan | `batik aggregate` yang dijalankan ulang di laptop (CPU, PyTorch 2.5.1) dari run Colab menghasilkan `tables.md`, `all_runs.csv`, `group_summary.csv`, `summary.json`, dan `final_model.json` yang identik byte per byte |
| Sanity check | Lulus semua di lingkungan pin (`artifacts/sanity/sanity_report.json`). Di Colab, dua pemeriksaan gagal: `split_reproducible` (membuat ulang split dengan versi library berbeda memberi hasil berbeda; training tetap memakai `split.csv` beku yang checksum-nya cocok) dan `overfit_efficientnet_b0` (akurasi 0,875 < 0,95 pada 32 gambar; training penuh G2 tetap mencapai 100%) |
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
