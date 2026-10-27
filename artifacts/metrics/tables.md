## Comparison of all run groups (test split, mean ± std over seeds)

| Group | Configuration | Params | Seeds | Test Macro F1 (%) | Test Acc (%) | Test Top-3 (%) | Val Macro F1 (%) | Train time / run (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G1 | M1 Custom CNN, scratch, aug ON | 391,466 | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 18.7 |
| G1g | M1 Custom CNN, scratch, aug ON, grayscale-eq | 391,466 | 3 | 84.41 ± 1.45 | 85.04 ± 1.28 | 98.67 ± 1.60 | 87.72 ± 2.74 | 12.0 |
| G2 | M2 EfficientNet-B0, scratch, aug ON | 4,020,358 | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 14.6 |
| G2g | M2 EfficientNet-B0, scratch, aug ON, grayscale-eq | 4,020,358 | 3 | 75.66 ± 2.38 | 75.70 ± 1.67 | 98.30 ± 0.26 | 73.02 ± 4.39 | 14.1 |
| G3 | M3 EfficientNet-B0, ImageNet, aug ON | 4,020,358 | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 17.8 |
| G3g | M3 EfficientNet-B0, ImageNet, aug ON, grayscale-eq | 4,020,358 | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 13.8 |
| G4 | M3 EfficientNet-B0, ImageNet, aug OFF | 4,020,358 | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 9.2 |
| G5 | EfficientNet-B0, ImageNet, two-stage (suppl.) | 4,020,358 | 3 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 15.0 |

## Every run

| Run | Seed | Best epoch | Val Macro F1 | Test Macro F1 | Test Acc | Test Top-3 | Train min | Device |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1_customcnn-scratch_augon_s42 | 42 | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 31.0 | Tesla T4 |
| G1_customcnn-scratch_augon_s43 | 43 | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.1 | Tesla T4 |
| G1_customcnn-scratch_augon_s44 | 44 | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12.1 | Tesla T4 |
| G1g_customcnn-scratch_augon_s42 | 42 | 25 | 0.9043 | 0.8549 | 0.8578 | 1.0000 | 12.6 | Tesla T4 |
| G1g_customcnn-scratch_augon_s43 | 43 | 10 | 0.8496 | 0.8276 | 0.8356 | 0.9689 | 12.6 | Tesla T4 |
| G1g_customcnn-scratch_augon_s44 | 44 | 23 | 0.8779 | 0.8497 | 0.8578 | 0.9911 | 10.8 | Tesla T4 |
| G2_effb0-scratch_augon_s42 | 42 | 27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.1 | Tesla T4 |
| G2_effb0-scratch_augon_s43 | 43 | 19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.0 | Tesla T4 |
| G2_effb0-scratch_augon_s44 | 44 | 28 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.6 | Tesla T4 |
| G2g_effb0-scratch_augon_s42 | 42 | 28 | 0.6816 | 0.7357 | 0.7400 | 0.9844 | 14.7 | Tesla T4 |
| G2g_effb0-scratch_augon_s43 | 43 | 23 | 0.7420 | 0.7514 | 0.7578 | 0.9844 | 14.6 | Tesla T4 |
| G2g_effb0-scratch_augon_s44 | 44 | 22 | 0.7670 | 0.7826 | 0.7733 | 0.9800 | 12.9 | Tesla T4 |
| G3_effb0-pt_augon_s42 | 42 | 26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.2 | Tesla T4 |
| G3_effb0-pt_augon_s43 | 43 | 27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 23.1 | Tesla T4 |
| G3_effb0-pt_augon_s44 | 44 | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.0 | Tesla T4 |
| G3g_effb0-pt_augon_s42 | 42 | 19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.6 | Tesla T4 |
| G3g_effb0-pt_augon_s43 | 43 | 28 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.7 | Tesla T4 |
| G3g_effb0-pt_augon_s44 | 44 | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.1 | Tesla T4 |
| G4_effb0-pt_augoff_s42 | 42 | 26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9.2 | Tesla T4 |
| G4_effb0-pt_augoff_s43 | 43 | 27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9.2 | Tesla T4 |
| G4_effb0-pt_augoff_s44 | 44 | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9.2 | Tesla T4 |
| G5_effb0-pt-2stage_augon_s42 | 42 | 26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 13.2 | Tesla T4 |
| G5_effb0-pt-2stage_augon_s43 | 43 | 26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 17.8 | Tesla T4 |
| G5_effb0-pt-2stage_augon_s44 | 44 | 23 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.0 | Tesla T4 |

## Paired comparisons (Δ Macro F1 = A − B, percentage points)

| A vs B | Question | Seeds | Mean Δ ± std (pp) | Per-seed 95% bootstrap CI (pp) |
|---|---|---|---:|---|
| G3 vs G1 | Transfer learning vs from-scratch baseline (headline) | 42,43,44 | 0.00 ± 0.00 | s42: [0.0, 0.0]; s43: [0.0, 0.0]; s44: [0.0, 0.0] |
| G2 vs G1 | Architecture effect (both random init) | 42,43,44 | 0.00 ± 0.00 | s42: [0.0, 0.0]; s43: [0.0, 0.0]; s44: [0.0, 0.0] |
| G3 vs G2 | Pretrained-initialisation effect (same architecture) | 42,43,44 | 0.00 ± 0.00 | s42: [0.0, 0.0]; s43: [0.0, 0.0]; s44: [0.0, 0.0] |
| G3 vs G4 | Ablation: augmentation ON vs OFF | 42,43,44 | 0.00 ± 0.00 | s42: [0.0, 0.0]; s43: [0.0, 0.0]; s44: [0.0, 0.0] |
| G5 vs G3 | Training strategy: two-stage vs full fine-tuning (suppl.) | 42,43,44 | 0.00 ± 0.00 | s42: [0.0, 0.0]; s43: [0.0, 0.0]; s44: [0.0, 0.0] |
| G3g vs G1g | Grayscale protocol: transfer learning vs from-scratch baseline | 42,43,44 | 15.59 ± 1.45 | s42: [11.5, 17.8]; s43: [13.9, 21.1]; s44: [12.2, 18.1] |
| G2g vs G1g | Grayscale protocol: architecture effect | 42,43,44 | -8.75 ± 2.78 | s42: [-16.0, -7.7]; s43: [-11.9, -3.2]; s44: [-10.7, -2.9] |
| G3g vs G2g | Grayscale protocol: pretrained-initialisation effect | 42,43,44 | 24.34 ± 2.38 | s42: [23.3, 29.8]; s43: [21.7, 28.5]; s44: [18.5, 25.2] |
| G3 vs G3g | Cost of removing colour for the transfer model | 42,43,44 | 0.00 ± 0.00 | s42: [0.0, 0.0]; s43: [0.0, 0.0]; s44: [0.0, 0.0] |
