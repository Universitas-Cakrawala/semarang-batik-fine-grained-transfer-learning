# Data licence and redistribution

**Dataset:** Semarang Batik Dataset — Kaggle `edywinarno/semarang-batik-dataset`
(version 2, last updated 2023-10-16)
<https://www.kaggle.com/datasets/edywinarno/semarang-batik-dataset>

**Licence:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(Attribution – NonCommercial – ShareAlike). The proposal described the licence as
“Academic Research Use”. That description was wrong, and this file supersedes it.

**Required citation** (requested by the dataset owner):

```bibtex
@article{10.1145/3631131,
  author    = {Winarno, Edy and Septiarini, Anindita and Hadikurniawati, Wiwien and Hamdani, Hamdani},
  title     = {The Hybrid Features and Supervised Learning for Batik Pattern Classification},
  journal   = {ACM Journal on Computing and Cultural Heritage},
  volume    = {17}, number = {2}, year = {2024},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3631131}
}
```

## What this repository does and does not contain

| Item | In git? | Why |
|---|---|---|
| Raw images (`data/raw/`, ~13 GB) | **No** | Size; redistribution is governed by CC BY-NC-SA. Fetch with `batik download` |
| Resized cache (`data/processed/`) | **No** | Derivative of the dataset (ShareAlike); rebuilt deterministically by `batik audit` |
| Manifests (`data/manifests/*.csv/json`) | Yes | File names, hashes, labels, and split assignment only (no pixels). Needed to reproduce the exact split |
| Trained weights (`best.pt`) | [GitHub Release v1.0](https://github.com/Universitas-Cakrawala/semarang-batik-fine-grained-transfer-learning/releases/tag/v1.0) asset | Model weights trained on NC data. Use is limited to non-commercial purposes, consistent with the dataset licence |

Every use of the dataset or of the models trained here must be **non-commercial**,
must credit the authors above, and must share derivatives under the same licence.
The "e-commerce verification" use case in the proposal is a motivating scenario
only, and it is out of scope for deployment.
