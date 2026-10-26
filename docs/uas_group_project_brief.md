# Deep Learning — Final Assessment (UAS), Session 16

## Group Project Brief

## Table of Contents

1. [Core Rule](#1-core-rule)
2. [Tracks](#2-tracks)
3. [Required Technical Content](#3-required-technical-content)
4. [Deliverables](#4-deliverables)
5. [Report Structure](#5-report-structure)
6. [Submission](#6-submission)
7. [References](#7-references)

## 1. Core Rule

Every project must train or fine-tune at least one model itself.

A project that only calls a hosted model API does not demonstrate the competencies
assessed in this course and will not pass. Retrieval-augmented generation (RAG),
prompting, and adapter methods are permitted, but they must be combined with a
model that the group trains, fine-tunes, or adapts using its own training loop.
This ensures all four CPMK of the course are exercised.

## 2. Tracks

Choose one track. Within a track, choose your own problem and dataset. Two groups
may not work on the same dataset and task combination; approvals are first come,
first served through the proposal.

| Track | Focus | Mandatory element |
|---|---|---|
| A | Vision classification with transfer learning | From-scratch vs. transfer-learning comparison |
| B | Sequence modelling: RNN vs. attention | Distance sweep (accuracy vs. dependency distance, as in Session 9) |
| C | Foundation models, adaptation and retrieval | Retrieval measured separately from generation (hit-at-rank-k on a fixed query set) |
| D | Generative models | Quantitative sample quality/diversity evaluation plus mode collapse or degradation analysis |

### Track A — Vision Classification with Transfer Learning

Train a convolutional classifier and compare a model trained from scratch against
a transfer-learning approach.

**Verified datasets:**

- Indonesian Batik Motifs — 983 images, 20 motifs (Kaggle)
- Batik Nusantara — 20 classes, 40 images per class at 224×224 (Kaggle)
- Semarang Batik — 10 classes, approximately 3,000 images (Kaggle; citable ACM paper)
- Microsoft Cats vs. Dogs
- Intel Image Classification
- Fashion-MNIST (available directly through Keras)
- Oxford-IIIT Pet

### Track B — Sequence Modelling: Recurrent vs. Attention

Model a sequence task and compare recurrent and attention-based approaches,
including a test of long-range dependency.

**Verified datasets:**

- UCR Time Series Classification Archive — train/test partitions per dataset, hundreds of datasets
- UCI Beijing PM2.5 — hourly, 2010–2014
- UCR/Hexagon time-series anomaly detection sets
- UCI Human Activity Recognition
- Gesture and sign-language landmark sequences

**Mandatory element:** a distance sweep showing accuracy against dependency
distance, as used in Session 9.

### Track C — Foundation Models, Adaptation and Retrieval

Adapt a pretrained language model to a narrow task and compare adaptation
strategies. At least one strategy must involve training or fine-tuning by the
group.

**Verified resources:**

- The course RPS itself as a retrieval corpus
- Any public document collection (regulations, product manuals, open textbooks)
- 1–3B open-weight models such as SmolLM2 or Qwen-1.5B for adapter experiments

**Mandatory element:** retrieval measured separately from generation, using
hit-at-rank-k over a fixed query set.

### Track D — Generative Models

Train a generative model on a narrow dataset and evaluate it. The strongest
version uses the generative model to augment a downstream classifier and measures
whether the classifier improves.

**Suggested datasets:**

- Narrower classes from any Track A dataset
- Fashion-MNIST
- Small face or flower collections with permissive licences
- Synthetic shapes

**Mandatory element:** a quantitative evaluation of sample quality or diversity
beyond displaying a grid of samples, plus a documented attempt at mode collapse
or degradation analysis.

## 3. Required Technical Content

Every project must include all four of the following:

1. **Two or more model families compared, with a stated reason.** Examples:
   from-scratch CNN vs. transfer learning; LSTM vs. transformer; prompting vs.
   retrieval vs. adapter tuning; GAN vs. diffusion.
2. **At least one ablation isolating a single design decision**, with everything
   else held constant. Examples: augmentation on/off; dropout 0.0 vs. 0.3;
   sequence length 20 vs. 100; retrieval `k = 1` vs. `3`; adapter rank 4 vs. 16.
3. **Error analysis with at least three concrete failures**, identifying what the
   model got wrong and giving a causal explanation rather than a description. A
   confusion matrix alone is not an error analysis.
4. **A stated baseline** a reasonable engineer would accept, reported with the
   same metric and the same test split as the proposed model.

## 4. Deliverables

| # | Deliverable | When | Notes |
|---:|---|---|---|
| 1 | Proposal (template provided) | End of Session 14 | One page. Must be approved before modelling begins. |
| 2 | Reproducible repository | With final submission | Fixed seeds, environment file, README with the single command that reproduces the headline number, saved model artefact |
| 3 | Report (8–12 pages) | Session 16 | Structure in Section 5 |
| 4 | Demo and Q&A (10 + 5 min) | Session 15 | Uses the Session 15 presentation slot |
| 5 | Individual viva (5 min per member) | Session 16 | Each member examined on their own code |

> **Warning:** A repository that does not reproduce its own headline result
> without lecturer intervention cannot score above 60, regardless of the
> reported metric.

## 5. Report Structure

1. Problem statement and why deep learning is appropriate (maximum 2 pages)
2. Data: source, licence, size, class balance, split procedure, and an explicit
   leakage check
3. Baselines and proposed approach, with architectures and exact training
   configuration
4. Results: comparison table of all runs, with metric, standard deviation across
   seeds where available, and training cost
5. Ablation results, with interpretation of each row
6. Error analysis: at least three failures with evidence and a causal explanation
7. Limitations, failure modes, and what you would do with ten times the compute
8. Reproducibility statement: seed, hardware, runtime, and exact command
9. Individual contribution statement signed by every member
10. AI use declaration per member

## 6. Submission

Submit a single archive or repository link containing:

- [ ] The report
- [ ] The repository
- [ ] The saved model artefact or a download link
- [ ] The per-member contribution statement
- [ ] The AI use declarations

Keep the folder structure as documented in the README so the automatic
submission checker behaves predictably.

## 7. References

### RPS Sources

- Chollet, F. (2021). *Deep Learning with Python*, 2nd ed. — primary RPS source.
- Géron, A. (2022). *Hands-On Machine Learning*, 3rd ed. — supporting RPS source.
- Xiao, T. & Zhu, J. (2025). *Foundations of Large Language Models*,
  arXiv:2501.09223 — supporting.
- Bommasani, R. et al. (2021). *On the Opportunities and Risks of Foundation
  Models*, arXiv:2108.07258 — supporting.

### Non-RPS Enrichment

- Hu et al. (2021). *LoRA*, arXiv:2106.09685.
- Lewis et al. (2020). *Retrieval-augmented generation*.
- Ho et al. (2020). *Denoising diffusion*.
- He et al. (2015). *ResNet*.
- Tan & Le (2019). *EfficientNet*.
