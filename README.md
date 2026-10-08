# Which generative-AI assistant does a phone use? Session-scarce observers on encrypted connection metadata

Code, session-level splits and analysis scripts for an anonymous submission. The paper measures what a passive
on-path observer learns about generative-AI use from the direction, size and timing of the first 64 packets of each
connection, when it can label only a handful of 15-minute capture sessions. This repository contains

* the **observer for scarce labels**: a Transformer encoder over per-packet metadata, pre-trained by masked packet
  modelling on unlabelled traffic and fine-tuned on K labelled sessions per class;
* the **reference observers** run under the same protocol: LightGBM on packet statistics, a nearest neighbour on the
  first ten packets, a patch Transformer, ported website-fingerprinting attacks (Deep Fingerprinting, Tik-Tok, Triplet
  Fingerprinting, Contrastive Fingerprinting, NetCLR), and the byte-level models (the dataset paper's 1D-CNN, YaTC,
  ET-BERT), plus a server-address lookup;
* the **defence simulations** (ECH handshake padding, padding and jitter, FRONT, Tamaraw, adversarial shaping) with
  their byte and delay costs, over the first 64 packets and over whole connections;
* every **split** used in the paper (`configs/splits/`) and the scripts that turn run outputs into the paper's tables
  and figures.

No traffic was captured for this work. Both campaigns are public datasets; they are not redistributed here.

## Layout

```
src/                     model and training code
  pretrain_meta.py         masked packet modelling on unlabelled connections        -> weights/<name>.pt
  train_meta_ssl.py        fine-tuning on K labelled sessions (also: no pre-training, linear probe)
  train_meta.py            input pipeline shared by all metadata observers; patch-Transformer reference
  train_wf_faithful.py     DF, Tik-Tok, TF, CF, NetCLR ported to the 64x4 metadata sequence (with train_wf_baselines.py)
  train_yatc.py, train_etbert.py, carriers/      byte-level reference observers built on the authors' code
  data_mfr.py              access to the derived arrays and the session splits
scripts/                 data building, reference observers, defences, analyses
  run_paper.sh             every training run of the paper, in order, with the exact flags
paper/                   generators of the LaTeX tables and PDF figures
configs/splits/          session-level splits (K-session draws, temporal, unseen phone, standard 60/20/20)
configs/lgbm_rounds.json LightGBM rounds, fixed once on validation sessions of the second campaign
```

## Setup

```bash
pip install -r requirements.txt          # install PyTorch for your CUDA version first
mkdir -p data third_party weights results paper/tables paper/figs
```

**Data** (from the MIRAGE project, https://traffic.comics.unina.it/mirage/):

| file | where |
|---|---|
| MIRAGE-GenAI-2025 | unzip to `data/mirage2025genai/` (so that `data/mirage2025genai/generic/` exists) |
| MIRAGE-COVID-CCMA-2022 | keep the archive as `data/MIRAGE-COVID-CCMA-2022.zip` |
| MIRAGE-2019 (only for the mixed pre-training corpora) | keep the archive as `data/MIRAGE-2019_v2.tar.gz` |

**Third-party code and weights**, only needed for the corresponding reference observers:

| observer | checkout | weights |
|---|---|---|
| patch Transformer | `third_party/Medformer` (DL4mHealth/Medformer, commit `891f65b`) | none |
| YaTC | `third_party/YaTC` (NSSL-SJTU/YaTC, commit `a220b75`) | released pre-trained model as `weights/yatc_pretrained-model.pth` |
| ET-BERT | `third_party/ET-BERT` (linwhitehat/ET-BERT, commit `d594706`) | released pre-trained model as `weights/etbert_pretrained_model.bin` |

## Pipeline

```bash
# 1. derived arrays: per connection the first 64 packets (direction, IP bytes, payload bytes, inter-arrival time),
#    payload windows for the byte-level observers, labels, capture session and phone
python scripts/build_dataset.py            # MIRAGE-GenAI-2025  -> data/derived/genai_*
python scripts/build_dataset_ccma.py       # MIRAGE-COVID-CCMA-2022 -> data/derived/ccma_*
python scripts/build_background.py         # connections of the same sessions that belong to no target app
python scripts/build_m2019_meta.py         # optional: MIRAGE-2019 metadata for the mixed pre-training corpora
python scripts/build_etbert_tokens.py      # optional: tokens for ET-BERT

# 2. splits are shipped in configs/splits/; they were written by scripts/make_splits*.py

# 3. pre-train the encoder (labels unused; about 40 minutes on one consumer GPU)
python src/pretrain_meta.py --epochs 400 --d 192 --layers 8 --heads 8 --ff 384 --name meta_ssl_mpm_XL

# 4. fine-tune on K = 4 labelled sessions per class and test on unseen sessions (under two minutes)
CKPT_SELECT=last python src/train_meta_ssl.py --run_name fs4_sslXL_s0 --tag main --seed 0 \
    --split configs/splits/fewshot_k4_s0.json --dataset genai --target joint \
    --init weights/meta_ssl_mpm_XL.pt --mode ft --lr 1e-3 --lr_enc 3e-4 --epochs 60 --bs 64 --drop_head 0.3

# the strongest metadata reference on the same sessions
LGBM_ROUNDS=1600 python scripts/train_baselines.py --which lgbm --seeds 0 \
    --split configs/splits/fewshot_k4_s0.json --dataset genai --target joint --name fs4 --tag main
```

`scripts/run_paper.sh` lists every run of the paper in this form (all K, five K-session draws, both campaigns, temporal
and unseen-phone splits, defences, open world, the 20-draw primary comparisons).

### Protocol

* **Labels are counted in capture sessions.** Every split is by session, never by connection. A K-session split holds K
  training sessions per class; its test sessions are those of the standard split and are never used for training,
  normalisation or selection. Pre-training corpora exclude the test sessions of the split they are evaluated on.
* **No labelled session beyond K.** With `CKPT_SELECT=last` every observer trains for a fixed schedule and keeps its
  final model. The encoder's recipe and LightGBM's number of rounds were fixed once on validation sessions of the
  nine-app campaign and are used unchanged everywhere.
* **Run records.** A run writes `results/runs/<run id>/test_preds.npz` (test indices, labels, predictions) and appends
  one line to `results/metrics.jsonl` (validation) and `results/locked_test_metrics.jsonl` (test). The analysis scripts
  read these files by run id.

## From runs to the paper

| in the paper | scripts |
|---|---|
| assistant, modality and session-vote accuracy of the main observers | `scripts/threat_stats.py`, `paper/make_table_threat.py` |
| all observers, K = 1, 2, 4, 8, both campaigns | `scripts/summarize_sota_fewshot.py`, `scripts/boot2.py` |
| primary comparisons over 20 draws, three pre-training seeds | `scripts/make_splits_draws20.py`, `scripts/draws20_primary.py` |
| device window without app grouping | `scripts/window_assistant.py`, `scripts/window_encoder_bg.py` |
| where the leak is (handshake, later packets, server names, classes) | `scripts/leak_source.py`, `scripts/construct_validity.py`, `scripts/google_app_check.py`, `scripts/session_extras.py` |
| server addresses: lookup observer, label age, unseen phone, front-end operators | `scripts/ip_baseline.py`, `scripts/addr_robust.py`, `scripts/cdn_attribution.py`, `scripts/archive_ip_ranges.py`, `scripts/cdn_attribution_archived.py` |
| pre-training corpora of equal compute | `scripts/dilution_stats.py` |
| LightGBM's number of rounds, fixed on validation sessions | `scripts/select_lgbm_rounds.py` |
| open world, per connection and per device window | `scripts/open_world_reject.py`, `scripts/ow_rescore.py`, `scripts/ow_windows_exact.py`, `paper/make_table_openworld.py` |
| label age and unseen phone | `scripts/robust_assistant.py`, `scripts/robust_assistant_knn.py`, `scripts/drift_gaps.py`, `paper/make_tables_def_drift.py`, `paper/make_tables_extra_v4.py`, `paper/make_tab_robust3.py` |
| defences over the first 64 packets and their cost | `scripts/wf_defenses.py`, `scripts/defence_stats.py`, `paper/make_tables_def_drift.py`, `paper/make_tab_robust3.py` |
| adversarial shaping and its transfer | `scripts/adv_eval.py`, `scripts/adv_transfer.py` |
| defences over whole connections, flow-record observer | `scripts/wf_defenses_whole.py`, `scripts/whole_connection_defences.py` |
| figures | `paper/make_figures_main.py` |

The operators' IP-range lists used for the front-end attribution change over time; `scripts/archive_ip_ranges.py` saves
the current lists to `data/ip_ranges/<date>/`.
