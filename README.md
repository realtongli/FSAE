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

No traffic was captured for this work. All three datasets are public (two campaigns and a third evaluation set from the
same laboratory); they are not redistributed here.

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
  appact_wrap.py           runs the trainers unchanged on the third evaluation set, records under results/appact/
paper/                   generators of the LaTeX tables and PDF figures
configs/splits/          session-level splits (K-session draws, temporal, unseen phone, standard 60/20/20;
                         appact_* for the third evaluation set)
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
| MIRAGE-AppAct-2024 (third evaluation set) | keep the archive as `data/MIRAGE-AppAct-2024.zip` |

**Third-party code and weights**, only needed for the corresponding reference observers (the YaTC checkout is also
needed by the nearest neighbour, `scripts/trivial_baselines.py`, which shares a metric helper with `src/train_yatc.py`,
and by the analyses that import it):

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
python scripts/build_dataset_appact.py     # MIRAGE-AppAct-2024 apps absent from CCMA -> data/derived/appact_* (after CCMA)
python scripts/dataset_stats.py            # sessions, connections, transports and split sizes of the three datasets

# 2. splits are shipped in configs/splits/; they were written by scripts/make_splits*.py
#    (scripts/make_splits_appact.py re-derives the third set's splits and checks them against the shipped files)

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
and unseen-phone splits, defences, open world, the 20-draw primary comparisons, and in its last block the third
evaluation set through `scripts/appact_wrap.py`).

### Protocol

* **Labels are counted in capture sessions.** Every split is by session, never by connection. A K-session split holds K
  training sessions per class; its test sessions are those of the standard split and are never used for training,
  normalisation or selection. Pre-training corpora exclude the test sessions of the split they are evaluated on.
* **No labelled session beyond K.** With `CKPT_SELECT=last` every observer trains for a fixed schedule and keeps its
  final model. The encoder's recipe and LightGBM's number of rounds were fixed once on validation sessions of CCMA and
  are used unchanged everywhere, the third evaluation set included.
* **Run records.** A run writes `results/runs/<run id>/test_preds.npz` (test indices, labels, predictions) and appends
  one line to `results/metrics.jsonl` (validation) and `results/locked_test_metrics.jsonl` (test). The analysis scripts
  read these files by run id. Runs of the third evaluation set write `results/appact/runs/<run id>/test_preds.npz`,
  `results/appact/metrics_appact.jsonl` and `results/appact/locked_test_metrics_appact.jsonl` instead.

## From runs to the paper

Some tables of the paper show part of a generated table or combine the outputs of several scripts; every number comes
from the scripts listed here.

| in the paper | scripts |
|---|---|
| Table 1 and the appendix's dataset numbers (connections per session, transports, split sizes, phones of the third set) | `scripts/dataset_stats.py` |
| Table 2: assistant, modality and session-vote accuracy of the main observers | `scripts/threat_stats.py`, `paper/make_table_threat.py`; server-name lookup row: `scripts/name_lookup_row.py` |
| Tables 3 and 7, Figure 3: all observers, K = 1, 2, 4, 8, GenAI and CCMA | `scripts/summarize_sota_fewshot.py`, `scripts/boot2.py`, `paper/make_figures_main.py` |
| Table 4, the third set's columns of Table 9 and its session votes | `scripts/build_dataset_appact.py`, `scripts/make_splits_appact.py`, `scripts/appact_leakcheck.py`, `scripts/appact_wrap.py`, `scripts/appact_primary.py` (block 8 of `scripts/run_paper.sh`) |
| Table 5: ablation (no pre-training, NetCLR's objective on our encoder), other-campaign corpus | `scripts/summarize_sota_fewshot.py`, `scripts/boot2.py`, `scripts/draws20_primary.py`, `paper/make_tables_extra_v4.py` |
| appendix ablation: encoder size and a linear probe on the frozen encoder | `scripts/encoder_ablation.py` |
| primary comparisons over 20 draws, three pre-training seeds | `scripts/make_splits_draws20.py`, `scripts/draws20_primary.py`; third set: `scripts/appact_primary.py` |
| device windows without app attribution | `scripts/window_assistant.py`, `scripts/window_encoder_bg.py` |
| where the signal is (handshake, later packets, server names, classes, app releases) | `scripts/leak_source.py`, `scripts/construct_validity.py`, `scripts/google_app_check.py`, `scripts/session_extras.py` |
| Tables 6, 8 and 9: label age and unseen phone | `scripts/robust_assistant.py`, `scripts/robust_assistant_knn.py`, `scripts/drift_gaps.py`, `paper/make_tab_robust3.py`, `paper/make_tables_def_drift.py`, `paper/make_tables_extra_v4.py` |
| server addresses: lookup observer (Addr column of Table 6), label age, unseen phone, front-end operators, apps behind one operator | `scripts/ip_baseline.py`, `scripts/addr_robust.py`, `scripts/cdn_attribution.py`, `scripts/archive_ip_ranges.py`, `scripts/cdn_attribution_archived.py`, `scripts/operator_confound.py` |
| Table 12: pre-training corpora of equal compute | `scripts/dilution_stats.py` |
| LightGBM's number of rounds, fixed on validation sessions | `scripts/select_lgbm_rounds.py` |
| Table 10: open world, per connection and per device window | `scripts/open_world_reject.py`, `scripts/build_bg_ownservice_mask.py`, `scripts/ow_rescore.py`, `scripts/ow_windows_exact.py`, `paper/make_table_openworld.py` |
| Tables 11 and 13, Figure 4: defences over the first 64 packets and their cost | `scripts/wf_defenses.py`, `scripts/defence_stats.py`, `paper/make_tab_robust3.py`, `paper/make_tables_def_drift.py`, `paper/make_figures_main.py` |
| adversarial shaping and its transfer | `scripts/adv_eval.py`, `scripts/adv_transfer.py` |
| defences over whole connections, flow-record observer, FRONT with the published budgets | `scripts/wf_defenses_whole.py`, `scripts/whole_connection_defences.py`, `scripts/front_whole_published.py` |

The operators' IP-range lists used for the front-end attribution change over time; `scripts/archive_ip_ranges.py` saves
the current lists to `data/ip_ranges/<date>/`.
