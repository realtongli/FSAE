# Ahead of Wider ECH Adoption: Measuring AI Assistant Identifiability from Encrypted Mobile Traffic

Code, session-level splits and analysis scripts for an anonymous submission.

The paper measures how much packet metadata reveals about which AI assistant a phone accesses once server names are
hidden. A passive observer at the device's gateway keeps the first 64 packets of each connection and uses only their
direction, IP size, payload size and inter-arrival time. Server names, destination addresses, ports, absolute
timestamps and payload bytes are excluded from its features. Labelled data are counted in complete 15-minute capture
sessions, and training and test sessions are always kept apart. The measurements relate

* the labelled-session budget K to the identification it buys;
* the labels to their reuse in device windows with unmonitored background traffic, on a held-out phone and in later
  captures;
* each defence's added bytes and delay to the identification it removes, including against observers that adapt to it.

The released traces are analysed as captured, and handshake-length equalisation is simulated offline on them. Nothing
here measures deployed ECH traffic.

## Contents

* **Observers**, run on the same labelled sessions under one protocol: LightGBM on packet statistics, a nearest
  neighbour on the first ten packets, a patch Transformer, a Transformer encoder pre-trained by masked packet modelling
  on unlabelled connections and fine-tuned on the K labelled sessions, and five website-fingerprinting attacks ported
  to the 64-packet metadata (Deep Fingerprinting, Tik-Tok, Triplet Fingerprinting, Contrastive Fingerprinting, NetCLR).
* **References outside the observer model**: byte-level classifiers that read the plaintext handshake (the dataset
  paper's 1D CNN, YaTC, ET-BERT), a server-name lookup and a server-address lookup.
* **Defence simulations** with their byte and delay costs: offline ClientHello padding, alone and with padding of the
  server's first flight; padding with jitter; FRONT; Tamaraw; white-box adversarial shaping and its transfer to other
  observers; FRONT and Tamaraw over whole connections against an observer of flow records.
* Every **split** of the paper (`configs/splits/`) and the scripts that turn run records into its tables and figures.

No traffic was captured for this work. The three datasets are public releases of the MIRAGE project and are not
redistributed here.

| name in the paper | release | classes | sessions (train/val/test) | sessions captured |
|---|---|---|---|---|
| GenAI | MIRAGE-GenAI-2025 | ChatGPT, Copilot and the Google app hosting Gemini, each with a text and an image activity | 233 (137/48/48) | Oct 2024 – Jan 2025 |
| CCMA | MIRAGE-COVID-CCMA-2022 | 9 communication apps | 765 (457/154/154) | Oct 2019 – Dec 2021 |
| AppAct | MIRAGE-AppAct-2024, the 9 apps absent from CCMA with at least 50 sessions | 9 apps | 801 (475/163/163) | May 2021 – Mar 2023 |

Capture dates are those of the session files listed in `configs/splits/`. The image activity is the dataset's `multi`
activity (class directories `*_multi`).

## Layout

```
src/                     model and training code
  pretrain_meta.py         masked packet modelling on unlabelled connections        -> weights/<name>.pt
  train_meta_ssl.py        fine-tuning on K labelled sessions (also: no pre-training, linear probe)
  train_meta.py            input pipeline shared by all metadata observers; patch-Transformer reference
  train_wf_faithful.py     DF, Tik-Tok, TF, CF, NetCLR ported to the 64x4 metadata sequence (with train_wf_baselines.py)
  train_yatc.py, train_etbert.py, carriers/      byte-level references built on the original authors' code
  data_mfr.py              access to the derived arrays and the session splits
scripts/                 data building, reference observers, defences, analyses
  run_paper.sh             every training run of the paper, in order, with the exact flags
  appact_wrap.py           runs the trainers unchanged on AppAct, records under results/appact/
paper/                   generators of the LaTeX tables and PDF figures
configs/splits/          session-level splits (K-session draws, temporal, unseen phone, standard 60/20/20;
                         appact_* for AppAct)
configs/lgbm_rounds.json LightGBM rounds, fixed once on validation sessions of CCMA
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
| MIRAGE-AppAct-2024 | keep the archive as `data/MIRAGE-AppAct-2024.zip` |

**Third-party code and weights.** Check out Medformer and YaTC before running anything. `src/train_meta.py`, which
the pre-training, the encoder, LightGBM and most analyses import, loads Medformer's model file and imports
`src/train_yatc.py`, which loads a helper from YaTC; the nearest neighbour (`scripts/trivial_baselines.py`) imports
`src/train_yatc.py` too. The ET-BERT checkout and the released weights are needed only for the corresponding references.

| code | checkout | needed by | weights |
|---|---|---|---|
| Medformer | `third_party/Medformer` (DL4mHealth/Medformer, commit `891f65b`) | every metadata observer, the pre-training and most analyses | none |
| YaTC | `third_party/YaTC` (NSSL-SJTU/YaTC, commit `a220b75`) | every metadata observer, the pre-training and most analyses | YaTC reference only: released pre-trained model as `weights/yatc_pretrained-model.pth` |
| ET-BERT | `third_party/ET-BERT` (linwhitehat/ET-BERT, commit `d594706`) | ET-BERT reference only | released pre-trained model as `weights/etbert_pretrained_model.bin` |

## Pipeline

```bash
# 1. derived arrays: per connection the first 64 packets (direction, IP bytes, payload bytes, inter-arrival time),
#    payload windows for the byte-level references, labels, capture session and phone
python scripts/build_dataset.py            # MIRAGE-GenAI-2025  -> data/derived/genai_*
python scripts/build_dataset_ccma.py       # MIRAGE-COVID-CCMA-2022 -> data/derived/ccma_*
python scripts/build_background.py         # connections of the same sessions that belong to no target app
python scripts/build_m2019_meta.py         # optional: MIRAGE-2019 metadata for the mixed pre-training corpora
python scripts/build_etbert_tokens.py      # optional: tokens for ET-BERT
python scripts/build_dataset_appact.py     # AppAct: MIRAGE-AppAct-2024 apps absent from CCMA -> data/derived/appact_* (after CCMA)
python scripts/dataset_stats.py            # sessions, connections, transports and split sizes of the three datasets

# 2. splits are shipped in configs/splits/; they were written by scripts/make_splits*.py
#    (scripts/make_splits_appact.py re-derives the AppAct splits and checks them against the shipped files)

# 3. pre-train the encoder (labels unused; about 42 minutes on one RTX 5070 Ti)
python src/pretrain_meta.py --epochs 400 --d 192 --layers 8 --heads 8 --ff 384 --name meta_ssl_mpm_XL

# 4. fine-tune on K = 4 labelled sessions per class and test on unseen sessions (under two minutes)
CKPT_SELECT=last python src/train_meta_ssl.py --run_name fs4_sslXL_s0 --tag main --seed 0 \
    --split configs/splits/fewshot_k4_s0.json --dataset genai --target joint \
    --init weights/meta_ssl_mpm_XL.pt --mode ft --lr 1e-3 --lr_enc 3e-4 --epochs 60 --bs 64 --drop_head 0.3

# LightGBM on the same sessions
CKPT_SELECT=last LGBM_ROUNDS=1600 python scripts/train_baselines.py --which lgbm --seeds 0 \
    --split configs/splits/fewshot_k4_s0.json --dataset genai --target joint --name fs4 --tag main

# the nearest neighbour on the first ten packets (training-free): both campaigns, K = 1, 2, 4, 8, draws 0-4
python scripts/trivial_baselines.py
```

`scripts/run_paper.sh` lists every training run behind the paper's tables and figures in this form: all K, five
K-session draws, both campaigns, temporal and unseen-phone splits, defences, open world, the 20-draw primary
comparisons, and in its last block AppAct through `scripts/appact_wrap.py`. It also runs the training-free nearest
neighbour: `scripts/trivial_baselines.py` on the K-session draws and `scripts/robust_assistant_knn.py --stage run` on the
drift, unseen-phone and defence settings.

### Protocol

* **Labels are counted in capture sessions.** Every split is by session, never by connection. A K-session split holds K
  training sessions per class; its test sessions are those of the standard split and are never used for training,
  normalisation or selection. Pre-training corpora exclude the test sessions of the split they are evaluated on.
* **No labelled session beyond K.** With `CKPT_SELECT=last` every observer trains for a fixed schedule and keeps its
  final model. The encoder's recipe and LightGBM's number of rounds were fixed once on validation sessions of CCMA and
  are used unchanged everywhere, AppAct included.
* **Run records.** A run writes `results/runs/<run id>/test_preds.npz` (test indices, labels, predictions) and appends
  one line to `results/metrics.jsonl` (validation) and `results/locked_test_metrics.jsonl` (test). The analysis scripts
  read these files by run id. AppAct runs write `results/appact/runs/<run id>/test_preds.npz`,
  `results/appact/metrics_appact.jsonl` and `results/appact/locked_test_metrics_appact.jsonl` instead.

## From runs to the paper

Tables, figures and in-text numbers, with the scripts that compute them.

| in the paper | scripts |
|---|---|
| Table 1: assistant identification on GenAI, per connection (K = 1, 2, 4, 8) and per session (K = 1, 4) | `scripts/threat_stats.py`, `paper/make_table_threat.py`; server-name lookup row: `scripts/name_lookup_row.py` |
| Table 2: sessions named under drift and on an unseen phone | `scripts/robust_assistant.py`, `scripts/robust_assistant_knn.py`, `paper/make_tab_robust3.py`; Addr column: `scripts/addr_robust.py` |
| Table 3 and the dataset numbers of Appendix A | `scripts/dataset_stats.py` |
| Table 4 and Figure 3: every observer, K = 1, 2, 4, 8, GenAI and CCMA | `scripts/summarize_sota_fewshot.py`, `scripts/boot2.py`, `paper/make_figures_main.py` |
| Table 5: temporal macro-F1 | `paper/make_tables_def_drift.py`, `scripts/robust_assistant_knn.py`, `scripts/drift_gaps.py` |
| Table 6: unseen phone | `paper/make_tables_extra_v4.py`, `scripts/robust_assistant_knn.py`; AppAct columns: `scripts/appact_primary.py` |
| Table 7: open-world PR-AUC | `scripts/open_world_reject.py`, `scripts/build_bg_ownservice_mask.py`, `scripts/ow_rescore.py`, `paper/make_table_openworld.py` |
| Table 8: identification under defences over the first 64 packets | `scripts/wf_defenses.py`, `scripts/defence_stats.py`, `scripts/robust_assistant.py`, `scripts/robust_assistant_knn.py`, `paper/make_tab_robust3.py` |
| Table 9: the encoder by pre-training corpus | `scripts/dilution_stats.py` |
| Table 10: FRONT and Tamaraw over the first 64 packets, every observer | `paper/make_tables_def_drift.py`, `scripts/robust_assistant_knn.py` |
| Figure 4: AppAct | runs of block 8 of `scripts/run_paper.sh` (checks: `scripts/appact_leakcheck.py`); numbers: `scripts/appact_primary.py` |
| Figure 5: ablation (no pre-training, NetCLR's objective on the same encoder) | numbers: `scripts/summarize_sota_fewshot.py`; 20-draw comparisons: `scripts/draws20_primary.py` |
| Figure 6: session identification against defence cost | `paper/make_figures_main.py`, from the outputs of `scripts/robust_assistant.py`, `scripts/whole_connection_defences.py` and `scripts/adv_transfer.py` |
| Section 4.3: session votes on GenAI and CCMA | `scripts/threat_stats.py`, `scripts/session_extras.py` |
| Section 4.3: server names and the first ten packets, packets 11–64 | `scripts/leak_source.py`, `scripts/construct_validity.py` |
| Section 4.3: device windows with background traffic, and with a background class | `scripts/window_assistant.py`, `scripts/window_encoder_bg.py` |
| Section 4.4 and Appendix A.1: primary comparisons over 20 draws, three pre-training seeds | `scripts/make_splits_draws20.py`, `scripts/draws20_primary.py`; AppAct: `scripts/appact_primary.py` |
| Section 4.5: pre-training on the other campaign only | `paper/make_tables_extra_v4.py`, `scripts/boot2.py` |
| Section 4.6: label age, app releases in the later sessions | `scripts/drift_gaps.py`, `scripts/session_extras.py`, `scripts/google_app_check.py` |
| Section 4.6: server addresses (lookup, front-end operators, apps behind one operator) | `scripts/ip_baseline.py`, `scripts/addr_robust.py`, `scripts/cdn_attribution.py`, `scripts/archive_ip_ranges.py`, `scripts/cdn_attribution_archived.py`, `scripts/operator_confound.py` |
| Section 4.7: open-world thresholds and device windows | `scripts/ow_rescore.py`, `scripts/ow_windows_exact.py` |
| Section 4.8: padding sizes, defence costs and delays | `scripts/defence_stats.py`, `scripts/session_extras.py` |
| Section 4.8: adversarial shaping and its transfer | `scripts/adv_eval.py`, `scripts/adv_transfer.py` |
| Section 4.8: whole connections and flow records | `scripts/wf_defenses_whole.py`, `scripts/whole_connection_defences.py`, `scripts/front_whole_published.py` |
| Appendix B: LightGBM's number of rounds | `scripts/select_lgbm_rounds.py` |
| Appendix C: encoder size and a linear probe on the frozen encoder | `scripts/encoder_ablation.py` |

Notes.

* The generators write working versions of the tables (`paper/tables/`) and figures (`paper/figs/`); the manuscript
  typesets them with its own captions and layout. Older names survive in run ids, docstrings and these working
  captions: *attacker* for observer and *modality* for activity.
* The uniform-random rows of Tables 4–7 are the mean and sample standard deviation over 500 draws of uniformly random
  predictions on each test set (Appendix C).
* The abundant-label reference of Appendix C trains the observers on the standard split
  (`configs/splits/session_split_s2026.json`, `configs/splits/ccma_session_split_s2026.json`) with `CKPT_SELECT=val`:
  the encoder keeps its best validation epoch and LightGBM stops early on the validation sessions, over three seeds.
* The operators' IP-range lists used for the front-end attribution change over time; `scripts/archive_ip_ranges.py`
  saves the current lists to `data/ip_ranges/<date>/`.
