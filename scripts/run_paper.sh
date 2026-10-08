#!/bin/bash
# Every training run behind the paper, in order. Each block can be run on its own; a run appends its validation record
# to results/metrics.jsonl and its test record to results/locked_test_metrics.jsonl and writes its predictions to
# results/runs/<run id>/test_preds.npz. The analysis scripts (see README) read those files by run id, so the run ids
# below must be kept.
#
# Protocol: CKPT_SELECT=last keeps the final epoch of every observer's fixed schedule, so no labelled session beyond
# the K training sessions is used. LightGBM's number of rounds is fixed once on the CCMA K=4 validation sessions
# (scripts/select_lgbm_rounds.py -> configs/lgbm_rounds.json).
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export CKPT_SELECT=last
R=1600                                   # configs/lgbm_rounds.json
G="--dataset genai --target joint"; C="--dataset ccma --target app"
XL="--epochs 400 --d 192 --layers 8 --heads 8 --ff 384"     # the 2.4M encoder
FT="--mode ft --lr 1e-3 --lr_enc 3e-4 --epochs 60 --bs 64 --drop_head 0.3"
SEEDS="0 1 2 3 4"

ours()    { $P src/train_meta_ssl.py --tag main $FT --init weights/$1.pt --run_name "$2" "${@:3}"; }
scratch() { $P src/train_meta_ssl.py --tag main $FT --init "" --arch 192,8,8,384 --norm_ckpt weights/$1.pt --run_name "$2" "${@:3}"; }
lgbm()    { LGBM_ROUNDS=$R $P scripts/train_baselines.py --which lgbm --tag main "$@"; }
base()    { $P scripts/train_baselines.py --tag main "$@"; }          # --which dfmeta (DF-style CNN) | cnn (1D-CNN on payload bytes)
wf()      { $P src/train_wf_faithful.py --tag main --run_name "$1" "${@:2}"; }
pre()     { [ -f weights/$1.pt ] || $P src/pretrain_meta.py $XL --name "$@"; }

# ---------------------------------------------------------------- 1. pre-training (masked packet modelling, no labels)
pre meta_ssl_mpm_XL                                              # main corpus: target-app flows of the train+val sessions
pre meta_ssl_mpm_XL_ps1 --seed 1; pre meta_ssl_mpm_XL_ps2 --seed 2          # two more pre-training seeds (20-draw comparisons)
# corpora without the test sessions of the unseen-phone and temporal splits (their test sets are the same for every seed)
for dev in 8e a6; do pre meta_ssl_mpm_XL_xcd$dev --exclude genai:configs/splits/cd_${dev}_k4_s0.json; done
for dev in 2c f4 f8; do pre meta_ssl_mpm_XL_xccmacd$dev --exclude ccma:configs/splits/ccma_cd_${dev}_k4_s0.json; done
pre meta_ssl_mpm_XL_xtemporal --exclude genai:configs/splits/temporal_k4_s0.json
pre meta_ssl_mpm_XL_xccmatemporal --exclude ccma:configs/splits/ccma_temporal_k4_s0.json
# pre-training on the other campaign only
pre meta_ssl_mpm_XL_ccmaonly --datasets ccma; pre meta_ssl_mpm_XL_genaionly --datasets genai
# corpora of equal optimiser steps with background and other-campaign traffic mixed in (scripts/dilution_stats.py)
STEPS="--mask_ratio 0.4 --bs 256 --lr 0.001 --seed 0 --steps_like 28394"
pre meta_ssl_mpm_XL_dilA --datasets genai,ccma,genai_bg,ccma_bg $STEPS
pre meta_ssl_mpm_XL_dilB --datasets genai,ccma,genai_bg,ccma_bg,m2019 $STEPS
pre meta_ssl_mpm_XL_dilC --datasets genai,ccma,genai_bg,ccma_bg,m2019 --target_frac 0.25 $STEPS

# ---------------------------------------------------------------- 2. K labelled sessions per class (main comparison)
for s in $SEEDS; do for K in 1 2 4 8; do
  SG="--seed $s --split configs/splits/fewshot_k${K}_s$s.json $G"; SC="--seed $s --split configs/splits/ccma_fewshot_k${K}_s$s.json $C"
  ours    meta_ssl_mpm_XL fs${K}_sslXL_s$s $SG;            ours    meta_ssl_mpm_XL ccma_fs${K}_sslXL_s$s $SC
  scratch meta_ssl_mpm_XL fs${K}_sslscratchXL_s$s $SG;     scratch meta_ssl_mpm_XL ccma_fs${K}_sslscratchXL_s$s $SC
  for split in "fewshot_k${K}_s$s.json $G --name fs$K" "ccma_fewshot_k${K}_s$s.json $C --name ccma_fs$K"; do
    lgbm --seeds $s --split configs/splits/$split
    base --which dfmeta --seeds $s --split configs/splits/$split
    base --which cnn --seeds $s --split configs/splits/$split
  done
  for m in tiktok tf; do wf fs${K}_${m}_s$s --method $m $SG; wf ccma_fs${K}_${m}_s$s --method $m $SC; done
  wf fs${K}_cfpub_s$s      --method cf       --pre_ckpt weights/cffaith2_genai_s$s.pt $SG
  wf ccma_fs${K}_cfpub_s$s --method cf       --pre_ckpt weights/cffaith2_ccma_s$s.pt $SC
  wf fs${K}_netclr_s$s     --method netclr   --pre_ckpt weights/netaug_df_s$s.pt $SG      # NetCLR, DF backbone
  wf ccma_fs${K}_netclr_s$s --method netclr  --pre_ckpt weights/netaug_df_s$s.pt $SC
  wf fs${K}_netclrtr_s$s   --method netclrtr --pre_ckpt weights/netaug_tr_s$s.pt $SG      # NetCLR objective, our encoder
  wf ccma_fs${K}_netclrtr_s$s --method netclrtr --pre_ckpt weights/netaug_tr_s$s.pt $SC
  $P src/train_meta.py --tag main --run_name meta_fewshot${K}_base_s$s $SG                # patch Transformer
  $P src/train_meta.py --tag main --run_name ccma_fewshot${K}_base_s$s $SC
  Y="--epochs 200 --warmup 20 --layer_decay 0.75 --mfr official --pretrained weights/yatc_pretrained-model.pth"
  $P src/train_yatc.py --tag main --run_name fs${K}_yatc_pre_s$s $Y $SG
  $P src/train_yatc.py --tag main --run_name ccma_fs${K}_yatc_pre_s$s $Y $SC
  $P src/train_etbert.py --tag main --run_name fs${K}_etbert_s$s --seed $s --split configs/splits/fewshot_k${K}_s$s.json
done; done

# ---------------------------------------------------------------- 3. label age (temporal splits) and unseen phone
for s in $SEEDS; do for K in 2 4 8; do
  TG="--seed $s --split configs/splits/temporal_k${K}_s$s.json $G"; TC="--seed $s --split configs/splits/ccma_temporal_k${K}_s$s.json $C"
  ours    meta_ssl_mpm_XL_xtemporal fs${K}_sslXL_drift_s$s $TG;         ours    meta_ssl_mpm_XL_xccmatemporal ccma_fs${K}_sslXL_drift_s$s $TC
  scratch meta_ssl_mpm_XL_xtemporal fs${K}_sslscratchXL_drift_s$s $TG;  scratch meta_ssl_mpm_XL_xccmatemporal ccma_fs${K}_sslscratchXL_drift_s$s $TC
  lgbm --seeds $s --split configs/splits/temporal_k${K}_s$s.json $G --name fs${K}_drift
  lgbm --seeds $s --split configs/splits/ccma_temporal_k${K}_s$s.json $C --name ccma_fs${K}_drift
  base --which dfmeta --seeds $s --split configs/splits/temporal_k${K}_s$s.json $G --name fs${K}_drift
  base --which dfmeta --seeds $s --split configs/splits/ccma_temporal_k${K}_s$s.json $C --name ccma_fs${K}_drift
  wf fs${K}_netclr_drift_s$s      --method netclr --pre_ckpt weights/netaug_df_xtemporal_genai_s$s.pt $TG
  wf ccma_fs${K}_netclr_drift_s$s --method netclr --pre_ckpt weights/netaug_df_xtemporal_ccma_s$s.pt $TC
  wf fs${K}_cfpub_drift_s$s       --method cf --pre_ckpt weights/cffaith2_drift_genai_s$s.pt $TG
  wf ccma_fs${K}_cfpub_drift_s$s  --method cf --pre_ckpt weights/cffaith2_drift_ccma_s$s.pt $TC
done; done
for s in $SEEDS; do
  for dev in 8e a6; do
    D="--seed $s --split configs/splits/cd_${dev}_k4_s$s.json $G"
    ours meta_ssl_mpm_XL_xcd$dev cd${dev}_k4_sslXL_s$s $D; scratch meta_ssl_mpm_XL_xcd$dev cd${dev}_k4_sslscratchXL_s$s $D
    lgbm --seeds $s --split configs/splits/cd_${dev}_k4_s$s.json $G --name cd${dev}_k4
    base --which dfmeta --seeds $s --split configs/splits/cd_${dev}_k4_s$s.json $G --name cd${dev}_k4
    $P src/train_yatc.py --tag main --run_name cd${dev}_k4_yatc_pre_s$s --epochs 200 --warmup 20 --layer_decay 0.75 --mfr official --pretrained weights/yatc_pretrained-model.pth $D
  done
  for dev in 2c f4 f8; do
    D="--seed $s --split configs/splits/ccma_cd_${dev}_k4_s$s.json $C"
    ours meta_ssl_mpm_XL_xccmacd$dev ccma_cd${dev}_k4_sslXL_s$s $D; scratch meta_ssl_mpm_XL_xccmacd$dev ccma_cd${dev}_k4_sslscratchXL_s$s $D
    lgbm --seeds $s --split configs/splits/ccma_cd_${dev}_k4_s$s.json $C --name ccma_cd${dev}_k4
    base --which dfmeta --seeds $s --split configs/splits/ccma_cd_${dev}_k4_s$s.json $C --name ccma_cd${dev}_k4
    $P src/train_yatc.py --tag main --run_name ccma_cd${dev}_k4_yatc_pre_s$s --epochs 200 --warmup 20 --layer_decay 0.75 --mfr official --pretrained weights/yatc_pretrained-model.pth $D
  done
done

# ---------------------------------------------------------------- 4. defences over the first 64 packets (K = 4)
# --defense_train 0: observer trained on undefended traffic (unaware); 1: trained on defended traffic (adaptive).
for s in $SEEDS; do
  SG="--seed $s --split configs/splits/fewshot_k4_s$s.json $G"; SC="--seed $s --split configs/splits/ccma_fewshot_k4_s$s.json $C"
  # ECH: ClientHello padded (ech) and the server's first flight padded too (ech_full); the observer only ever sees ECH traffic
  for e in "ech ech" "ech_full echfull"; do set -- $e
    ours meta_ssl_mpm_XL fs4_sslXL_$2_s$s $SG --defense $1 --defense_train 1
    ours meta_ssl_mpm_XL ccma_fs4_sslXL_$2_s$s $SC --defense $1 --defense_train 1
    lgbm --seeds $s --split configs/splits/fewshot_k4_s$s.json $G --name fs4L$2 --defense $1 --defense_train 1
    lgbm --seeds $s --split configs/splits/ccma_fewshot_k4_s$s.json $C --name ccma_fs4L$2 --defense $1 --defense_train 1
  done
  for adv in 0 1; do
    for d in "pad256_jitter20 pj" "front front" "tamaraw tamaraw"; do set -- $d
      ours meta_ssl_mpm_XL fs4_sslXL_$2_adv${adv}_s$s $SG --defense $1 --defense_train $adv
      ours meta_ssl_mpm_XL ccma_fs4_sslXL_$2_adv${adv}_s$s $SC --defense $1 --defense_train $adv
      for split in "fewshot_k4_s$s.json $G --name fs4_$2_adv$adv" "ccma_fewshot_k4_s$s.json $C --name ccma_fs4_$2_adv$adv"; do
        lgbm --seeds $s --split configs/splits/$split --defense $1 --defense_train $adv
        base --which dfmeta --seeds $s --split configs/splits/$split --defense $1 --defense_train $adv
      done
    done
    for d in front tamaraw; do
      wf fs4_netclr_${d}_adv${adv}_s$s      --method netclr --pre_ckpt weights/netaug_df_s$s.pt $SG --defense $d --defense_train $adv
      wf ccma_fs4_netclr_${d}_adv${adv}_s$s --method netclr --pre_ckpt weights/netaug_df_s$s.pt $SC --defense $d --defense_train $adv
      wf fs4_cfpub_${d}_adv${adv}_s$s       --method cf --pre_ckpt weights/cffaith2_genai_${d}_adv${adv}_s$s.pt $SG --defense $d --defense_train $adv
      wf ccma_fs4_cfpub_${d}_adv${adv}_s$s  --method cf --pre_ckpt weights/cffaith2_ccma_${d}_adv${adv}_s$s.pt $SC --defense $d --defense_train $adv
    done
  done
  # adversarial shaping: PGD on padding and delay against the encoder (white box) and through a surrogate; adversarial training
  ours meta_ssl_mpm_XL fs4_sslXL_advtrain_s$s $SG --adv_train 1; ours meta_ssl_mpm_XL ccma_fs4_sslXL_advtrain_s$s $SC --adv_train 1
  for pre_ in fs4 ccma_fs4; do for b in "64 10" "256 50" "1460 200"; do set -- $b
    for att in ${pre_}_sslXL_s$s ${pre_}_sslXL_advtrain_s$s; do
      $P scripts/adv_eval.py --tag adv_shaping --steps 20 --run $att --budget_bytes $1 --budget_ms $2
      $P scripts/adv_eval.py --tag adv_shaping --steps 20 --run $att --surrogate ${pre_}_sslscratchXL_s$s --budget_bytes $1 --budget_ms $2
    done
  done; done
done
$P scripts/robust_assistant_knn.py --stage run          # nearest neighbour on the first ten packets, every condition above
$P scripts/adv_transfer.py                              # does the shaping transfer to LightGBM and the nearest neighbour?
$P scripts/whole_connection_defences.py --tamaraw_published 100,1000        # FRONT / Tamaraw over whole connections

# ---------------------------------------------------------------- 5. open world
for K in 2 4 8; do for bg in 0 1; do
  O="--K $K --bg_class $bg --scorer ens --attackers ours,scratch,lgbm --lgbm_rounds $R --protocol budget --tag open_world"
  $P scripts/open_world_reject.py $G --n_mon 3 --n_proxy 1 $O        # 3 of the 6 GenAI classes monitored, seeds 0-4
  $P scripts/open_world_reject.py $C $O                              # 5 of the 9 CCMA apps monitored, seeds 0-4
done; done

# ---------------------------------------------------------------- 6. pre-training corpora of equal compute (K = 1, 2, 4)
for v in dilA dilB dilC; do for K in 1 2 4; do for s in $SEEDS; do
  ours meta_ssl_mpm_XL_$v fs${K}_sslXL${v}_s$s --seed $s --split configs/splits/fewshot_k${K}_s$s.json $G
  ours meta_ssl_mpm_XL_$v ccma_fs${K}_sslXL${v}_s$s --seed $s --split configs/splits/ccma_fewshot_k${K}_s$s.json $C
done; done; done

# ---------------------------------------------------------------- 7. the six primary comparisons over 20 draws
# Draws 0-4 are the runs of block 2; scripts/make_splits_draws20.py writes the K-session draws 5-19. Draw r uses the
# pre-training seed r mod 3, so draws 0 and 3 reuse the block-2 encoder runs. scripts/draws20_primary.py computes the
# paired t-tests with Holm correction and the two-level bootstrap.
$P scripts/make_splits_draws20.py
for r in $(seq 1 19); do
  p=$((r % 3)); W=meta_ssl_mpm_XL; [ $p -ne 0 ] && W=meta_ssl_mpm_XL_ps$p
  for K in 1 2; do
    SG="--seed $r --split configs/splits/fewshot_k${K}_s$r.json $G"; SC="--seed $r --split configs/splits/ccma_fewshot_k${K}_s$r.json $C"
    if [ $r -ne 3 ]; then ours $W fs${K}_sslXL_d${r}_p$p $SG; ours $W ccma_fs${K}_sslXL_d${r}_p$p $SC; fi
    if [ $r -ge 5 ]; then
      scratch meta_ssl_mpm_XL fs${K}_sslscratchXL_d$r $SG; scratch meta_ssl_mpm_XL ccma_fs${K}_sslscratchXL_d$r $SC
      lgbm --seeds $r --split configs/splits/fewshot_k${K}_s$r.json $G --name fs${K}_draws20
      lgbm --seeds $r --split configs/splits/ccma_fewshot_k${K}_s$r.json $C --name ccma_fs${K}_draws20
    fi
  done
done
$P scripts/draws20_primary.py
