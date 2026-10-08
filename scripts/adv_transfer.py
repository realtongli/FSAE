"""Adversarial shaping computed against the pre-trained encoder, scored by LightGBM and the 1-NN trained on unshaped traffic.

QUESTION. scripts/adv_eval.py (tag adv_shaping) adds padding and delay chosen by PGD through a differentiable observer
(white-box: the pre-trained encoder itself; transfer: the same encoder without pre-training as a surrogate) and reports
only the encoder's macro-F1. Does the same shaped test traffic also defeat the two cheap observers of the paper,
LightGBM on packet statistics and the 1-NN on the first ten packets, trained on the unshaped K labelled sessions?
This script scores every observer on the test traffic shaped against the encoder, against its surrogate and against
the adversarially trained encoder, and on a shuffled control of the white-box shaping.

RULES.
 1. Shaping reproduces adv_eval.py exactly: its load_attacker, raw_to_feat and pgd are imported, not copied. Grid = the
    adv_shaping grid: per-packet budgets (64 B, 10 ms), (256 B, 50 ms), (1460 B, 200 ms); 20 PGD steps of budget/8;
    padding only on payload-bearing packets and capped at 1460 B of payload, delay on every packet, direction and packet
    count unchanged; batches of 256 test connections in test-index order; first 64 packets. K=4 splits
    configs/splits/fewshot_k4_s{0..4}.json (GenAI, 6 app x modality classes, y_joint) and ccma_fewshot_k4_s{0..4}.json
    (CCMA, 9 apps, y_app), as the runs were trained. Checkpoints results/runs/<run>/best.pt (= the final epoch of the
    fixed 60-epoch fine-tune, select='last').
    Shaping sources (the model whose gradients PGD follows):
      wb           white-box: the evaluated pre-trained encoder {ccma_}fs4_sslXL_s{s}
      transfer     the surrogate {ccma_}fs4_sslscratchXL_s{s} (same 2.4M encoder, no pre-training, same K sessions)
      wb_advtrain  reference: white-box against the adversarially trained encoder {ccma_}fs4_sslXL_advtrain_s{s}
      wb_shuffled  control: for each connection, the byte additions that 'wb' made on its payload-bearing
                   packets with room below 1460 B of payload are randomly permuted among those same packets and then
                   clipped to each destination packet's room; its delay additions are permuted among all its packets.
                   Same kind and nearly the same amount of shaping without the targeting; RandomState(20260927 +
                   1000*[genai 0, ccma 1] + 10*seed + budget index). Its achieved overhead is reported (clipping can
                   only lower the bytes).
    The shaped test connections (IP bytes, payload bytes, IAT in s; float32; direction and validity are the unshaped
    ones) are saved to results/adv_transfer_shaped.npz (key '<ds>|s<seed>|<source>|<bytes>_<ms>'), and every score
    below is computed from the reloaded file.
 2. Observers, all trained on the unshaped connections of the K labelled training sessions of the split only; nothing
    is refit, tuned or selected on shaped traffic, and the validation sessions are not used:
      encoder          {ccma_}fs4_sslXL_s{s}, loaded by adv_eval.load_attacker (the attacker of the adv_shaping records)
      lgbm             the main run {ccma_}fs4_lgbm_meta64_s{s} (scripts/train_baselines.py --which lgbm with
                       CKPT_SELECT=last LGBM_ROUNDS=1600): train_baselines.stat_features on all flows, standardisation
                       fitted on the training rows, LGBMClassifier(n_estimators=1600, learning_rate=0.03, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=seed, verbose=-1) fitted
                       on the training rows; shaped test connections pass through the same stat_features and the same
                       training-row standardisation
      knn10            {ccma_}fs4_knn10_s{s}: trivial_baselines.feats_knn (first 10 packets) and the verbatim nn1 (L1,
                       ties to the lowest training index), imported from scripts/robust_assistant_knn.py
      encoder_advtrain {ccma_}fs4_sslXL_advtrain_s{s} and surrogate {ccma_}fs4_sslscratchXL_s{s}: scored as references
    Every observer scores every traffic set: none (unshaped), wb, transfer, wb_advtrain, wb_shuffled.
 3. Metrics: the rule of scripts/robust_assistant.py, unchanged. Task macro-F1 (GenAI 6 classes, CCMA 9 apps; sklearn
    macro over the labels present in truth or prediction, zero_division=0). GenAI assistant F1: macro-F1 of pred//2
    against y//2 over all test connections. Session vote: the assistant (GenAI, pred//2) or app (CCMA) of a test
    session is the majority vote over its test connections (np.bincount(...).argmax(), ties to the lowest index),
    compared with the session's majority true label; accuracy over the test sessions (48 GenAI, 154 CCMA). Every
    number per seed (0-4) and summarised as mean, population sd, min and max over seeds.
    Overheads per source and budget: (a) adv_eval's own statistic (mean over the 256-connection batches of the per-batch
    mean bytes added per payload-bearing packet and ms added per packet), used for the reproduction check and quoted as
    in the paper; (b) the same pooled over all test packets; (c) added bytes / IP bytes of the observed 64-packet prefix;
    (d) added delay summed over a connection's prefix (mean and median over connections) and pooled added delay / sum of
    the original IATs of the prefix.
    Chance references: uniform guess (session vote 100/C; connection macro-F1 approximated by mean_c 2 p_c (1/C) /
    (p_c + 1/C) over the test prevalence p_c, as robust_assistant.uniform_f1) and the most frequent class of the test
    sessions (session vote).
    Retained margin per seed: R = (F_shaped - F_chance) / (F_clean - F_chance), for the task F1, the GenAI assistant F1
    and the session vote, with the uniform guess as chance. Wording rule: an observer keeps "most of the leak" at a
    budget when its mean R > 0.5, and shaping "removes most of the leak" for it when mean R <= 0.5; both the task-F1 R
    and the session-vote R are reported and the verdict names which one.
 4. Reproduction checks and their tolerances:
      R1 the encoder's and adv-trained encoder's clean macro-F1 equal test_f1_clean of every adv_shaping record (|d| < 1e-6);
      R2 the target's macro-F1 on the reloaded shaped features equals test_f1_adv of every adv_shaping record within
         0.5 points (PGD on the GPU is not bit-deterministic, so a few sign steps may differ); the maximum |d| is
         reported, and so is the difference between the in-memory F1 (computed during PGD, as adv_eval does) and the F1
         on the reloaded file (expected 0);
      R3 overhead (a) within 2% (relative) of mean_pad_bytes and mean_delay_ms of the record;
      R4 LightGBM's and the 1-NN's unshaped test predictions equal results/runs/{ccma_}fs4_lgbm_meta64_s{s} and
         {ccma_}fs4_knn10_s{s}/test_preds.npz (number of differing predictions, expected 0), and their macro-F1 equals
         the last non-smoke record of the run in results/locked_test_metrics.jsonl;
      R5 the encoder's unshaped predictions through adv_eval's feature path vs results/runs/{ccma_}fs4_sslXL_s{s}/
         test_preds.npz (number of differing predictions).
    A failed check is flagged in the JSON; no number is dropped or recomputed selectively because of it.
 5. Two-level paired bootstrap (scripts/boot2.Cell unchanged: B=4000, RandomState(20260926), draws then test sessions),
    one cell per campaign on the task macro-F1 and, for GenAI, one on the assistant F1 (the test sessions are the same
    for every draw, asserted). Comparisons, for every budget:
      C1 lgbm@S - encoder@S and knn10@S - encoder@S for S in {wb, transfer}
      C2 obs@S - obs@none for obs in {encoder, lgbm, knn10}, S in {wb, transfer}
      C3 obs@wb - obs@wb_shuffled for obs in {encoder, lgbm, knn10} (does the targeting matter beyond its cost?)
      C4 obs@wb_advtrain - obs@none for obs in {encoder_advtrain, lgbm, knn10}
    Session votes get per-seed paired differences and seed counts only. No multiple-comparison correction; the
    intervals are descriptive.
 6. Every number above is reported, favourable or not.

Usage: python scripts/adv_transfer.py [--stage all|shape|score]   (shape = GPU PGD + npz; score = observers + JSON)
Writes results/adv_transfer_shaped.npz (shape) and results/adv_transfer.json (score). Reads, never writes, every other
project file.
"""
import argparse, json, os, sys, time
import numpy as np, torch
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
import adv_eval                                   # load_attacker, raw_to_feat, pgd (the adv_shaping machinery)
from train_baselines import stat_features         # LightGBM feature map of the paper
from trivial_baselines import feats_knn           # 1-NN feature map of the paper
from robust_assistant_knn import nn1              # verbatim nn1 of trivial_baselines
import boot2

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [(64.0, 10.0), (256.0, 50.0), (1460.0, 200.0)]
STEPS, BS, FIRST_K = 20, 256, 64
CAMPAIGNS = {'genai': dict(pre='', split='fewshot_k4', target='joint', dsi=0),
             'ccma': dict(pre='ccma_', split='ccma_fewshot_k4', target='app', dsi=1)}
RUNS = {'encoder': '{pre}fs4_sslXL_s{s}', 'encoder_advtrain': '{pre}fs4_sslXL_advtrain_s{s}', 'surrogate': '{pre}fs4_sslscratchXL_s{s}'}
SOURCES = {'wb': 'encoder', 'transfer': 'surrogate', 'wb_advtrain': 'encoder_advtrain'}   # PGD sources (model followed)
TRAFFIC = ['none', 'wb', 'transfer', 'wb_advtrain', 'wb_shuffled']
OBSERVERS = ['encoder', 'lgbm', 'knn10', 'encoder_advtrain', 'surrogate']
TAG = 'adv_shaping'
SHAPED_NPZ = os.path.join(ROOT, 'results', 'adv_transfer_shaped.npz')
OUT_JSON = os.path.join(ROOT, 'results', 'adv_transfer.json')


def bkey(bb, bm): return f'{int(bb)}_{int(bm)}'


def akey(ds, s, src, bb, bm): return f'{ds}|s{s}|{src}|{bkey(bb, bm)}'


def mf1(y, p): return 100.0 * f1_score(y, p, average='macro', zero_division=0)


def uniform_f1(y, C):  # robust_assistant.uniform_f1 (not imported: importing that script would execute it)
    p = np.bincount(y, minlength=C) / len(y); p = p[p > 0]
    return 100 * float(np.mean(2 * p / C / (p + 1 / C)))


def summ(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std()), min=float(v.min()), max=float(v.max()), per_seed=[float(x) for x in v])


def rnd(o, nd=4):
    if isinstance(o, float): return round(o, nd)
    if isinstance(o, dict): return {k: rnd(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v, nd) for v in o]
    return o


def split_idx(D, c, s):
    return D.split_indices(os.path.join(ROOT, 'configs', 'splits', f"{c['split']}_s{s}.json"))


def labels(D, c):
    y = D.y_joint if c['target'] == 'joint' else D.y_app
    NC = D.n_joint if c['target'] == 'joint' else D.n_app
    return y, NC


# ------------------------------------------------------------------------------------------------ stage 1: shaping
def stage_shape():
    dev = torch.device('cuda'); arrays, log = {}, {}; t0 = time.time()
    for ds, c in CAMPAIGNS.items():
        D = GenAIData(ROOT, prefix=ds); y_all, NC = labels(D, c)
        raw_all = torch.from_numpy(D.meta[:, :FIRST_K].astype(np.float32)); y_t = torch.from_numpy(y_all)
        for s in SEEDS:
            idx, _ = split_idx(D, c, s); te = idx['test']
            if f'{ds}|test_idx' in arrays: assert np.array_equal(arrays[f'{ds}|test_idx'], te), 'test sessions differ across draws'
            else: arrays[f'{ds}|test_idx'] = te
            M = {}
            for k, fmt in RUNS.items():
                rid = fmt.format(pre=c['pre'], s=s); fn, mu, sd, a_ = adv_eval.load_attacker(rid, dev, NC)
                assert a_.first_k == FIRST_K and getattr(a_, 'select', 'val') == 'last' and a_.epochs == 60, rid
                assert os.path.normpath(a_.split) == os.path.normpath(os.path.join('configs', 'splits', f"{c['split']}_s{s}.json")), rid
                M[k] = (fn, mu, sd, rid)
            # clean predictions of the two record targets, in adv_eval's feature path and batching
            clean = {k: [] for k in ('encoder', 'encoder_advtrain')}
            for b in range(0, len(te), BS):
                raw = raw_all[te[b:b + BS]]; v = (raw[..., 1] > 0).to(dev)
                with torch.no_grad():
                    for k in clean:
                        fn, mu, sd, _ = M[k]; clean[k].append(fn(adv_eval.raw_to_feat(raw.to(dev), mu, sd, FIRST_K), v).argmax(1).cpu().numpy())
            log[f'{ds}|s{s}|clean_f1_mem'] = {k: mf1(y_all[te], np.concatenate(p)) for k, p in clean.items()}
            for bb, bm in BUDGETS:
                for src, mk in SOURCES.items():
                    fn_s, mu_s, sd_s, rid_s = M[mk]; shaped, pbs, pms = [], [], []; mem = {k: [] for k in ('encoder', 'encoder_advtrain')}
                    for b in range(0, len(te), BS):
                        bi = te[b:b + BS]; raw = raw_all[bi]; yb = y_t[bi]; v = (raw[..., 1] > 0).to(dev)
                        r_adv, pb, pm = adv_eval.pgd(fn_s, raw, yb, mu_s, sd_s, FIRST_K, bb, bm, STEPS, dev)
                        with torch.no_grad():
                            for k in mem:
                                fn, mu, sd, _ = M[k]; mem[k].append(fn(adv_eval.raw_to_feat(r_adv, mu, sd, FIRST_K), v).argmax(1).cpu().numpy())
                        shaped.append(r_adv[..., 1:].cpu().numpy().astype(np.float32)); pbs.append(pb); pms.append(pm)
                    key = akey(ds, s, src, bb, bm); arrays[key] = np.concatenate(shaped)
                    log[key] = dict(shaper=rid_s, batch_pad_bytes=pbs, batch_delay_ms=pms, adv_eval_mean_pad_bytes=float(np.mean(pbs)),
                                    adv_eval_mean_delay_ms=float(np.mean(pms)), f1_mem={k: mf1(y_all[te], np.concatenate(p)) for k, p in mem.items()})
                    print(f'{key}: shaper {rid_s} pad {np.mean(pbs):.1f} B delay {np.mean(pms):.1f} ms | in-memory F1 enc {log[key]["f1_mem"]["encoder"]:.1f} '
                          f'advtrain {log[key]["f1_mem"]["encoder_advtrain"]:.1f} ({time.time() - t0:.0f}s)', flush=True)
            del M; torch.cuda.empty_cache()
    arrays['__log__'] = np.array(json.dumps(log))
    np.savez_compressed(SHAPED_NPZ, **arrays)
    print(f'wrote {SHAPED_NPZ} ({os.path.getsize(SHAPED_NPZ) / 1e6:.1f} MB) in {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ stage 2: scoring
def shuffled_control(raw, shaped, rs):
    """raw [n,64,4], shaped [n,64,4] (wb): permute each connection's byte additions among its payload packets with room,
    clip to the room; permute its delay additions among its packets (rule 1, wb_shuffled)."""
    has = raw[..., 2] > 0; real = raw[..., 1] > 0; room = np.clip(1460.0 - raw[..., 2].astype(np.float64), 0.0, None)
    db = shaped[..., 2].astype(np.float64) - raw[..., 2]; dt = shaped[..., 3].astype(np.float64) - raw[..., 3]
    ndb = np.zeros_like(db); ndt = np.zeros_like(dt)
    for i in range(len(raw)):
        el = np.where(has[i] & (room[i] > 0))[0]
        if len(el): ndb[i, el] = np.minimum(db[i, el][rs.permutation(len(el))], room[i, el])
        rl = np.where(real[i])[0]
        if len(rl): ndt[i, rl] = dt[i, rl][rs.permutation(len(rl))]
    out = raw.astype(np.float64).copy(); out[..., 1] += ndb; out[..., 2] += ndb; out[..., 3] += ndt
    return out.astype(np.float32)


def overheads(raw, shaped):
    has = raw[..., 2] > 0; real = raw[..., 1] > 0
    db = shaped[..., 2].astype(np.float64) - raw[..., 2]; dt = shaped[..., 3].astype(np.float64) - raw[..., 3]
    per_conn = (dt * real).sum(1) * 1000.0
    return dict(pooled_pad_bytes=float(db[has].sum() / has.sum()), pooled_delay_ms=float(dt[real].sum() / real.sum() * 1000.0),
                bytes_ratio=float(db[has].sum() / raw[..., 1][real].astype(np.float64).sum()),
                conn_delay_ms_mean=float(per_conn.mean()), conn_delay_ms_median=float(np.median(per_conn)),
                delay_ratio=float(dt[real].sum() / raw[..., 3][real].astype(np.float64).sum()))


def read_locked():
    L = {}
    for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
        j = json.loads(l)
        if not str(j.get('tag', '')).startswith('smoke'): L[j['run_id']] = j
    return L


def read_records():
    R = []
    for l in open(os.path.join(ROOT, 'results', 'adv_eval.jsonl'), encoding='utf-8'):
        if l.strip():
            j = json.loads(l)
            if j.get('tag') == TAG: R.append(j)
    return R


def stage_score():
    import lightgbm as lgb
    dev = torch.device('cuda'); t0 = time.time()
    Z = np.load(SHAPED_NPZ); log = json.loads(str(Z['__log__'])); LOCKED = read_locked()
    OUT = dict(rules=__doc__.split('RULES.')[1].split('Usage:')[0].strip(),
               grid=dict(budgets=[bkey(*b) for b in BUDGETS], steps=STEPS, batch=BS, first_k=FIRST_K, seeds=SEEDS, K=4),
               runs={k: v.replace('{pre}', '{ccma_}') for k, v in RUNS.items()} | dict(lgbm='{ccma_}fs4_lgbm_meta64_s{s} (refit, main recipe)', knn10='{ccma_}fs4_knn10_s{s} (recomputed)'),
               shaped_file=os.path.relpath(SHAPED_NPZ, ROOT), checks={}, chance={}, overheads={}, results={}, bootstrap={}, table=[])
    PRED = {}   # (ds, obs, traffic, budget) -> [pred per seed]
    chk = dict(R4_lgbm_npz_diff=[], R4_knn_npz_diff=[], R4_locked_absdiff_pp=[], R5_encoder_npz_diff=[], reload_vs_memory_absdiff_pp=[])
    for ds, c in CAMPAIGNS.items():
        D = GenAIData(ROOT, prefix=ds); y_all, NC = labels(D, c); nact = 2 if ds == 'genai' else 1; NA = NC // nact
        te = Z[f'{ds}|test_idx']; y = y_all[te]; sess = D.session_id[te]; raw_te = D.meta[te, :FIRST_K].astype(np.float32); len_te = D.meta_len[te]
        us, inv = np.unique(sess, return_inverse=True); nS = len(us); members = [np.where(inv == j)[0] for j in range(nS)]
        ys = np.array([np.bincount(y[m], minlength=NC).argmax() for m in members]); ysa = ys // nact
        OUT['chance'][ds] = dict(n_test_connections=int(len(te)), n_test_sessions=int(nS), task_uniform_f1=uniform_f1(y, NC),
                                 asst_uniform_f1=uniform_f1(y // nact, NA) if ds == 'genai' else None, sess_uniform_acc=100.0 / NA,
                                 sess_majority_acc=100.0 * np.bincount(ysa, minlength=NA).max() / nS)
        X_all = stat_features(D.meta, D.meta_len); raw_t = torch.from_numpy(raw_te); v_all = (raw_t[..., 1] > 0)
        OUT['overheads'][ds] = {}
        for s in SEEDS:
            idx, _ = split_idx(D, c, s); assert np.array_equal(idx['test'], te); tr = idx['train']
            # --- LightGBM, exactly the main run of train_baselines.py (CKPT_SELECT=last, LGBM_ROUNDS=1600)
            X = X_all.copy(); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
            clf = lgb.LGBMClassifier(n_estimators=1600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=s, verbose=-1)
            clf.fit(X[tr], y_all[tr])
            Ftr = feats_knn(D.meta[tr], D.meta_len[tr]); ytr = y_all[tr]
            M = {k: adv_eval.load_attacker(RUNS[k].format(pre=c['pre'], s=s), dev, NC) for k in RUNS}

            def predict(obs, R):
                if obs == 'lgbm': return clf.predict((stat_features(R, len_te) - mu) / sd).astype(int)
                if obs == 'knn10': return nn1(Ftr, ytr, feats_knn(R, len_te)).astype(int)
                fn, m_, s_, _ = M[obs]; out = []
                Rt = torch.from_numpy(R)
                with torch.no_grad():
                    for b in range(0, len(R), BS):
                        out.append(fn(adv_eval.raw_to_feat(Rt[b:b + BS].to(dev), m_, s_, FIRST_K), v_all[b:b + BS].to(dev)).argmax(1).cpu().numpy())
                return np.concatenate(out).astype(int)

            sets = {('none', 'none'): raw_te}
            for bi_, (bb, bm) in enumerate(BUDGETS):
                for src in SOURCES:
                    R = raw_te.copy(); R[..., 1:] = Z[akey(ds, s, src, bb, bm)]; sets[(src, bkey(bb, bm))] = R
                rs = np.random.RandomState(20260927 + 1000 * c['dsi'] + 10 * s + bi_)
                sets[('wb_shuffled', bkey(bb, bm))] = shuffled_control(raw_te, sets[('wb', bkey(bb, bm))], rs)
            for (tr_name, bk), R in sets.items():
                if tr_name != 'none':
                    OUT['overheads'][ds].setdefault(tr_name, {}).setdefault(bk, []).append(overheads(raw_te, R))
                for obs in OBSERVERS:
                    PRED.setdefault((ds, obs, tr_name, bk), []).append(predict(obs, R))
            # --- reproduction checks R4, R5 and reload-vs-memory
            for obs, rid in (('lgbm', f"{c['pre']}fs4_lgbm_meta64_s{s}"), ('knn10', f"{c['pre']}fs4_knn10_s{s}"), ('encoder', f"{c['pre']}fs4_sslXL_s{s}")):
                z = np.load(os.path.join(ROOT, 'results', 'runs', rid, 'test_preds.npz')); assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], y), rid
                ndiff = int((z['pred'].astype(int) != PRED[(ds, obs, 'none', 'none')][-1]).sum())
                chk[{'lgbm': 'R4_lgbm_npz_diff', 'knn10': 'R4_knn_npz_diff', 'encoder': 'R5_encoder_npz_diff'}[obs]].append(dict(run=rid, n_diff=ndiff, n=int(len(te))))
                if obs != 'encoder':
                    chk['R4_locked_absdiff_pp'].append(dict(run=rid, tag=LOCKED[rid].get('tag'), absdiff=abs(mf1(y, PRED[(ds, obs, 'none', 'none')][-1]) - 100 * LOCKED[rid]['test']['macro_f1'])))
            for bb, bm in BUDGETS:
                for src in SOURCES:
                    lg = log[akey(ds, s, src, bb, bm)]
                    for k in ('encoder', 'encoder_advtrain'):
                        chk['reload_vs_memory_absdiff_pp'].append(abs(lg['f1_mem'][k] - mf1(y, PRED[(ds, k, src, bkey(bb, bm))][-1])))
            print(f'{ds} s{s}: scored ({time.time() - t0:.0f}s)', flush=True)
            del M; torch.cuda.empty_cache()

        # --- metrics per observer x traffic x budget
        def scores(p):
            pa = p // nact; psa = np.array([np.bincount(pa[m], minlength=NA).argmax() for m in members])
            d = dict(task_f1=mf1(y, p), sess_vote=100.0 * float((psa == ysa).mean()), sess_n_correct=int((psa == ysa).sum()))
            if ds == 'genai': d['asst_f1'] = mf1(y // nact, pa)
            return d
        ch = OUT['chance'][ds]; chance = dict(task_f1=ch['task_uniform_f1'], asst_f1=ch['asst_uniform_f1'], sess_vote=ch['sess_uniform_acc'])
        res = {}; per = {}
        for (d_, obs, trn, bk), preds in PRED.items():
            if d_ != ds: continue
            sc = [scores(p) for p in preds]; per[(obs, trn, bk)] = sc
        for (obs, trn, bk), sc in per.items():
            e = {m: summ([x[m] for x in sc]) for m in sc[0] if m != 'sess_n_correct'}
            e['sess_n_correct'] = [x['sess_n_correct'] for x in sc]
            if trn != 'none':
                base = per[(obs, 'none', 'none')]
                for m in ('task_f1', 'asst_f1', 'sess_vote'):
                    if m not in sc[0]: continue
                    Rm = [(x[m] - chance[m]) / (b[m] - chance[m]) for x, b in zip(sc, base)]
                    e[f'retained_{m}'] = summ(Rm)
                    e[f'drop_{m}'] = summ([b[m] - x[m] for x, b in zip(sc, base)])
                e['verdict_task'] = 'keeps most of the leak' if e['retained_task_f1']['mean'] > 0.5 else 'most of the leak removed'
                e['verdict_vote'] = 'keeps most of the leak' if e['retained_sess_vote']['mean'] > 0.5 else 'most of the leak removed'
            res.setdefault(trn, {}).setdefault(bk, {})[obs] = e
        OUT['results'][ds] = res
        # overhead summaries (definition (a) from the PGD log; (b)-(d) from the saved arrays)
        ov = {}
        for trn, bd in OUT['overheads'][ds].items():
            ov[trn] = {}
            for bk, lst in bd.items():
                o = {k: summ([x[k] for x in lst]) for k in lst[0]}
                if trn in SOURCES:
                    bb, bm = [float(t) for t in bk.split('_')]
                    o['adv_eval_mean_pad_bytes'] = summ([log[akey(ds, s, trn, bb, bm)]['adv_eval_mean_pad_bytes'] for s in SEEDS])
                    o['adv_eval_mean_delay_ms'] = summ([log[akey(ds, s, trn, bb, bm)]['adv_eval_mean_delay_ms'] for s in SEEDS])
                ov[trn][bk] = o
        OUT['overheads'][ds] = ov
        # --- bootstrap (rule 5)
        cells = {'task': boot2.Cell(y, sess, NC)}
        if ds == 'genai': cells['asst'] = boot2.Cell(y // nact, sess, NA)
        bs_out = {}
        for cname, cell in cells.items():
            conv = (lambda p: p) if cname == 'task' else (lambda p: p // nact)
            for (d_, obs, trn, bk), preds in PRED.items():
                if d_ == ds: cell.add(f'{obs}|{trn}|{bk}', [conv(p) for p in preds])
            comp = {}
            for bb, bm in BUDGETS:
                bk = bkey(bb, bm)
                for S in ('wb', 'transfer'):
                    for o in ('lgbm', 'knn10'): comp[f'C1 {o}@{S} - encoder@{S} [{bk}]'] = cell.compare(f'{o}|{S}|{bk}', f'encoder|{S}|{bk}')
                    for o in ('encoder', 'lgbm', 'knn10'): comp[f'C2 {o}@{S} - {o}@none [{bk}]'] = cell.compare(f'{o}|{S}|{bk}', f'{o}|none|none')
                for o in ('encoder', 'lgbm', 'knn10'): comp[f'C3 {o}@wb - {o}@wb_shuffled [{bk}]'] = cell.compare(f'{o}|wb|{bk}', f'{o}|wb_shuffled|{bk}')
                for o in ('encoder_advtrain', 'lgbm', 'knn10'): comp[f'C4 {o}@wb_advtrain - {o}@none [{bk}]'] = cell.compare(f'{o}|wb_advtrain|{bk}', f'{o}|none|none')
            bs_out[cname] = {k: dict(diff=v['diff'], ci=v['ci'], verdict=v['verdict'], per_draw=v['per_draw'], n_pos=v['n_pos'], n_neg=v['n_neg']) for k, v in comp.items()}
        OUT['bootstrap'][ds] = bs_out
        # --- paired per-seed session-vote differences (no test)
        pv = {}
        for bb, bm in BUDGETS:
            bk = bkey(bb, bm)
            for S in ('wb', 'transfer', 'wb_shuffled', 'wb_advtrain'):
                for o in ('lgbm', 'knn10'):
                    ref = 'encoder_advtrain' if S == 'wb_advtrain' else 'encoder'
                    a = [x['sess_vote'] for x in per[(o, S, bk)]]; b = [x['sess_vote'] for x in per[(ref, S, bk)]]
                    dd = np.array(a) - np.array(b)
                    pv[f'{o}@{S} - {ref}@{S} [{bk}]'] = dict(mean=float(dd.mean()), per_seed=[float(x) for x in dd], n_pos=int((dd > 0).sum()), n_neg=int((dd < 0).sum()))
        OUT['bootstrap'][ds]['session_vote_paired'] = pv
        # --- compact table rows
        for bb, bm in [('none', None)] + BUDGETS:
            bk = 'none' if bb == 'none' else bkey(bb, bm)
            for trn in (['none'] if bk == 'none' else ['wb', 'transfer', 'wb_shuffled', 'wb_advtrain']):
                row = dict(ds=ds, budget=bk, traffic=trn)
                if trn != 'none':
                    o = OUT['overheads'][ds][trn][bk]
                    row['pad_B_adv_eval_def'] = o['adv_eval_mean_pad_bytes']['mean'] if 'adv_eval_mean_pad_bytes' in o else None
                    row['delay_ms_adv_eval_def'] = o['adv_eval_mean_delay_ms']['mean'] if 'adv_eval_mean_delay_ms' in o else None
                    row['pad_B_pooled'] = o['pooled_pad_bytes']['mean']; row['delay_ms_pooled'] = o['pooled_delay_ms']['mean']
                    row['bytes_ratio'] = o['bytes_ratio']['mean']; row['delay_ratio'] = o['delay_ratio']['mean']
                for obs in OBSERVERS:
                    e = res[trn][bk][obs]
                    row[obs] = {m: [e[m]['mean'], e[m]['min'], e[m]['max']] for m in ('task_f1', 'asst_f1', 'sess_vote') if m in e}
                OUT['table'].append(row)
    # --- reproduction check R1-R3 against adv_eval.jsonl
    recs = read_records(); r12 = []
    for j in recs:
        ds = j['dataset']; c = CAMPAIGNS[ds]; s = int(j['seed']); bk = bkey(j['budget_bytes'], j['budget_ms'])
        tgt = 'encoder_advtrain' if '_advtrain_' in j['run'] else 'encoder'
        assert j['run'] == RUNS[tgt].format(pre=c['pre'], s=s) and int(j['steps']) == STEPS, j
        src = ('wb' if tgt == 'encoder' else 'wb_advtrain') if j['surrogate'] == 'whitebox' else 'transfer'
        if src == 'transfer': assert j['surrogate'] == RUNS['surrogate'].format(pre=c['pre'], s=s), j
        f_clean = OUT['results'][ds]['none']['none'][tgt]['task_f1']['per_seed'][s]
        f_adv = OUT['results'][ds][src][bk][tgt]['task_f1']['per_seed'][s]
        lg = log[akey(ds, s, src, j['budget_bytes'], j['budget_ms'])]
        r12.append(dict(run=j['run'], surrogate=j['surrogate'], budget=bk, source=src,
                        d_clean_pp=f_clean - 100 * j['test_f1_clean'], d_adv_pp=f_adv - 100 * j['test_f1_adv'],
                        rel_d_pad=lg['adv_eval_mean_pad_bytes'] / j['mean_pad_bytes'] - 1, rel_d_delay=lg['adv_eval_mean_delay_ms'] / j['mean_delay_ms'] - 1,
                        d_clean_mem_vs_score_pp=log[f'{ds}|s{s}|clean_f1_mem'][tgt] - f_clean))
    OUT['checks'] = dict(
        n_records=len(r12),
        R1_max_abs_clean_diff_pp=max(abs(r['d_clean_pp']) for r in r12), R1_pass=all(abs(r['d_clean_pp']) < 1e-4 for r in r12),
        R2_max_abs_adv_diff_pp=max(abs(r['d_adv_pp']) for r in r12), R2_mean_abs_adv_diff_pp=float(np.mean([abs(r['d_adv_pp']) for r in r12])),
        R2_pass=all(abs(r['d_adv_pp']) <= 0.5 for r in r12), R2_n_exact=int(sum(abs(r['d_adv_pp']) < 1e-9 for r in r12)),
        R3_max_rel_pad=max(abs(r['rel_d_pad']) for r in r12), R3_max_rel_delay=max(abs(r['rel_d_delay']) for r in r12),
        R3_pass=all(abs(r['rel_d_pad']) <= 0.02 and abs(r['rel_d_delay']) <= 0.02 for r in r12),
        R4_lgbm_npz_diff=chk['R4_lgbm_npz_diff'], R4_knn_npz_diff=chk['R4_knn_npz_diff'],
        R4_pass=all(x['n_diff'] == 0 for x in chk['R4_lgbm_npz_diff'] + chk['R4_knn_npz_diff']) and all(x['absdiff'] < 1e-6 for x in chk['R4_locked_absdiff_pp']),
        R4_locked=chk['R4_locked_absdiff_pp'], R5_encoder_npz_diff=chk['R5_encoder_npz_diff'],
        reload_vs_memory_max_absdiff_pp=max(chk['reload_vs_memory_absdiff_pp']), per_record=r12)
    json.dump(rnd(OUT), open(OUT_JSON, 'w', encoding='utf-8'), indent=1)
    print(f'wrote {OUT_JSON} ({time.time() - t0:.0f}s)')
    c_ = OUT['checks']
    print('checks: R1', c_['R1_pass'], c_['R1_max_abs_clean_diff_pp'], '| R2', c_['R2_pass'], c_['R2_max_abs_adv_diff_pp'], 'exact', c_['R2_n_exact'],
          '| R3', c_['R3_pass'], c_['R3_max_rel_pad'], c_['R3_max_rel_delay'], '| R4', c_['R4_pass'], '| reload', c_['reload_vs_memory_max_absdiff_pp'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='all', choices=['all', 'shape', 'score']); a = ap.parse_args()
    if a.stage in ('all', 'shape'): stage_shape()
    if a.stage in ('all', 'score'): stage_score()
