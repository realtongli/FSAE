"""Closed-world device windows: the pre-trained encoder with a background class (variant B of
scripts/window_assistant.py, part (b)), next to its plain vote (A), the OS-grouped reference (O), and LightGBM / 1-NN under B.
Writes one key, 'closed_world_encoder_B', into results/window_assistant.json (every other key is left unchanged; asserted
before writing). Per-draw predictions are cached under results/cache/ (WEB_CACHE).

Rules. The window rules are those of scripts/window_assistant.py, part (b); the encoder's B model is defined here.
 1. Splits [ccma_]fewshot_k{K}_s{s}.json, K = 1, 2, 4, 8, seeds 0-4; GenAI (6 joint classes -> 3 assistants, j // 2) and
    CCMA (9 apps, the app is the task). A window is one held-out test session: its target-app test connections plus all
    its background connections (data/derived/<prefix>_background.npz, same session id); no grouping. True assistant = the
    session's label.
 2. Encoder B model = the fs{K}_sslXL / ccma_fs{K}_sslXL fine-tuning recipe of src/train_meta_ssl.py (as launched by
    scripts/run_paper.sh: --mode ft --epochs 60 --bs 64 --lr 0.001 --lr_enc 0.0003 --drop_head 0.3 --first_k 64
    --defense none --select last --init weights/meta_ssl_mpm_XL.pt; GenAI --target joint, CCMA --target app) with ONE extra
    output class: the pre-trained encoder weights and the standardisation mu/sd stored in weights/meta_ssl_mpm_XL.pt;
    inputs src/train_meta.features (first 64 packets, no defence), validity = direction != 0, standardised inputs zeroed
    at invalid positions; head Dropout(0.3) + Linear(2d, C + 1); AdamW (encoder lr 3e-4, head lr 1e-3, weight decay 0.01);
    unweighted cross-entropy; batches of 64 from np.random.RandomState(seed).permutation each epoch (a batch of one is
    skipped); 60 epochs; the final model is kept; train_yatc.set_seed(seed) right before the model is built.
    Training set = the target connections of the K labelled sessions per class (exactly the fs{K}_sslXL training set) plus
    every background connection of those same training sessions, labelled C (no cap, no re-weighting; as LightGBM B).
    Validation sessions are used for nothing (the original loop only logs a validation F1 per epoch under select=last;
    that call is skipped here, and it draws no random numbers).
 3. B vote (identical to window_assistant.py): majority over the window's connections not predicted background, ties to
    the lowest assistant index (np.bincount(...).argmax()); a window whose connections are all predicted background counts
    as wrong. B+ repeats B with the window's connections of the other monitored packages added (the cached cross
    connections of window_assistant.cross_flows). A / A+ / O as in window_assistant.py.
 4. Reference numbers recomputed here with the same per-window code and checked against results/window_assistant.json
    (every per-seed share must agree at the stored 2 decimals): encoder A / A+ / O from the saved checkpoints
    results/runs/[ccma_]fs{K}_sslXL_s{s}/best.pt (inference only), LightGBM A / A+ / O / B / B+ and 1-NN A / A+ / O / B / B+
    (the recipes of window_assistant.py). The refitted LightGBM and the checkpoints must reproduce the saved test_preds.
 5. Recipe check: the same training function without the background class (C outputs, target connections only) re-trains
    every fs{K}_sslXL run; its test predictions are compared with the saved test_preds.npz (disagreeing connections and
    the macro-F1 difference are reported, whatever they are; GPU kernels need not be bitwise deterministic).
 6. Reported per campaign and K: share (%) of windows with the right assistant, mean and min-max over seeds 0-4, for encoder
    O / A / B / A+ / B+, LightGBM O / A / B / B+, 1-NN O / A / B / B+; for every B model: share of test background
    connections predicted background, share of test app connections predicted background (lost to it), windows with every
    connection predicted background, B-vote ties, B accuracy by true assistant; descriptive: connection-level task
    macro-F1 on the app test connections of the encoder B model (a background prediction counts as a wrong class) next to
    that of the checkpoint (plain model).
 7. Paired two-level bootstrap of the window share (the resampling of scripts/boot2.py, rule 4, with windows in place of
    test sessions: 4000 resamples from np.random.RandomState(20260926), drawn once per (campaign, K) cell in this order:
    draw indices (4000 x 5) then window multiplicities Multinomial(n_windows, uniform) (4000 x n_windows); statistic
    mean_j [share(X_r*j; w) - share(Y_r*j; w)] with the same w for X and Y; interval 2.5 / 97.5 percentiles; verdict
    X>Y / X<Y / ns). Comparisons: encoder B - encoder A, encoder B - LightGBM B, encoder B - 1-NN B, encoder B - encoder O.
    Reported next to them: the per-draw differences and the number of draws with X > Y.
The recipe is held constant and the vote has no threshold.
Usage: python scripts/window_encoder_bg.py run:<genai|ccma>[:K,K..]   (GPU + CPU; caches per-draw predictions)
       python scripts/window_encoder_bg.py summ                       (windows, checks, bootstrap; writes the JSON key)"""
import collections, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from window_assistant import cross_flows, enc_feats, nn1, summ, SEEDS   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'window_assistant.json')
KEY = 'closed_world_encoder_B'
CACHE = os.environ.get('WEB_CACHE', os.path.join(ROOT, 'results', 'cache', 'window_encoder_bg'))
KS = (1, 2, 4, 8)
RECIPE = dict(epochs=60, bs=64, lr=1e-3, lr_enc=3e-4, drop_head=0.3, wd=0.01, first_k=64)
INIT = os.path.join(ROOT, 'weights', 'meta_ssl_mpm_XL.pt')
B_BOOT, SEED_BOOT = 4000, 20260926


def ds_info(ds):
    from data_mfr import GenAIData
    data = GenAIData(ROOT, prefix=ds)
    y = data.y_joint if ds == 'genai' else data.y_app; NC = data.n_joint if ds == 'genai' else data.n_app
    NA = 3 if ds == 'genai' else 9
    to_a = (lambda c: np.asarray(c) // 2) if ds == 'genai' else (lambda c: np.asarray(c))
    bgz = np.load(os.path.join(ROOT, 'data', 'derived', f'{ds}_background.npz'), allow_pickle=True)
    cmeta, clen, csid, _ = cross_flows(ds)
    return data, y, NC, NA, to_a, bgz['meta'], bgz['meta_len'], bgz['session_id'], cmeta, clen, csid


def split(data, bsid, csid, ds, K, s):
    pre_ = '' if ds == 'genai' else 'ccma_'
    idx, _ = data.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre_}fewshot_k{K}_s{s}.json'))
    tr, it = idx['train'], idx['test']
    tr_sess = np.unique(data.session_id[tr]); te_sess = np.unique(data.session_id[it])
    btr = np.where(np.isin(bsid, tr_sess))[0]; bte = np.where(np.isin(bsid, te_sess))[0]; cte = np.where(np.isin(csid, te_sess))[0]
    return tr, it, btr, bte, cte, te_sess


# --------------------------------------------------------------------------------------------------------------- training
def finetune(pre, Xtr, Vtr, ytr, n_cls, seed, dev):
    """src/train_meta_ssl.main, --mode ft, select=last, without the per-epoch validation logging."""
    import torch, torch.nn as nn
    from train_yatc import set_seed
    from pretrain_meta import MetaEncoder
    from train_meta_ssl import Classifier
    set_seed(seed); c = pre['cfg']
    enc = MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L']); enc.load_state_dict(pre['encoder'])
    model = Classifier(enc, n_cls, RECIPE['drop_head']).to(dev)
    opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': RECIPE['lr_enc']}, {'params': model.head.parameters(), 'lr': RECIPE['lr']}], weight_decay=RECIPE['wd'])
    crit = nn.CrossEntropyLoss(); rng = np.random.RandomState(seed); n = len(ytr)
    for ep in range(RECIPE['epochs']):
        model.train(); perm = rng.permutation(n)
        for b in range(0, n, RECIPE['bs']):
            bi = perm[b:b + RECIPE['bs']]
            if len(bi) < 2: continue
            loss = crit(model(Xtr[bi].to(dev), Vtr[bi].to(dev)), torch.from_numpy(ytr[bi]).to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


def run(ds, Ks):
    import torch, lightgbm as lgb
    from sklearn.metrics import f1_score
    from trivial_baselines import feats_knn
    from train_baselines import stat_features
    from train_meta import features as meta_features
    from pretrain_meta import MetaEncoder
    from train_meta_ssl import Classifier, predict
    os.makedirs(CACHE, exist_ok=True); dev = torch.device('cuda')
    data, y, NC, NA, to_a, bmeta, blen, bsid, cmeta, clen, csid = ds_info(ds); BG = NC; pre_ = '' if ds == 'genai' else 'ccma_'
    pre = torch.load(INIT, weights_only=False); mu, sd = np.asarray(pre['mu']), np.asarray(pre['sd'])
    t0 = time.time()
    Xr = meta_features(data, RECIPE['first_k']); assert np.array_equal(Xr, enc_feats(data.meta, data.meta_len))
    Xbr = enc_feats(bmeta, blen); Xcr = enc_feats(cmeta, clen) if len(clen) else np.zeros((0, 64, 4), np.float32)

    def std(Xraw):
        V = Xraw[..., 0] != 0.0
        return torch.from_numpy(((Xraw - mu) / sd * V[..., None]).astype(np.float32)), torch.from_numpy(V)
    X, V = std(Xr); Xb, Vb = std(Xbr); Xc, Vc = std(Xcr)
    Fk = feats_knn(data.meta, data.meta_len); Fkb = feats_knn(bmeta, blen); Fkc = feats_knn(cmeta, clen) if len(clen) else np.zeros((0, 40), np.float32)
    Fs = stat_features(data.meta, data.meta_len); Fsb = stat_features(bmeta, blen); Fsc = stat_features(cmeta, clen) if len(clen) else np.zeros((0, Fs.shape[1]), np.float32)
    print(f'{ds}: features {time.time() - t0:.0f}s; target {len(y)}, background {len(blen)}, cross {len(clen)}', flush=True)

    def ep(model, Xt, Vt):
        return predict(model, Xt, Vt, dev).astype(np.int64) if len(Xt) else np.zeros(0, np.int64)

    for K in Ks:
        for s in SEEDS:
            f = os.path.join(CACHE, f'{ds}_K{K}_s{s}.npz')
            if os.path.exists(f): print('cached', f); continue
            tr, it, btr, bte, cte, te_sess = split(data, bsid, csid, ds, K, s); out = dict(it=it, bte=bte, cte=cte, btr=btr, te_sess=te_sess)
            q = lambda m: np.concatenate([ep(m, X[it], V[it]), ep(m, Xb[bte], Vb[bte]), ep(m, Xc[cte], Vc[cte])])
            # encoder B: target + background of the K training sessions, one extra class
            t1 = time.time()
            XB = torch.cat([X[tr], Xb[btr]]); VB = torch.cat([V[tr], Vb[btr]]); yB = np.concatenate([y[tr], np.full(len(btr), BG, dtype=y.dtype)])
            m = finetune(pre, XB, VB, yB, NC + 1, s, dev); out['enc_B'] = q(m); out['time_B'] = time.time() - t1; del m, XB, VB
            # recipe check: the same function, no background class
            t1 = time.time(); m = finetune(pre, X[tr], V[tr], y[tr], NC, s, dev); out['check_plain'] = ep(m, X[it], V[it]); out['time_plain'] = time.time() - t1; del m
            # encoder A / O: saved checkpoint, inference only
            run_id = f'{pre_}fs{K}_sslXL_s{s}'; ck = torch.load(os.path.join(ROOT, 'results', 'runs', run_id, 'best.pt'), weights_only=False)
            assert ck['args']['seed'] == s and ck['args']['select'] == 'last' and ck['args']['split'].replace('\\', '/').endswith(f'{pre_}fewshot_k{K}_s{s}.json')
            assert np.array_equal(np.asarray(ck['norm'][0]), mu) and np.array_equal(np.asarray(ck['norm'][1]), sd)
            for k_ in ('epochs', 'bs', 'lr', 'lr_enc', 'drop_head', 'first_k'): assert float(ck['args'][k_]) == float(RECIPE[k_]), (run_id, k_)
            c = pre['cfg']; m = Classifier(MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L']), NC, 0.3); m.load_state_dict(ck['model']); m.to(dev).eval()
            out['enc_A'] = q(m); del m; torch.cuda.empty_cache()
            zp = np.load(os.path.join(ROOT, 'results', 'runs', run_id, 'test_preds.npz')); assert np.array_equal(zp['idx'], it)
            out['saved_enc_pred'] = zp['pred'].astype(np.int64)
            # 1-NN (first 10 packets, L1)
            Qk = np.concatenate([Fk[it], Fkb[bte], Fkc[cte]])
            out['knn10_A'] = nn1(Fk[tr], y[tr], Qk)
            out['knn10_B'] = nn1(np.concatenate([Fk[tr], Fkb[btr]]), np.concatenate([y[tr], np.full(len(btr), BG)]), Qk)
            # LightGBM (fixed recipe of window_assistant.py)
            mu_s, sd_s = Fs[tr].mean(0), Fs[tr].std(0) + 1e-6; st = lambda F: (F - mu_s) / sd_s
            Q = st(np.concatenate([Fs[it], Fsb[bte], Fsc[cte]]))
            kw = dict(n_estimators=1600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=s, verbose=-1)
            clf = lgb.LGBMClassifier(**kw); clf.fit(st(Fs[tr]), y[tr]); out['lgbm_A'] = clf.predict(Q).astype(np.int64)
            clf = lgb.LGBMClassifier(**kw); clf.fit(st(np.concatenate([Fs[tr], Fsb[btr]])), np.concatenate([y[tr], np.full(len(btr), BG)])); out['lgbm_B'] = clf.predict(Q).astype(np.int64)
            zl = np.load(os.path.join(ROOT, 'results', 'runs', f'{pre_}fs{K}_lgbm_meta64_s{s}', 'test_preds.npz')); assert np.array_equal(zl['idx'], it)
            out['saved_lgbm_pred'] = zl['pred'].astype(np.int64)
            np.savez_compressed(f, **out)
            nT = len(it); yt = y[it]
            print(f'{ds} K={K} s={s}: train {len(tr)} target + {len(btr)} background; B {out["time_B"]:.0f}s, plain {out["time_plain"]:.0f}s; '
                  f'check: plain-retrain vs saved preds differ on {int((out["check_plain"] != out["saved_enc_pred"]).sum())}/{nT}, '
                  f'F1 {100 * f1_score(yt, out["check_plain"], average="macro"):.2f} vs {100 * f1_score(yt, out["saved_enc_pred"], average="macro"):.2f}; '
                  f'ckpt vs saved {int((out["enc_A"][:nT] != out["saved_enc_pred"]).sum())}, lgbm vs saved {int((out["lgbm_A"][:nT] != out["saved_lgbm_pred"]).sum())} '
                  f'({time.time() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------------------------------------------------- windows + JSON
def window_eval(pA, pB, seg_sid, seg_kind, te_sess, true_w, to_a, NA, BG):
    """Per-window correctness (window_assistant.py rules). Returns {tag: bool[nW]} and B counters."""
    C = {t: np.zeros(len(te_sess), bool) for t in ('A', 'A+', 'O', 'B', 'B+')}; n_allbg = {'B': 0, 'B+': 0}; ties_B = 0
    for i, w in enumerate(te_sess):
        m = seg_sid == w; k = seg_kind[m]; true = true_w[i]
        if pA is not None:
            a_all = to_a(pA[m])
            C['A'][i] = int(np.bincount(a_all[k < 2], minlength=NA).argmax()) == true
            C['A+'][i] = int(np.bincount(a_all, minlength=NA).argmax()) == true
            C['O'][i] = int(np.bincount(a_all[k == 0], minlength=NA).argmax()) == true
        if pB is not None:
            b_all = pB[m]
            for tag, sel in (('B', k < 2), ('B+', np.ones(len(k), bool))):
                keep = b_all[sel]; keep = keep[keep != BG]
                if len(keep):
                    cnt = np.bincount(to_a(keep), minlength=NA); C[tag][i] = int(cnt.argmax()) == true
                    if tag == 'B': ties_B += int((cnt == cnt.max()).sum() > 1)
                else: n_allbg[tag] += 1
    return C, n_allbg, ties_B


def boot_cell(Cx, Cy):
    """Cx, Cy: [5, nW] 0/1 per draw and window (same windows). Two-level paired bootstrap, rule 7."""
    R, nW = Cx.shape; rs = np.random.RandomState(SEED_BOOT)
    Ss = rs.randint(0, R, size=(B_BOOT, R)); Ws = rs.multinomial(nW, np.ones(nW) / nW, size=B_BOOT)
    D = (Cx.astype(float) - Cy.astype(float)); Dw = D @ Ws.T / nW                          # [R, B]
    stat = 100 * Dw[Ss, np.arange(B_BOOT)[:, None]].mean(1)
    lo, hi = np.percentile(stat, [2.5, 97.5]); per = 100 * D.mean(1)
    return dict(diff=round(float(per.mean()), 2), ci95=[round(float(lo), 2), round(float(hi), 2)],
                verdict='X>Y' if lo > 0 else ('X<Y' if hi < 0 else 'ns'), per_draw=[round(float(v), 2) for v in per],
                draws_X_gt_Y=int((per > 0).sum()), draws_X_lt_Y=int((per < 0).sum()))


def summarise():
    from sklearn.metrics import f1_score
    wa = json.load(open(OUT, encoding='utf-8')); before = json.dumps({k: v for k, v in wa.items() if k != KEY}, sort_keys=True)
    ref = wa['closed_world']['cells']
    res = dict(rules=__doc__.split('Rules.')[1].split('Usage:')[0].strip(),
               script='scripts/window_encoder_bg.py', recipe=RECIPE, init='weights/meta_ssl_mpm_XL.pt',
               checks=dict(ref_share_max_abs_diff=0.0, ref_cells_compared=0, ckpt_vs_saved_mismatch=[], lgbm_vs_saved_mismatch=[],
                           plain_retrain_vs_saved={}, same_windows_every_seed={}),
               cells={}, diagnostics={}, bootstrap={}, training={})
    for ds in ('genai', 'ccma'):
        data, y, NC, NA, to_a, bmeta, blen, bsid, cmeta, clen, csid = ds_info(ds); BG = NC; pre_ = '' if ds == 'genai' else 'ccma_'
        for K in KS:
            cellkey = f'{ds}_K{K}'; S = collections.defaultdict(list); D = collections.defaultdict(list); Cw = collections.defaultdict(list); T = collections.defaultdict(list)
            chk = collections.defaultdict(list); wins = None
            for s in SEEDS:
                z = np.load(os.path.join(CACHE, f'{ds}_K{K}_s{s}.npz'))
                tr, it, btr, bte, cte, te_sess = split(data, bsid, csid, ds, K, s)
                assert all(np.array_equal(z[k], v) for k, v in (('it', it), ('bte', bte), ('cte', cte), ('btr', btr), ('te_sess', te_sess)))
                wins = te_sess if wins is None else wins; assert np.array_equal(wins, te_sess)
                nT, nB = len(it), len(bte); yt = y[it]
                seg_sid = np.concatenate([data.session_id[it], bsid[bte], csid[cte]])
                seg_kind = np.concatenate([np.zeros(nT, int), np.ones(nB, int), np.full(len(cte), 2)])
                true_w = np.array([int(to_a(np.bincount(yt[data.session_id[it] == w]).argmax())) for w in te_sess])
                nm = int((z['enc_A'][:nT] != z['saved_enc_pred']).sum())
                if nm: res['checks']['ckpt_vs_saved_mismatch'].append([f'{pre_}fs{K}_sslXL_s{s}', nm, nT])
                nm = int((z['lgbm_A'][:nT] != z['saved_lgbm_pred']).sum())
                if nm: res['checks']['lgbm_vs_saved_mismatch'].append([f'{pre_}fs{K}_lgbm_meta64_s{s}', nm, nT])
                chk['n_disagree'].append(int((z['check_plain'] != z['saved_enc_pred']).sum())); chk['n_test'].append(nT)
                chk['f1_retrain'].append(round(100 * f1_score(yt, z['check_plain'], average='macro'), 2))
                chk['f1_saved'].append(round(100 * f1_score(yt, z['saved_enc_pred'], average='macro'), 2))
                obs = dict(encoder=(z['enc_A'], z['enc_B']), lgbm=(z['lgbm_A'], z['lgbm_B']), knn10=(z['knn10_A'], z['knn10_B']))
                for o, (pA, pB) in obs.items():
                    C, allbg, tiesB = window_eval(pA, pB, seg_sid, seg_kind, te_sess, true_w, to_a, NA, BG)
                    for t, v in C.items(): S[(o, t)].append(float(v.mean())); Cw[(o, t)].append(v)
                    D[(o, 'B_bg_conn_pred_bg')].append(float((pB[nT:nT + nB] == BG).mean()) if nB else 0.0)
                    D[(o, 'B_app_conn_pred_bg')].append(float((pB[:nT] == BG).mean()))
                    D[(o, 'B_windows_all_bg')].append(allbg['B']); D[(o, 'B+_windows_all_bg')].append(allbg['B+']); D[(o, 'B_ties')].append(tiesB)
                    D[(o, 'B_app_conn_macro_f1')].append(float(f1_score(yt, pB[:nT], average='macro', labels=np.arange(NC))))
                    D[(o, 'A_app_conn_macro_f1')].append(float(f1_score(yt, pA[:nT], average='macro', labels=np.arange(NC))))
                    for a_ in range(NA):
                        sel = true_w == a_
                        if sel.any(): D[(o, f'B_correct_true{a_}')].append(float(C['B'][sel].mean())); D[(o, f'A_correct_true{a_}')].append(float(C['A'][sel].mean()))
                T['n_train_target'].append(int(len(tr))); T['n_train_background'].append(int(len(btr)))
                T['train_time_B_s'].append(round(float(z['time_B']), 1)); T['train_time_plain_s'].append(round(float(z['time_plain']), 1))
                T['n_windows'].append(int(len(te_sess))); T['n_test_app_conn'].append(nT); T['n_test_bg_conn'].append(nB)
            # reference check against window_assistant.json (every observer/tag it holds)
            for (o, t), v in S.items():
                rk = f'{ds}_{o}_K{K}_{t}'
                if rk in ref:
                    mine = [round(100 * x, 2) for x in v]; res['checks']['ref_cells_compared'] += 1
                    res['checks']['ref_share_max_abs_diff'] = max(res['checks']['ref_share_max_abs_diff'], max(abs(a - b) for a, b in zip(mine, ref[rk]['per_seed'])))
            res['checks']['plain_retrain_vs_saved'][cellkey] = dict(chk)
            res['checks']['same_windows_every_seed'][cellkey] = True
            res['cells'][cellkey] = {f'{o}_{t}': summ(v) for (o, t), v in sorted(S.items())}
            frac = lambda k: not (k.endswith('_all_bg') or k.endswith('_ties'))
            res['diagnostics'][cellkey] = {f'{o}|{k}': (summ(v) if frac(k) else v) for (o, k), v in sorted(D.items())}
            res['training'][cellkey] = dict(T)
            A = lambda o, t: np.array(Cw[(o, t)])
            res['bootstrap'][cellkey] = {'encoder_B-encoder_A': boot_cell(A('encoder', 'B'), A('encoder', 'A')),
                                         'encoder_B-lgbm_B': boot_cell(A('encoder', 'B'), A('lgbm', 'B')),
                                         'encoder_B-knn10_B': boot_cell(A('encoder', 'B'), A('knn10', 'B')),
                                         'encoder_B-encoder_O': boot_cell(A('encoder', 'B'), A('encoder', 'O'))}
            c = res['cells'][cellkey]; d = res['diagnostics'][cellkey]
            f = lambda k: f'{c[k]["mean"]:.1f} [{c[k]["min"]:.1f}-{c[k]["max"]:.1f}]'
            print(f'{ds} K={K}: enc O {f("encoder_O")} A {f("encoder_A")} B {f("encoder_B")} B+ {f("encoder_B+")} | LGB A {f("lgbm_A")} B {f("lgbm_B")} O {f("lgbm_O")} | '
                  f'1-NN A {f("knn10_A")} B {f("knn10_B")} O {f("knn10_O")}', flush=True)
            for o in ('encoder', 'lgbm', 'knn10'):
                print(f'    {o:7s} bg->bg {d[o + "|B_bg_conn_pred_bg"]["mean"]:.1f} [{d[o + "|B_bg_conn_pred_bg"]["min"]:.1f}-{d[o + "|B_bg_conn_pred_bg"]["max"]:.1f}]  '
                      f'app->bg {d[o + "|B_app_conn_pred_bg"]["mean"]:.1f} [{d[o + "|B_app_conn_pred_bg"]["min"]:.1f}-{d[o + "|B_app_conn_pred_bg"]["max"]:.1f}]  '
                      f'all-bg windows {d[o + "|B_windows_all_bg"]}  B ties {d[o + "|B_ties"]}  app F1 A {d[o + "|A_app_conn_macro_f1"]["mean"]:.1f} B {d[o + "|B_app_conn_macro_f1"]["mean"]:.1f}', flush=True)
            for cmp_, b in res['bootstrap'][cellkey].items():
                print(f'    {cmp_}: {b["diff"]:+.2f} [{b["ci95"][0]:+.2f}, {b["ci95"][1]:+.2f}] {b["verdict"]} per-draw {b["per_draw"]}', flush=True)
            print(f'    plain re-train vs saved: disagree {chk["n_disagree"]} of {chk["n_test"]}; F1 {chk["f1_retrain"]} vs {chk["f1_saved"]}', flush=True)
    print('checks:', json.dumps({k: v for k, v in res['checks'].items() if k not in ('plain_retrain_vs_saved', 'same_windows_every_seed')}), flush=True)
    wa2 = json.load(open(OUT, encoding='utf-8')); assert json.dumps({k: v for k, v in wa2.items() if k != KEY}, sort_keys=True) == before, 'window_assistant.json changed meanwhile'
    wa2[KEY] = res; json.dump(wa2, open(OUT, 'w', encoding='utf-8'), indent=1)
    wa3 = json.load(open(OUT, encoding='utf-8')); assert json.dumps({k: v for k, v in wa3.items() if k != KEY}, sort_keys=True) == before
    print('written', OUT, 'key', KEY)


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'summ'
    if part.startswith('run:'):
        a = part.split(':'); run(a[1], tuple(int(k) for k in a[2].split(',')) if len(a) > 2 else KS)
    elif part == 'summ':
        summarise()
    else:
        raise SystemExit(__doc__)
