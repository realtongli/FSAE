"""Assistant-level decision per device window WITHOUT grouping connections by the OS app attribution
(GenAI: ChatGPT / Copilot / Gemini, i.e. joint class // 2; CCMA: the app, which is already the task).
Writes results/window_assistant.json; the connections of the other monitored packages (sensitivity A+ / B+) are cached
under results/cache/ (WA_CACHE).

Rules:
(a) Open world, from the saved open_world arrays behind the open-world window table (results/runs/ow_scores/open_world_*.npz; observers lgbm,
    scratch, ours; negative regimes bg0 = own background only, bg1 = + labelled negatives; K = 2, 4, 8; seeds 0-4).
    Windows, window class and window score exactly as scripts/ow_windows_exact.py: a window is one held-out capture
    session with all its test connections (target app + background, no grouping); positives = sessions of monitored
    classes, negatives = sessions of never-labelled classes (proxy-class sessions are not windows); for each monitored
    class c, s_c = mean of the 5 highest calibrated-ensemble report scores among the window's connections predicted as c
    (fewer: mean of those present; none: 0); window class = argmax_c s_c; window score = max_c s_c.
    Assistant of a class: GenAI joint class j -> j // 2 (window class index i -> monitored[i]); CCMA: the app itself.
    Reported: share of monitored windows whose window class has the right assistant; assistant-level window PR-AUC
    (scripts/ow_metrics.pr_auc; positives = monitored windows, correct = right assistant; negatives = the never-labelled
    windows, unchanged). As a descriptive addition on GenAI: the share of never-labelled windows whose window class names
    their own assistant, and the share over all windows. Check: the exact-class share and PR-AUC recomputed here must
    equal results/ow_windows.json.
    The open_world class draw differs per seed. On GenAI the never-labelled classes are the other modality of monitored
    assistants in seeds 0 and 3, and both classes of an assistant that is not monitored in seeds 1, 2, 4 (Gemini,
    Copilot, ChatGPT), where naming the own assistant is impossible by construction. A per-seed-group breakdown
    ('seed_groups') is also reported; it splits the seeds by this property of the class draw only and reports both
    groups.
(b) Closed world (splits <prefix>fewshot_k{K}_s{seed}.json, K = 1, 2, 4, 8, seeds 0-4). A window is one held-out test
    session of one device: all its target-app test connections plus all its background connections
    (data/derived/<prefix>_background.npz, same session id); no grouping. True assistant = the session's label.
      A  plain vote: majority of the observer's per-connection predicted assistants over every connection of the window
         (np.bincount(...).argmax(): ties go to the lowest index, as in scripts/robust_assistant.py);
      B  background class: the observer is refitted with one extra class holding the background connections of its own
         training sessions (the K labelled sessions per class, which it captured itself); vote over the window's
         connections not predicted background (same tie rule); a window whose connections are all predicted background
         counts as wrong;
      O  reference only: the paper's session vote over the app's own (OS-attributed) connections, same model.
    Observers: 1-NN, L1, first 10 packets (scripts/trivial_baselines.feats_knn, the paper's knn10); LightGBM, fixed
    recipe (scripts/train_baselines.stat_features standardised with the mean/std of the target training connections,
    1,600 rounds, lr 0.03, 31 leaves, subsample 0.8, colsample 0.8, min_child_samples 5, random_state = seed), for B the
    same recipe on target + background training connections; pre-trained encoder: the saved fine-tuned checkpoints
    results/runs/<prefix>fs{K}_sslXL_s{seed}/best.pt, inference only. B for the encoder needs re-training (a new output
    class), so the encoder gets A and O only.
    Sensitivity: build_background.py drops in every session the connections of EVERY monitored package, so a session's
    connections of the other monitored apps (e.g. the Google app, i.e. the Gemini host, inside a ChatGPT session) are in
    neither data file. A+ / B+ add them to the window (read from the raw JSON with build_background.extract) and repeat
    A / B.
    Checks: the refitted plain LightGBM and the checkpointed encoder must reproduce the saved test_preds.npz of
    <prefix>fs{K}_lgbm_meta64_s{seed} / <prefix>fs{K}_sslXL_s{seed}; 1-NN must reproduce the macro-F1 of the last locked
    record of <prefix>fs{K}_knn10_s{seed}.
Summaries: per-seed shares in percent; mean and min-max over seeds 0-4.
Usage: python scripts/window_assistant.py [ow|cw|all]"""
import glob, io, json, os, re, sys, time, zipfile, collections
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = os.path.join(ROOT, 'results', 'window_assistant.json')
CACHE = os.environ.get('WA_CACHE', os.path.join(ROOT, 'results', 'cache', 'window_assistant'))
SEEDS = range(5)


def summ(v):
    v = [100.0 * float(x) for x in v]
    return dict(mean=round(float(np.mean(v)), 2), min=round(float(np.min(v)), 2), max=round(float(np.max(v)), 2), per_seed=[round(x, 2) for x in v])


# ----------------------------------------------------------------------------------------------------------- (a) open world
def ow_part():
    from ow_metrics import pr_auc
    recs = {}
    for line in open(os.path.join(ROOT, 'results', 'open_world_reject.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        if d.get('tag') == 'open_world': recs[(d['dataset'], d['K'], d['bg_class'], d['attacker'], d['seed'])] = d   # last record wins
    ref = json.load(open(os.path.join(ROOT, 'results', 'ow_windows.json')))
    acc = collections.defaultdict(lambda: collections.defaultdict(list)); maxdiff = 0.0; n_files = 0
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'runs', 'ow_scores', 'open_world_*.npz'))):
        mm = re.search(r'_(genai|ccma)_K(\d)_bg(\d)_(ours|scratch|lgbm)_s(\d)\.npz$', f)
        ds, K, bg, att, s = mm.group(1), int(mm.group(2)), int(mm.group(3)), mm.group(4), int(mm.group(5))
        z = np.load(f); rec = recs[(ds, K, bg, att, s)]; mon = rec['monitored']; C = len(mon); n_files += 1
        assert max(z['y'].max(), z['pred'].max(), z['pred_neg'].max()) < C
        asst = (lambda j: j // 2) if ds == 'genai' else (lambda j: j)
        score = np.concatenate([z['test_ens'], z['neg_ens']]); pred = np.concatenate([z['pred'], z['pred_neg']])
        sid = np.concatenate([z['sid_mon'], z['sid_neg']])
        pos_true = {int(a): int(b) for a, b in zip(z['sid_mon'], z['y'])}                      # index into monitored
        neg_cls = {}
        for a, k, c in zip(z['sid_neg'], z['kind_neg'], z['yclass_neg']):
            if k == 1: neg_cls[int(a)] = int(c)                                                  # original class id
        W = []
        for w in sorted(set(pos_true) | set(neg_cls)):
            m = sid == w; sc, pr = score[m], pred[m]; sv = np.zeros(C)
            for c in range(C):
                v = np.sort(sc[pr == c])[::-1][:5]; sv[c] = v.mean() if len(v) else 0.0
            ch = int(np.argmax(sv)); is_pos = w in pos_true
            true_orig = mon[pos_true[w]] if is_pos else neg_cls[w]
            W.append((is_pos, is_pos and pos_true[w] == ch, asst(mon[ch]) == asst(true_orig), float(sv[ch])))
        pos = np.array([w[0] for w in W]); ex = np.array([w[1] for w in W]); asc = np.array([w[2] for w in W]); ws = np.array([w[3] for w in W])
        r = ref[f'{ds}_K{K}_bg{bg}_{att}_s{s}']
        e_share = float(ex[pos].mean()); e_auc = pr_auc(ws[pos], ex[pos], ws[~pos])
        maxdiff = max(maxdiff, abs(e_share - r['win_correct']), abs(e_auc - r['win_pr_auc']))
        a_share = float(asc[pos].mean()); a_auc = pr_auc(ws[pos], asc[pos], ws[~pos])
        key = (ds, bg, att, K); A = acc[key]
        A['exact_share'].append(e_share); A['exact_pr_auc'].append(e_auc)
        A['asst_share'].append(a_share); A['asst_pr_auc'].append(a_auc)
        A['n_pos'].append(int(pos.sum())); A['n_neg'].append(int((~pos).sum())); A['n_seed'].append(s)
        if ds == 'genai':
            un_a = set(u // 2 for u in rec['unseen']); mon_a = set(c // 2 for c in mon)
            A['n_group'].append('other_modality' if un_a <= mon_a else ('unmonitored_assistant' if not (un_a & mon_a) else 'mixed'))
            A['neglabelled_own_asst_share'].append(float(asc[~pos].mean()))
            A['all_windows_asst_share'].append(float(asc.mean()))
    res = dict(n_files=n_files, check_max_abs_diff_vs_ow_windows=maxdiff, cells={})
    for (ds, bg, att, K), A in sorted(acc.items()):
        assert len(A['asst_share']) == 5, (ds, bg, att, K)
        cell = {k: summ(v) for k, v in A.items() if not k.startswith('n_')}
        cell['n_pos_per_seed'] = A['n_pos']; cell['n_neg_per_seed'] = A['n_neg']; assert A['n_seed'] == list(SEEDS)
        if ds == 'genai':   # by the class draw only (see the rules)
            cell['seed_group'] = A['n_group']
            cell['seed_groups'] = {g: dict(seeds=[s for s, gg in zip(A['n_seed'], A['n_group']) if gg == g],
                                                   **{m: summ([v for v, gg in zip(A[m], A['n_group']) if gg == g]) for m in ('asst_share', 'asst_pr_auc', 'exact_share', 'exact_pr_auc')})
                                           for g in sorted(set(A['n_group']))}
        res['cells'][f'{ds}_bg{bg}_{att}_K{K}'] = cell
    print(f'open world: {n_files} files, max |diff| vs ow_windows.json = {maxdiff:.2e}')
    for ds in ('genai', 'ccma'):
        for bg in (0, 1):
            for att in ('lgbm', 'scratch', 'ours'):
                line = f'{ds} bg{bg} {att:7s}'
                for K in (2, 4, 8):
                    c = res['cells'][f'{ds}_bg{bg}_{att}_K{K}']
                    line += (f' | K={K} asst {c["asst_share"]["mean"]:.1f} [{c["asst_share"]["min"]:.1f}-{c["asst_share"]["max"]:.1f}]'
                             f' AUC {c["asst_pr_auc"]["mean"]:.1f} [{c["asst_pr_auc"]["min"]:.1f}-{c["asst_pr_auc"]["max"]:.1f}] (exact {c["exact_share"]["mean"]:.1f}/{c["exact_pr_auc"]["mean"]:.1f})')
                    if ds == 'genai': line += f' negown {c["neglabelled_own_asst_share"]["mean"]:.1f}'
                print(line, flush=True)
    return res


# ------------------------------------------------------------------------------------------------------- (b) closed world
def cross_flows(ds):
    """Connections of the OTHER monitored packages inside each session (excluded from both derived files)."""
    os.makedirs(CACHE, exist_ok=True); cf = os.path.join(CACHE, f'{ds}_cross.npz')
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True); return z['meta'], z['meta_len'], z['session_id'], z['pkg']
    import pandas as pd
    from build_background import extract
    metas, lens, sids, pkgs = [], [], [], []; t0 = time.time()
    if ds == 'genai':
        from build_dataset import TARGET_PKG
        ANY = set().union(*TARGET_PKG.values()); RAW = os.path.join(ROOT, 'data', 'mirage2025genai')
        ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'genai_sessions.csv'))
        file2sid = dict(zip(ses.file, ses.session_id)); sid2app = dict(zip(ses.session_id, ses.app_dir))
        for f in sorted(glob.glob(os.path.join(RAW, '*', '*', '*.json'))):
            rel = os.path.relpath(f, RAW).replace('\\', '/')
            if rel not in file2sid: continue
            sid = int(file2sid[rel]); own = TARGET_PKG[sid2app[sid]]; d = json.load(open(f))
            for key, b in d.items():
                pkg = b['flow_metadata']['BF_label']
                if pkg in ANY and pkg not in own:
                    r = extract(b, key)
                    if r is None: continue
                    metas.append(r[2]); lens.append(r[3]); sids.append(sid); pkgs.append(pkg)
    else:
        from build_dataset_ccma import APPS
        targets = set(APPS.values()); z = zipfile.ZipFile(os.path.join(ROOT, 'data', 'MIRAGE-COVID-CCMA-2022.zip'))
        ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'ccma_sessions.csv')); ses = ses[ses.n_kept > 0]
        file2sid = dict(zip(ses.file, ses.session_id))
        for app, own in APPS.items():
            inner = zipfile.ZipFile(io.BytesIO(z.read(f'MIRAGE-COVID-CCMA-2022/Raw_JSON/{app}.zip')))
            for fn in sorted(n for n in inner.namelist() if n.endswith('.json')):
                if fn not in file2sid: continue
                sid = int(file2sid[fn]); d = json.loads(inner.read(fn))
                for key, b in d.items():
                    pkg = b['flow_metadata']['BF_label']
                    if pkg in targets and pkg != own:
                        r = extract(b, key)
                        if r is None: continue
                        metas.append(r[2]); lens.append(r[3]); sids.append(sid); pkgs.append(pkg)
            del inner
    meta = np.array(metas, np.float32).reshape(-1, 64, 4); ml = np.array(lens, np.int32); sid = np.array(sids, np.int64); pk = np.array(pkgs)
    np.savez_compressed(cf, meta=meta, meta_len=ml, session_id=sid, pkg=pk)
    print(f'{ds}: {len(ml)} cross-monitored connections in {len(set(sids))} sessions {collections.Counter(pkgs).most_common(6)} ({time.time() - t0:.0f}s)', flush=True)
    return meta, ml, sid, pk


def enc_feats(meta, meta_len, L=64):
    """src/train_meta.features without defence, for any meta array."""
    m = meta[:, :L]; x = np.zeros_like(m)
    x[..., 0] = m[..., 0]; x[..., 1] = np.log1p(m[..., 1]); x[..., 2] = np.log1p(m[..., 2]); x[..., 3] = np.log1p(m[..., 3] * 1000.0)
    x[~(np.arange(L)[None, :] < np.minimum(meta_len, L)[:, None])] = 0.0
    return x.astype(np.float32)


def nn1(X, Y, Q):
    out = np.empty(len(Q), np.int64)
    for b in range(0, len(Q), 256): out[b:b + 256] = Y[np.abs(Q[b:b + 256, None, :] - X[None]).sum(-1).argmin(1)]
    return out


def cw_part(dsets=('genai', 'ccma'), Ks=(1, 2, 4, 8)):
    import torch, lightgbm as lgb
    from sklearn.metrics import f1_score
    from data_mfr import GenAIData
    from trivial_baselines import feats_knn
    from train_baselines import stat_features
    from train_meta import features as meta_features
    from pretrain_meta import MetaEncoder
    from train_meta_ssl import Classifier, predict
    LOCKED = {}
    for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
        j = json.loads(l)
        if not str(j.get('tag', '')).startswith('smoke'): LOCKED[j['run_id']] = j['test']
    pre = torch.load(os.path.join(ROOT, 'weights', 'meta_ssl_mpm_XL.pt'), weights_only=False); ecfg = pre['cfg']; dev = torch.device('cuda')
    res = dict(checks=dict(lgbm_pred_mismatch=[], enc_pred_mismatch=[], knn_f1_absdiff_max=0.0, enc_feats_equal=[]), cells={}, windows={})
    for ds in dsets:
        data = GenAIData(ROOT, prefix=ds); pre_ = '' if ds == 'genai' else 'ccma_'
        y = data.y_joint if ds == 'genai' else data.y_app; NC = data.n_joint if ds == 'genai' else data.n_app
        NA = 3 if ds == 'genai' else 9; BG = NC
        to_a = (lambda c: np.asarray(c) // 2) if ds == 'genai' else (lambda c: np.asarray(c))
        bgz = np.load(os.path.join(ROOT, 'data', 'derived', f'{ds}_background.npz'), allow_pickle=True)
        bmeta, blen, bsid = bgz['meta'], bgz['meta_len'], bgz['session_id']
        cmeta, clen, csid, cpkg = cross_flows(ds)
        t0 = time.time()
        Fk = feats_knn(data.meta, data.meta_len); Fkb = feats_knn(bmeta, blen); Fkc = feats_knn(cmeta, clen) if len(clen) else np.zeros((0, 40), np.float32)
        Fs = stat_features(data.meta, data.meta_len); Fsb = stat_features(bmeta, blen); Fsc = stat_features(cmeta, clen) if len(clen) else np.zeros((0, Fs.shape[1]), np.float32)
        Xe = meta_features(data, 64); Xe2 = enc_feats(data.meta, data.meta_len); res['checks']['enc_feats_equal'].append(bool(np.array_equal(Xe, Xe2)))
        Xeb = enc_feats(bmeta, blen); Xec = enc_feats(cmeta, clen) if len(clen) else np.zeros((0, 64, 4), np.float32)
        print(f'{ds}: features {time.time() - t0:.0f}s; target {len(y)}, background {len(blen)}, cross {len(clen)}', flush=True)
        for K in Ks:
            acc = collections.defaultdict(list); wstats = collections.defaultdict(list)
            for s in SEEDS:
                sp = os.path.join(ROOT, 'configs', 'splits', f'{pre_}fewshot_k{K}_s{s}.json')
                idx, _ = data.split_indices(sp); tr, it = idx['train'], idx['test']
                tr_sess = np.unique(data.session_id[tr]); te_sess = np.unique(data.session_id[it])
                btr = np.where(np.isin(bsid, tr_sess))[0]; bte = np.where(np.isin(bsid, te_sess))[0]; cte = np.where(np.isin(csid, te_sess))[0]
                # ---- per-connection predictions of every observer on: target test (it), background test (bte), cross test (cte)
                P = {}
                # 1-NN
                pk = nn1(Fk[tr], y[tr], np.concatenate([Fk[it], Fkb[bte], Fkc[cte]]))
                Xb_ = np.concatenate([Fk[tr], Fkb[btr]]); yb_ = np.concatenate([y[tr], np.full(len(btr), BG)])
                pkB = nn1(Xb_, yb_, np.concatenate([Fk[it], Fkb[bte], Fkc[cte]]))
                P['knn10'] = (pk, pkB)
                lk = LOCKED.get(f'{pre_}fs{K}_knn10_s{s}')
                if lk is not None:
                    res['checks']['knn_f1_absdiff_max'] = max(res['checks']['knn_f1_absdiff_max'], abs(f1_score(y[it], pk[:len(it)], average='macro') - lk['macro_f1']))
                # LightGBM
                mu, sd = Fs[tr].mean(0), Fs[tr].std(0) + 1e-6; st = lambda F: (F - mu) / sd
                Q = st(np.concatenate([Fs[it], Fsb[bte], Fsc[cte]]))
                kw = dict(n_estimators=1600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=s, verbose=-1)
                clf = lgb.LGBMClassifier(**kw); clf.fit(st(Fs[tr]), y[tr]); pl = clf.predict(Q).astype(np.int64)
                clfB = lgb.LGBMClassifier(**kw); clfB.fit(st(np.concatenate([Fs[tr], Fsb[btr]])), np.concatenate([y[tr], np.full(len(btr), BG)])); plB = clfB.predict(Q).astype(np.int64)
                P['lgbm'] = (pl, plB)
                zp = np.load(os.path.join(ROOT, 'results', 'runs', f'{pre_}fs{K}_lgbm_meta64_s{s}', 'test_preds.npz'))
                assert np.array_equal(zp['idx'], it)
                nm = int((zp['pred'].astype(np.int64) != pl[:len(it)]).sum())
                if nm: res['checks']['lgbm_pred_mismatch'].append([f'{pre_}fs{K}_lgbm_meta64_s{s}', nm, len(it)])
                # encoder (checkpoint, inference only)
                run = f'{pre_}fs{K}_sslXL_s{s}'; ck = torch.load(os.path.join(ROOT, 'results', 'runs', run, 'best.pt'), weights_only=False)
                assert ck['args']['split'].replace('\\', '/').endswith(f'{pre_}fewshot_k{K}_s{s}.json') and ck['args']['seed'] == s and ck['args']['select'] == 'last'
                model = Classifier(MetaEncoder(ecfg['d'], ecfg['layers'], ecfg['heads'], ecfg['ff'], ecfg['L']), NC, 0.3)
                model.load_state_dict(ck['model']); model.to(dev).eval(); emu, esd = (np.asarray(t) for t in ck['norm'])   # dtype as in training

                def epred(Xr):
                    if len(Xr) == 0: return np.zeros(0, np.int64)
                    V = Xr[..., 0] != 0.0; X = torch.from_numpy(((Xr - emu) / esd * V[..., None]).astype(np.float32))
                    return predict(model, X, torch.from_numpy(V), dev).astype(np.int64)
                pe = np.concatenate([epred(Xe[it]), epred(Xeb[bte]), epred(Xec[cte])])
                zp = np.load(os.path.join(ROOT, 'results', 'runs', run, 'test_preds.npz')); assert np.array_equal(zp['idx'], it)
                nm = int((zp['pred'].astype(np.int64) != pe[:len(it)]).sum())
                if nm: res['checks']['enc_pred_mismatch'].append([run, nm, len(it)])
                P['encoder'] = (pe, None)
                del model; torch.cuda.empty_cache()
                # ---- windows
                nT, nB = len(it), len(bte)
                seg_sid = np.concatenate([data.session_id[it], bsid[bte], csid[cte]])
                seg_kind = np.concatenate([np.zeros(nT, int), np.ones(nB, int), np.full(len(cte), 2)])   # 0 app, 1 background, 2 other monitored app
                for obs, (pA, pB) in P.items():
                    ok = collections.Counter(); ties = 0; allbg = 0; allbg_p = 0; per_a = collections.defaultdict(lambda: [0, 0])
                    for w in te_sess:
                        m = seg_sid == w; k = seg_kind[m]; true = int(to_a(np.bincount(y[it][data.session_id[it] == w]).argmax()))
                        a_all = to_a(pA[m]); a_nox = a_all[k < 2]; a_app = a_all[k == 0]
                        cnt = np.bincount(a_nox, minlength=NA); vA = int(cnt.argmax()); ties += int((cnt == cnt.max()).sum() > 1)
                        ok['A'] += vA == true; ok['A+'] += int(np.bincount(a_all, minlength=NA).argmax()) == true
                        ok['O'] += int(np.bincount(a_app, minlength=NA).argmax()) == true
                        per_a[true][0] += vA == true; per_a[true][1] += 1
                        if pB is not None:
                            b_all = pB[m]
                            for tag, sel in (('B', k < 2), ('B+', np.ones(len(k), bool))):
                                keep = b_all[sel]; keep = keep[keep != BG]
                                if len(keep): ok[tag] += int(np.bincount(to_a(keep), minlength=NA).argmax()) == true
                                elif tag == 'B': allbg += 1
                                else: allbg_p += 1
                    n = len(te_sess)
                    for tag in ('A', 'A+', 'O') + (('B', 'B+') if pB is not None else ()):
                        acc[(obs, tag)].append(ok[tag] / n)
                    wstats[(obs, 'ties_A')].append(ties); wstats[(obs, 'n_windows')].append(n)
                    for a_ in range(NA):
                        if per_a[a_][1]: wstats[(obs, f'A_correct_true{a_}')].append(per_a[a_][0] / per_a[a_][1])
                    if pB is not None:
                        wstats[(obs, 'B_windows_all_bg')].append(allbg); wstats[(obs, 'B+_windows_all_bg')].append(allbg_p)
                        wstats[(obs, 'B_bg_conn_pred_bg')].append(float((pB[nT:nT + nB] == BG).mean()) if nB else 0.0)
                        wstats[(obs, 'B_app_conn_pred_bg')].append(float((pB[:nT] == BG).mean()))
                    ab = to_a(pA[nT:nT + nB])
                    for a_ in range(NA): wstats[(obs, f'A_bg_conn_to_asst{a_}')].append(float((ab == a_).mean()) if nB else 0.0)
                cnt_app = np.bincount(np.searchsorted(te_sess, data.session_id[it]), minlength=len(te_sess))
                cnt_bg = np.bincount(np.searchsorted(te_sess, bsid[bte]), minlength=len(te_sess))
                cnt_cr = np.bincount(np.searchsorted(te_sess, csid[cte]), minlength=len(te_sess))
                wstats[('all', 'bg_share_of_window')].append(float(np.mean(cnt_bg / (cnt_app + cnt_bg))))
                wstats[('all', 'median_app_conn')].append(float(np.median(cnt_app))); wstats[('all', 'median_bg_conn')].append(float(np.median(cnt_bg)))
                wstats[('all', 'windows_with_cross')].append(int((cnt_cr > 0).sum())); wstats[('all', 'n_cross_conn')].append(int(cnt_cr.sum()))
                print(f'{ds} K={K} s={s} ' + ' | '.join(f'{o}: ' + ' '.join(f'{t} {100 * acc[(o, t)][-1]:.1f}' for t in ('O', 'A', 'B', 'A+', 'B+') if (o, t) in acc)
                                                     for o in P) + f' ({time.time() - t0:.0f}s)', flush=True)
            for (obs, tag), v in acc.items():
                res['cells'][f'{ds}_{obs}_K{K}_{tag}'] = summ(v)
            frac = lambda k: any(t in k for t in ('_correct_', '_conn_pred_bg', '_conn_to_asst', 'bg_share'))
            res['windows'][f'{ds}_K{K}'] = {f'{o}|{k}': (summ(v) if frac(k) else v) for (o, k), v in wstats.items()}
    return res


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    out['rules'] = __doc__.split('Rules:')[1].split('Usage:')[0].strip()
    if part in ('ow', 'all'):
        out['open_world'] = ow_part(); json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    if part in ('cw', 'all') or part.startswith('cw:'):
        dsets = tuple(part.split(':')[1].split(',')) if ':' in part else ('genai', 'ccma')
        r = cw_part(dsets)
        cw = out.get('closed_world', dict(checks={}, cells={}, windows={}))
        for k in ('cells', 'windows'): cw.setdefault(k, {}).update(r[k])
        for k, v in r['checks'].items(): cw['checks'][k + '_' + '_'.join(dsets)] = v
        out['closed_world'] = cw; json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('written', OUT)
