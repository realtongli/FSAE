"""Google-app check for the 'Gemini' class of MIRAGE-GenAI-2025 (read-only for data, models and existing results).

The Gemini class is the Google app (com.google.android.googlequicksearchbox) that hosts Gemini, and no capture session has
Google-app traffic without Gemini use. The campaign does, however, attribute some connections inside ChatGPT and Copilot
sessions to the Google app. This script
 (a) finds them, separates exact socket attributions from majority-vote ones, flags the mis-attributed ones (server name of
     ChatGPT/Copilot), extracts the same 64-packet features, and scores them with the K-session observers (pre-trained
     encoder, LightGBM, 1-NN) for K = 1/2/4/8 and seeds 0-4; plus, from the saved open-world score files, how often Google
     Play-services and other Google background connections of ChatGPT/Copilot test sessions are assigned to a monitored
     Gemini class, with and without the deployable 5%-FPR report threshold; plus the same for the Google-app connections
     under the open-world LightGBM (replayed exactly, checked against the saved scores);
 (b) per-assistant F1 and session accuracy of the three observers;
 (c) the Gemini session vote after re-labelling proactivebackend-pa.googleapis.com as a Google-app
     endpoint, with a check of every other Gemini-named host against Google-app contacts in ChatGPT/Copilot sessions.

Rules:
 R1 Google-app connection: biflow with BF_label == com.google.android.googlequicksearchbox inside a ChatGPT or Copilot session
    (generic and controlled) of MIRAGE-GenAI-2025, with >= 1 payload packet (the inclusion rule of the monitored flows);
    features = first 64 packets exactly as scripts/build_background.extract (identical to build_dataset.py).
    Attribution: BF_labeling_type == 'exact' vs any other value (majority vote over sockets).
 R2 Server name (ground truth only, never read by an observer): construct_validity.server_name (TLS/QUIC SNI, HTTP Host).
    mis-attributed = name matches ChatGPT's or Copilot's own-endpoint regex (construct_validity.ASSIST_RE[0|1]) or their
    image-delivery hosts (files.oaiusercontent.com, *.mm.bing.net).
    Google destination = name matches the Google-owned domain regex (construct_validity.VENDOR_RE[2] plus Google-owned
    telemetry: app-measurement, google-analytics, crashlytics, doubleclick, googleadservices, googlesyndication), or no name
    and a destination /24 that serves a named Google host somewhere in the campaign and no named ChatGPT/Copilot host.
    Headline subset = Google destination and not mis-attributed ('genuine'); every subset is reported.
 R3 Observers: saved results/runs/fs{K}_sslXL_s{s}/best.pt (argmax); LightGBM re-fit exactly as fs{K}_lgbm_meta64_s{s}
    (train_baselines.py, 1,600 rounds, lr 0.03, standardised on the K training sessions, random_state = seed), its test
    predictions checked against the saved ones; 1-NN on the first 10 packets (L1, trivial_baselines.feats_knn), its test
    macro-F1 checked against the fs{K}_knn10_s{s} records. Encoder test predictions checked against the saved ones.
 R4 Gemini assignment = 6-class argmax in {Gemini text, Gemini image}; session vote = majority of the assistant decisions
    (6-class prediction // 2) over the session's connections of the subset, ties to the lower index. Mean and seed range.
 R5 Open world: saved open_world GenAI score files (3 monitored of 6 classes, budget protocol, 'ens' score), only runs whose
    monitored set contains a Gemini class; Google background = package com.google.* or com.android.vending / chrome /
    webview; Play services = com.google.android.gms; non-Gemini test sessions = ChatGPT/Copilot test sessions; threshold =
    the run's own selected_t (0.95 quantile of its calibration negatives' ens scores = 5% FPR); 'assigned' = argmax over
    the monitored classes is a Gemini class; 'reported' = assigned and ens >= threshold. The open-world LightGBM is
    replayed from open_world_reject.py (same seeds, pools, caps) and used for the Google-app connections only if its
    predictions and scores reproduce the saved file.
 R6 Per-assistant: assistant decision = 6-class prediction // 2; per-assistant F1 (labels 0-2); session accuracy = majority
    vote per test session (assistant; and 6-class, as scripts/threat_stats.py).
 R7 Gemini-specific hosts: proactivebackend-pa.googleapis.com is a Google-app endpoint. Every other host (digits folded)
    matching the Gemini regex (gemini|bard|makersuite|proactivebackend|generativelanguage|assistant) and the image-delivery
    host lh#.googleusercontent.com stay Gemini-specific / image-delivery only if the Google app contacts them in no
    ChatGPT or Copilot session (generic or controlled). Vote over only those connections of each Gemini test session;
    sessions without any such connection are counted (and, in a second figure, scored as errors). The original rule
    (construct_validity categories 'assistant' + 'image') is recomputed as a reproduction check.
Writes results/google_app_check.json."""
import argparse, collections, json, os, pickle, re, sys, types
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from construct_validity import server_name, fold, selftest, category, IMG_RE, ASSIST_RE, VENDOR_RE
from build_background import extract
from train_meta import features as meta_features
from train_baselines import stat_features
from trivial_baselines import feats_knn
from pretrain_meta import MetaEncoder
from open_world_apps import load_bg
from ow_metrics import ens_tiefree
import train_meta_ssl
import lightgbm as lgb

RAW = os.path.join(ROOT, 'data', 'mirage2025genai')
GAPP = 'com.google.android.googlequicksearchbox'
APPS = ['ChatGPT', 'Copilot', 'Gemini']; APP_OF_DIR = {'Chatgpt': 0, 'Copilot': 1, 'Gemini': 2}
KS, SEEDS, OBS, L = (1, 2, 4, 8), range(5), ('ours', 'lgbm', 'knn1'), 64
GOOGLE_RE = re.compile(VENDOR_RE[2].pattern + r'|(^|\.)(app-measurement\.com|google-analytics\.com|crashlytics\.com|doubleclick\.net|googleadservices\.com|googlesyndication\.com)$')
AC_IMG_RE = re.compile(r'^files\.oaiusercontent\.com$|\.mm\.bing\.net$')
GEM_RE = ASSIST_RE[2]; PROACTIVE = 'proactivebackend-pa.googleapis.com'; GEM_IMG = 'lh#.googleusercontent.com'


def pfx(ip):
    return ':'.join(ip.split(':')[:4]) if ':' in ip else '.'.join(ip.split('.')[:3])


def is_ac_name(n):  # ChatGPT / Copilot own endpoint or image host
    return bool(n) and bool(ASSIST_RE[0].search(n) or ASSIST_RE[1].search(n) or AC_IMG_RE.search(n))


def is_google_pkg(p):
    return p.startswith('com.google.') or p in ('com.android.vending', 'com.android.chrome', 'com.android.webview')


def rng3(v):  # mean, min, max in percent
    v = 100 * np.asarray(v, float)
    return dict(mean=round(float(np.mean(v)), 2), min=round(float(np.min(v)), 2), max=round(float(np.max(v)), 2))


def walk(D, cache):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    ses = D.sessions; N = len(D.meta)
    row_of = {(f, k): i for i, (f, k) in enumerate(zip(ses.set_index('session_id').loc[D.index.session_id, 'file'].values, D.index.biflow_key))}
    H = np.array([None] * N, dtype=object)
    contact = collections.defaultdict(set)      # (session_id, package) -> folded server names
    g24, a24 = set(), set()                     # /24 prefixes serving named Google / named ChatGPT-Copilot hosts
    G = collections.defaultdict(list); inv = collections.Counter()
    for sid, f, adir in zip(ses.session_id, ses.file, ses.app_dir):
        d = json.load(open(os.path.join(RAW, f)))
        for key, b in d.items():
            proto = int(key.split(',')[-1]); pkg = b['flow_metadata']['BF_label']; dst = key.split(',')[2]
            nm, how = server_name(b['packet_data'], proto)
            if nm:
                contact[(int(sid), pkg)].add(fold(nm))
                if GOOGLE_RE.search(nm): g24.add(pfx(dst))
                if is_ac_name(nm): a24.add(pfx(dst))
            if (f, key) in row_of: H[row_of[(f, key)]] = nm; inv[('matched_labelled_rows', 'all')] += 1
            if APP_OF_DIR[adir] != 2 and pkg == GAPP:
                inv[('all', adir)] += 1
                r = extract(b, key)
                if r is None: inv[('no_payload', adir)] += 1; continue
                inv[('kept', adir)] += 1
                G['meta'].append(r[2]); G['meta_len'].append(r[3]); G['sid'].append(int(sid)); G['lt'].append(b['flow_metadata'].get('BF_labeling_type'))
                G['name'].append(nm); G['how'].append(how); G['dst'].append(dst); G['proto'].append(proto)
        print(f'walked session {sid} ({f.split("/")[1]})', flush=True) if sid % 40 == 0 else None
    G = {k: np.array(v, dtype=object if k in ('name', 'how', 'lt', 'dst') else None) for k, v in G.items()}
    G['meta'] = G['meta'].astype(np.float32); G['meta_len'] = G['meta_len'].astype(int)
    res = dict(H=H, contact=dict(contact), g24=g24, a24=a24, G=G, inv={f'{a}|{b}': n for (a, b), n in inv.items()})
    if cache: pickle.dump(res, open(cache, 'wb'))
    return res


def last_records(fn='locked_test_metrics.jsonl'):
    recs = {}
    for l in open(os.path.join(ROOT, 'results', fn), encoding='utf-8'):
        try: r = json.loads(l)
        except Exception: continue
        if str(r.get('tag', '')).startswith('smoke'): continue
        recs[r['run_id']] = r
    return recs


def enc_model(K, s):
    ck = torch.load(os.path.join(ROOT, 'results', 'runs', f'fs{K}_sslXL_s{s}', 'best.pt'), weights_only=False)
    c = torch.load(os.path.join(ROOT, ck['args']['init']), weights_only=False)['cfg']
    m = train_meta_ssl.Classifier(MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L']), 6, ck['args']['drop_head']).cuda()
    m.load_state_dict(ck['model']); m.eval(); return m, ck['norm']


def enc_predict(m, norm, Xr):
    mu, sd = norm; V = Xr[..., 0] != 0; X = ((Xr - mu) / sd * V[..., None]).astype(np.float32); out = []
    with torch.no_grad():
        for b in range(0, len(X), 512):
            out.append(m(torch.from_numpy(X[b:b + 512]).cuda(), torch.from_numpy(V[b:b + 512]).cuda()).argmax(1).cpu().numpy())
    return np.concatenate(out)


def lgbm_fit(F, y, tr, s):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    clf = lgb.LGBMClassifier(n_estimators=1600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=s, verbose=-1)
    clf.fit((F[tr] - mu) / sd, y[tr]); return lambda Q: clf.predict((Q - mu) / sd)


def nn1(Fk, y, tr, Q):
    out = np.empty(len(Q), int)
    for b in range(0, len(Q), 512): out[b:b + 512] = y[tr][np.abs(Q[b:b + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)]
    return out


def vote(p_app, groups):  # groups: array of session ids per connection -> dict sid -> voted assistant
    return {int(g): int(np.bincount(p_app[groups == g], minlength=3).argmax()) for g in np.unique(groups)}


# ------------------------------------------------------------------ open-world LightGBM replay (open_world_reject.py)
def ow_lgbm_replay(data, Fall, bg, K, seed, bgc, F_extra, n_mon=3, n_proxy=1, rounds=1600):
    bg_sid, bg_meta, bg_len = bg
    y_all = data.y_joint; NCfull = data.n_joint
    rng = np.random.RandomState(700 + seed); perm = rng.permutation(NCfull)
    mon = np.sort(perm[:n_mon]); proxy = np.sort(perm[n_mon:n_mon + n_proxy]); unseen = np.sort(perm[n_mon + n_proxy:])
    remap = {int(c): i for i, c in enumerate(mon)}; C = len(mon); BGC = C
    split = os.path.join(ROOT, 'configs', 'splits', f'fewshot_k{K}_s{seed}.json')
    idx, _ = data.split_indices(split); sp = json.load(open(split))
    sids = {k: np.array([data.file2sid[f] for f in sp[k]]) for k in ('train', 'val', 'test')}
    bgi = {k: np.where(np.isin(bg_sid, sids[k]))[0] for k in sids}
    sess_lab = dict(zip(data.session_id.tolist(), y_all.tolist()))
    own_cls = set(int(c) for c in (np.concatenate([mon, proxy]) if bgc else mon))
    own_train_sids = [s_ for s_ in sids['train'] if sess_lab.get(int(s_), -1) in own_cls]
    bg_own = np.where(np.isin(bg_sid, own_train_sids))[0]
    sel = lambda ii, cls: ii[np.isin(y_all[ii], cls)]
    tr = sel(idx['train'], mon); tr_px = sel(idx['train'], proxy); va_px = sel(idx['val'], proxy)
    te_mon = sel(idx['test'], mon); te_un = sel(idx['test'], unseen)
    ytr = np.array([remap[int(c)] for c in y_all[tr]])
    rb = np.random.RandomState(900 + seed); cap = max(len(tr), 200)
    n_neg_tr = len(bgi['train']) + len(tr_px)
    if n_neg_tr > cap: rb.choice(n_neg_tr, cap, replace=False)
    n_neg_va = len(bgi['val']) + len(va_px)
    if n_neg_va > 3000: rb.choice(n_neg_va, 3000, replace=False)
    pool_s = [bg_sid[bg_own]]; pool_src = [('bg', bg_own)]
    if bgc: pool_s.append(data.session_id[tr_px]); pool_src.append(('px', tr_px))
    PS = np.concatenate(pool_s); us = np.unique(PS); rs = np.random.RandomState(1100 + seed); rs.shuffle(us)
    if len(us) >= 2: cal_sess = set(us[: len(us) // 2]); is_cal = np.array([x in cal_sess for x in PS])
    else: is_cal = rs.rand(len(PS)) < 0.5
    if int((~is_cal).sum()) > cap: rb.choice(int((~is_cal).sum()), cap, replace=False)
    src_all = np.concatenate([np.stack([np.full(len(ii), k), ii], 1) for k, (_, ii) in enumerate(pool_src)])
    cal_src = src_all[is_cal]; fit_src_all = src_all[~is_cal]
    mu_, sd_ = Fall[tr].mean(0), Fall[tr].std(0) + 1e-6; st_ = lambda F: (F - mu_) / sd_

    def stat_of(src):
        mm = np.concatenate([bg_meta[src[src[:, 0] == 0, 1]], data.meta[src[src[:, 0] == 1, 1]][:, :L]])
        ll = np.concatenate([bg_len[src[src[:, 0] == 0, 1]], np.minimum(data.meta_len[src[src[:, 0] == 1, 1]], L)])
        return stat_features(mm, ll)
    Fneg_tr, Fneg_va = stat_of(fit_src_all), stat_of(cal_src)
    Fneg_te = stat_features(np.concatenate([bg_meta[bgi['test']], data.meta[te_un][:, :L]]), np.concatenate([bg_len[bgi['test']], np.minimum(data.meta_len[te_un], L)]))
    Ftr_, ytr_ = Fall[tr], ytr
    if bgc:
        if len(Fneg_tr) > cap: Fneg_tr = Fneg_tr[rb.choice(len(Fneg_tr), cap, replace=False)]
        Ftr_ = np.concatenate([Ftr_, Fneg_tr]); ytr_ = np.concatenate([ytr_, np.full(len(Fneg_tr), BGC)])
    clf = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=seed, verbose=-1)
    clf.fit(st_(Ftr_), ytr_)

    def sc2(F):
        P = clf.predict_proba(st_(F)); Pm = P[:, :C]; pred = Pm.argmax(1)
        s = {'msp': Pm.max(1) / np.maximum(Pm.sum(1), 1e-9) * Pm.sum(1), 'energy': np.log(np.maximum(Pm, 1e-12)).max(1), 'maxlogit': Pm.max(1)}
        if P.shape[1] > C: s['bgprob'] = 1.0 - P[:, C]
        return pred, s
    _, sv_neg = sc2(Fneg_va); pt, st = sc2(Fall[te_mon]); pn, sn = sc2(Fneg_te); pg, sg = sc2(F_extra)
    cal_e, te_e, ne_e, g_e = ens_tiefree([sv_neg, st, sn, sg], sv_neg)
    return dict(mon=mon, pred=pt, test_ens=te_e, pred_neg=pn, neg_ens=ne_e, cal_ens=cal_e, pred_g=pg, ens_g=g_e)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', default='', help='optional pickle for the raw-JSON walk (scratch only)'); a = ap.parse_args()
    selftest()
    D = GenAIData(ROOT, prefix='genai'); ses = D.sessions; y6 = D.y_joint
    sid_app = dict(zip(ses.session_id.astype(int), [APP_OF_DIR[x] for x in ses.app_dir]))
    sid_part = dict(zip(ses.session_id.astype(int), ses.part)); sid_cls = dict(zip(ses.session_id.astype(int), ses.class_dir))
    W = walk(D, a.cache); H, contact, g24, a24, G = W['H'], W['contact'], W['g24'], W['a24'], W['G']
    out = dict(protocol=__doc__, checks={}, a={}, b={}, c={})

    # ------------------------------------------------------------ (a) inventory of the Google-app connections
    gs, gname, glt, gdst = G['sid'], G['name'], G['lt'], G['dst']; ng = len(gs)
    exact = glt == 'exact'
    mis = np.array([is_ac_name(n) for n in gname])
    gname_google = np.array([bool(n) and bool(GOOGLE_RE.search(n)) for n in gname])
    unnamed = np.array([n is None for n in gname])
    un_gdest = unnamed & np.array([(pfx(d) in g24) and (pfx(d) not in a24) for d in gdst])
    genuine = ~mis & (gname_google | un_gdest)
    test_sids = set(int(D.file2sid[f]) for f in json.load(open(os.path.join(ROOT, 'configs', 'splits', 'fewshot_k4_s0.json')))['test'])
    in_test = np.array([int(s) in test_sids for s in gs])
    gapp_ = np.array([sid_app[int(s)] for s in gs])
    subsets = collections.OrderedDict([
        ('all', np.ones(ng, bool)), ('exact', exact), ('majority_vote', ~exact),
        ('misattributed_chatgpt_copilot_name', mis), ('misattributed_exact', mis & exact),
        ('genuine_google_destination', genuine), ('genuine_exact', genuine & exact), ('genuine_majority_vote', genuine & ~exact),
        ('genuine_named_google', genuine & gname_google), ('genuine_in_chatgpt_sessions', genuine & (gapp_ == 0)),
        ('genuine_in_copilot_sessions', genuine & (gapp_ == 1)), ('genuine_test_sessions', genuine & in_test),
        ('neither_misattributed_nor_google', ~mis & ~genuine)])
    per_sess = collections.defaultdict(lambda: collections.Counter())
    for i in range(ng):
        k = 'misattributed' if mis[i] else ('genuine' if genuine[i] else 'other')
        per_sess[int(gs[i])][f'{k}_{"exact" if exact[i] else "majority"}'] += 1
    inv = dict(raw_counts=W['inv'], n_kept=int(ng), labeling_types=dict(collections.Counter(glt.tolist())),
               subsets={k: dict(n=int(m.sum()), n_sessions=int(len(np.unique(gs[m])))) for k, m in subsets.items()},
               per_session={str(s): dict(class_dir=sid_cls[s], part=sid_part[s], test=s in test_sids, **dict(c)) for s, c in sorted(per_sess.items())},
               names={k: dict(collections.Counter(fold(n) if n else '<none>' for n in gname[m]).most_common(40)) for k, m in
                      (('misattributed', mis), ('genuine', genuine), ('other', ~mis & ~genuine))},
               n_unnamed=int(unnamed.sum()), n_unnamed_google_dest=int(un_gdest.sum()))
    out['a']['inventory'] = inv
    print('Google-app connections in ChatGPT/Copilot sessions:', json.dumps(inv['subsets']))

    # ------------------------------------------------------------ predictions of the three observers
    ns_g = types.SimpleNamespace(meta=G['meta'], meta_len=G['meta_len'])
    Xall = meta_features(D, L); Xg = meta_features(ns_g, L)
    Fall = stat_features(D.meta, D.meta_len); Fg = stat_features(G['meta'], G['meta_len'])
    Kall = feats_knn(D.meta, D.meta_len); Kg = feats_knn(G['meta'], G['meta_len'])
    recs = last_records(); trecs = last_records('metrics.jsonl'); P = {}; T = None; chk = collections.defaultdict(dict)
    for K in KS:
        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'fewshot_k{K}_s{s}.json')); tr, it = idx['train'], idx['test']
            if T is None: T = it
            assert np.array_equal(T, it)
            m, norm = enc_model(K, s); pe_t = enc_predict(m, norm, Xall[it]); pe_g = enc_predict(m, norm, Xg); del m
            z = np.load(os.path.join(ROOT, 'results', 'runs', f'fs{K}_sslXL_s{s}', 'test_preds.npz')); assert np.array_equal(z['idx'], it)
            chk['encoder_test_agreement'][f'K{K}_s{s}'] = float(np.mean(pe_t == z['pred']))
            f = lgbm_fit(Fall, y6, tr, s); pl_t = f(Fall[it]); pl_g = f(Fg)
            zl = np.load(os.path.join(ROOT, 'results', 'runs', f'fs{K}_lgbm_meta64_s{s}', 'test_preds.npz')); assert np.array_equal(zl['idx'], it)
            chk['lgbm_test_agreement'][f'K{K}_s{s}'] = float(np.mean(pl_t == zl['pred']))
            rl = trecs.get(f'fs{K}_lgbm_meta64_s{s}', {}); chk['lgbm_record_rounds'][f'K{K}_s{s}'] = (rl.get('cfg') or {}).get('n_est')
            pk_t = nn1(Kall, y6, tr, Kall[it]); pk_g = nn1(Kall, y6, tr, Kg)
            rk = recs.get(f'fs{K}_knn10_s{s}')
            chk['knn1_macro_f1_minus_record'][f'K{K}_s{s}'] = None if rk is None else float(f1_score(y6[it], pk_t, average='macro') - rk['test']['macro_f1'])
            for o, pt_, pg_ in (('ours', pe_t, pe_g), ('lgbm', pl_t, pl_g), ('knn1', pk_t, pk_g)):
                P[(o, K, s)] = dict(test=pt_, g=pg_)
            print(f'K={K} s={s} enc agree {chk["encoder_test_agreement"][f"K{K}_s{s}"]:.4f} lgbm agree {chk["lgbm_test_agreement"][f"K{K}_s{s}"]:.4f} '
                  f'knn dF1 {chk["knn1_macro_f1_minus_record"][f"K{K}_s{s}"]}', flush=True)
    out['checks'] = {k: dict(v) for k, v in chk.items()}

    # ------------------------------------------------------------ (a) scores of the Google-app connections
    sc = {}
    for sn, msk in subsets.items():
        e = dict(n=int(msk.sum()), n_sessions=int(len(np.unique(gs[msk]))))
        if msk.any():
            for K in KS:
                ek = {}
                for o in OBS:
                    shares = collections.defaultdict(list); cnt_g = []; sv_g = []
                    for s in SEEDS:
                        p = P[(o, K, s)]['g'][msk]
                        for nm_, c_ in (('gemini', None), ('gemini_text', 4), ('gemini_image', 5), ('chatgpt', 'a0'), ('copilot', 'a1')):
                            v = (p // 2 == 2) if c_ is None else ((p // 2 == int(c_[1])) if isinstance(c_, str) else (p == c_))
                            shares[nm_].append(float(np.mean(v)))
                        cnt_g.append(int(np.sum(p // 2 == 2)))
                        vv = vote(p // 2, gs[msk]); sv_g.append(float(np.mean([v == 2 for v in vv.values()])))
                    ek[o] = dict({k: rng3(v) for k, v in shares.items()}, count_gemini_per_seed=cnt_g, sessions_voted_gemini=rng3(sv_g))
                e[f'K{K}'] = ek
        sc[sn] = e
    out['a']['scores'] = sc
    # reference: ChatGPT / Copilot test connections assigned to Gemini
    yT = y6[T]; aT = yT // 2; ref = {}
    for K in KS:
        ref[f'K{K}'] = {o: {APPS[a_]: rng3([np.mean(P[(o, K, s)]['test'][aT == a_] // 2 == 2) for s in SEEDS]) for a_ in (0, 1)} for o in OBS}
    out['a']['reference_chatgpt_copilot_test_connections_assigned_gemini'] = ref
    for sn in ('all', 'exact', 'misattributed_chatgpt_copilot_name', 'genuine_google_destination', 'genuine_exact', 'genuine_test_sessions'):
        e = sc[sn]; print(f'\n[{sn}] n={e["n"]} sessions={e["n_sessions"]}')
        if e['n'] == 0: continue
        for K in KS:
            print(f'  K={K} ' + '  '.join(f'{o}: Gem {e[f"K{K}"][o]["gemini"]["mean"]:.1f} [{e[f"K{K}"][o]["gemini"]["min"]:.0f},{e[f"K{K}"][o]["gemini"]["max"]:.0f}] '
                                        f'(txt {e[f"K{K}"][o]["gemini_text"]["mean"]:.0f}/img {e[f"K{K}"][o]["gemini_image"]["mean"]:.0f}) sessGem {e[f"K{K}"][o]["sessions_voted_gemini"]["mean"]:.0f}' for o in OBS))

    # ------------------------------------------------------------ (a) open world: Google background of non-Gemini test sessions
    bgz = np.load(os.path.join(ROOT, 'data', 'derived', 'genai_background.npz'), allow_pickle=True)
    bg_sid, bg_pkg = bgz['session_id'], bgz['pkg']
    rows = {}
    for l in open(os.path.join(ROOT, 'results', 'open_world_reject.jsonl'), encoding='utf-8'):
        r = json.loads(l)
        if r.get('tag') == 'open_world' and r['dataset'] == 'genai': rows[(r['K'], r['seed'], r['attacker'], r['bg_class'])] = r
    ow = {}; ow_runs = []
    for (K, seed, att, bgc), r in sorted(rows.items()):
        if att not in ('ours', 'lgbm'): continue
        mon = r['monitored']; gem = [i for i, c in enumerate(mon) if c in (4, 5)]
        ow_runs.append(dict(K=K, seed=seed, attacker=att, bg_class=bgc, monitored=mon, gemini_monitored=bool(gem)))
        if not gem: continue
        sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', f'fewshot_k{K}_s{seed}.json')))
        te = np.array([D.file2sid[f] for f in sp['test']]); bgi = np.where(np.isin(bg_sid, te))[0]
        z = np.load(os.path.join(ROOT, 'results', 'runs', 'ow_scores', f'open_world_genai_K{K}_bg{bgc}_{att}_s{seed}.npz'))
        nb = int((z['kind_neg'] == 0).sum()); assert nb == len(bgi) and np.array_equal(z['sid_neg'][:nb], bg_sid[bgi])
        thr = float(r['selected_t']); assert abs(thr - float(np.quantile(z['cal_ens'], 0.95))) < 1e-9
        pk, sa = bg_pkg[bgi], np.array([sid_app[int(x)] for x in bg_sid[bgi]])
        pred, rep = z['pred_neg'][:nb], z['neg_ens'][:nb] >= thr; asg = np.isin(pred, gem)
        goog = np.array([is_google_pkg(p) for p in pk]); gms = pk == 'com.google.android.gms'; ng_s = sa != 2
        groups = {'play_services_nonGemini_sessions': gms & ng_s, 'other_google_nonGemini_sessions': goog & ~gms & ng_s,
                  'google_nonGemini_sessions': goog & ng_s, 'nongoogle_nonGemini_sessions': ~goog & ng_s,
                  'google_Gemini_sessions': goog & ~ng_s, 'all_background': np.ones(nb, bool)}
        gm = np.isin(z['y'], gem)
        for gname_, m in groups.items():
            ow.setdefault((att, bgc, K, gname_), []).append(dict(seed=seed, n=int(m.sum()), assigned=float(asg[m].mean()) if m.any() else None,
                                                                  assigned_reported=float((asg & rep)[m].mean()) if m.any() else None,
                                                                  reported_any_class=float(rep[m].mean()) if m.any() else None,
                                                                  n_assigned=int(asg[m].sum()), n_assigned_reported=int((asg & rep)[m].sum())))
        ow.setdefault((att, bgc, K, 'gemini_monitored_recall_at_threshold'), []).append(dict(seed=seed, n=int(gm.sum()), recall=float(((z['pred'] == z['y']) & (z['test_ens'] >= thr) & gm).sum() / max(gm.sum(), 1)),
                                                                                      assigned_gemini_reported=float((np.isin(z['pred'], gem) & (z['test_ens'] >= thr) & gm).sum() / max(gm.sum(), 1))))
    ow_out = {}
    for (att, bgc, K, gname_), v in sorted(ow.items()):
        key = f'{att}_bg{bgc}_K{K}'
        if gname_ == 'gemini_monitored_recall_at_threshold':
            ow_out.setdefault(key, {})[gname_] = dict(per_seed=v, recall=rng3([x['recall'] for x in v]), assigned_gemini_and_reported=rng3([x['assigned_gemini_reported'] for x in v]))
        else:
            ok = [x for x in v if x['n'] > 0]
            ow_out.setdefault(key, {})[gname_] = dict(per_seed=v, seeds=[x['seed'] for x in ok], n_per_seed=[x['n'] for x in ok],
                                                      assigned_gemini=rng3([x['assigned'] for x in ok]) if ok else None,
                                                      assigned_gemini_and_reported=rng3([x['assigned_reported'] for x in ok]) if ok else None,
                                                      reported_any_class=rng3([x['reported_any_class'] for x in ok]) if ok else None)
    out['a']['open_world_background'] = dict(runs=ow_runs, results=ow_out)
    print('\nopen world: Google background of ChatGPT/Copilot test sessions assigned to Gemini (argmax | argmax & reported at 5%-FPR threshold)')
    for key, e in ow_out.items():
        for gname_ in ('play_services_nonGemini_sessions', 'other_google_nonGemini_sessions', 'nongoogle_nonGemini_sessions', 'google_Gemini_sessions'):
            x = e.get(gname_)
            if x and x['assigned_gemini']:
                print(f'  {key:14s} {gname_:34s} n={x["n_per_seed"]} seeds={x["seeds"]} {x["assigned_gemini"]["mean"]:5.1f} [{x["assigned_gemini"]["min"]:.0f},{x["assigned_gemini"]["max"]:.0f}] | '
                      f'{x["assigned_gemini_and_reported"]["mean"]:5.1f} [{x["assigned_gemini_and_reported"]["min"]:.0f},{x["assigned_gemini_and_reported"]["max"]:.0f}]  reported(any) {x["reported_any_class"]["mean"]:.1f}')
        print(f'  {key:14s} Gemini monitored recall at threshold {e["gemini_monitored_recall_at_threshold"]["recall"]}')

    # ------------------------------------------------------------ (a) open-world LightGBM on the Google-app connections (exact replay)
    bgl = load_bg('genai'); bgt = (bgl[2], bgl[3], bgl[4]); owg = {}; rep_chk = {}
    for (K, seed, att, bgc), r in sorted(rows.items()):
        if att != 'lgbm' or not any(c in (4, 5) for c in r['monitored']): continue
        rp = ow_lgbm_replay(D, Fall, bgt, K, seed, bgc, Fg)
        z = np.load(os.path.join(ROOT, 'results', 'runs', 'ow_scores', f'open_world_genai_K{K}_bg{bgc}_lgbm_s{seed}.npz'))
        ok = (list(rp['mon']) == r['monitored'] and np.array_equal(rp['pred'], z['pred']) and np.array_equal(rp['pred_neg'], z['pred_neg'])
              and np.allclose(rp['test_ens'], z['test_ens'], atol=1e-9) and np.allclose(rp['neg_ens'], z['neg_ens'], atol=1e-9) and np.allclose(rp['cal_ens'], z['cal_ens'], atol=1e-9))
        rep_chk[f'K{K}_bg{bgc}_s{seed}'] = dict(reproduced=bool(ok), pred_agree=float(np.mean(rp['pred'] == z['pred'])), neg_pred_agree=float(np.mean(rp['pred_neg'] == z['pred_neg'])),
                                                max_abs_ens_diff=float(max(np.abs(rp['test_ens'] - z['test_ens']).max(), np.abs(rp['neg_ens'] - z['neg_ens']).max(), np.abs(rp['cal_ens'] - z['cal_ens']).max())))
        if not ok: continue
        thr = float(r['selected_t']); gem = [i for i, c in enumerate(r['monitored']) if c in (4, 5)]
        asg = np.isin(rp['pred_g'], gem); rpt = rp['ens_g'] >= thr
        for sn in ('all', 'exact', 'genuine_google_destination', 'genuine_exact', 'genuine_test_sessions', 'misattributed_chatgpt_copilot_name'):
            m = subsets[sn]
            if not m.any(): continue
            owg.setdefault(f'bg{bgc}_K{K}', {}).setdefault(sn, []).append(dict(seed=seed, n=int(m.sum()), assigned=float(asg[m].mean()), assigned_reported=float((asg & rpt)[m].mean()),
                                                                              n_assigned=int(asg[m].sum()), n_assigned_reported=int((asg & rpt)[m].sum())))
    owg_out = {k: {sn: dict(per_seed=v, seeds=[x['seed'] for x in v], assigned_gemini=rng3([x['assigned'] for x in v]),
                            assigned_gemini_and_reported=rng3([x['assigned_reported'] for x in v])) for sn, v in d.items()} for k, d in owg.items()}
    out['a']['open_world_lgbm_google_app_connections'] = dict(replay_check=rep_chk, results=owg_out)
    print('\nopen-world LightGBM replay check:', json.dumps(rep_chk))
    for k, d in owg_out.items():
        for sn, e in d.items():
            print(f'  {k} {sn:36s} seeds {e["seeds"]} assigned {e["assigned_gemini"]["mean"]:.1f} [{e["assigned_gemini"]["min"]:.0f},{e["assigned_gemini"]["max"]:.0f}]  '
                  f'assigned&reported {e["assigned_gemini_and_reported"]["mean"]:.1f} [{e["assigned_gemini_and_reported"]["min"]:.0f},{e["assigned_gemini_and_reported"]["max"]:.0f}]')

    # ------------------------------------------------------------ (b) per-assistant F1 and session accuracy
    sidT = D.session_id[T]; us, inv_ = np.unique(sidT, return_inverse=True); nS = len(us)
    s_lab = np.array([np.bincount(yT[inv_ == j]).argmax() for j in range(nS)]); s_app = s_lab // 2
    for K in KS:
        bk = {}
        for o in OBS:
            f1a, sa_, sj_, mac = [], [], [], []
            for s in SEEDS:
                p = P[(o, K, s)]['test']; pa = p // 2
                f1a.append(f1_score(aT, pa, labels=[0, 1, 2], average=None, zero_division=0)); mac.append(f1_score(aT, pa, average='macro'))
                va = np.array([np.bincount(pa[inv_ == j], minlength=3).argmax() for j in range(nS)])
                vj = np.array([np.bincount(p[inv_ == j], minlength=6).argmax() for j in range(nS)])
                sa_.append([np.mean(va[s_app == a_] == a_) for a_ in range(3)]); sj_.append([np.mean(vj[s_app == a_] == s_lab[s_app == a_]) for a_ in range(3)])
            f1a, sa_, sj_ = np.array(f1a), np.array(sa_), np.array(sj_)
            bk[o] = dict(assistant_f1={APPS[a_]: rng3(f1a[:, a_]) for a_ in range(3)}, assistant_macro_f1=rng3(mac),
                         session_acc_assistant={APPS[a_]: rng3(sa_[:, a_]) for a_ in range(3)},
                         session_acc_app_x_modality={APPS[a_]: rng3(sj_[:, a_]) for a_ in range(3)},
                         n_test_sessions_per_assistant=[int((s_app == a_).sum()) for a_ in range(3)],
                         n_test_connections_per_assistant=[int((aT == a_).sum()) for a_ in range(3)])
        out['b'][f'K{K}'] = bk
        print(f'\n(b) K={K} ' + ' | '.join(f'{o}: F1 ' + '/'.join(f'{bk[o]["assistant_f1"][A]["mean"]:.1f}' for A in APPS) + ' sess ' +
                                          '/'.join(f'{bk[o]["session_acc_assistant"][A]["mean"]:.0f}' for A in APPS) for o in OBS))

    # ------------------------------------------------------------ (c) Gemini-specific hosts and the vote over them
    ac_sessions = [s for s in sid_app if sid_app[s] != 2]; gem_sessions = [s for s in sid_app if sid_app[s] == 2]
    cand = set()
    for (s, pkg), names in contact.items():
        for h in names:
            if GEM_RE.search(h): cand.add(h)
    HT = H[T]; FHT = np.array([fold(h) if h else '<none>' for h in HT], dtype=object)
    gemT = aT == 2
    top_hosts = [h for h, n in collections.Counter(FHT[gemT]).most_common() if n / gemT.sum() >= 0.02 and h != '<none>']
    cand |= {GEM_IMG}; cand_all = sorted(cand | set(top_hosts))
    ctab = {}
    for h in cand_all:
        e = {}
        for grp, ss in (('ChatGPT_Copilot_generic', [s for s in ac_sessions if sid_part[s] == 'generic']), ('ChatGPT_Copilot_controlled', [s for s in ac_sessions if sid_part[s] == 'controlled']),
                        ('Gemini_text', [s for s in gem_sessions if sid_cls[s] == 'Gemini_text']), ('Gemini_multi', [s for s in gem_sessions if sid_cls[s] == 'Gemini_multi']),
                        ('Gemini_controlled', [s for s in gem_sessions if sid_part[s] == 'controlled'])):
            by_gapp = [s for s in ss if h in contact.get((s, GAPP), ())]
            by_any = [s for s in ss if any(h in v for (s2, _), v in contact.items() if s2 == s)]
            pk_any = sorted({p for (s2, p), v in contact.items() if s2 in set(ss) and h in v})
            e[grp] = dict(n_sessions=len(ss), google_app=len(by_gapp), any_package=len(by_any), packages=pk_any)
        e['gemini_regex'] = bool(GEM_RE.search(h)); e['n_gemini_test_connections'] = int(np.sum(FHT[gemT] == h))
        e['google_app_contacts_in_chatgpt_copilot_sessions'] = e['ChatGPT_Copilot_generic']['google_app'] + e['ChatGPT_Copilot_controlled']['google_app']
        ctab[h] = e
    specific = sorted(h for h in cand if h != GEM_IMG and GEM_RE.search(h) and h != PROACTIVE and ctab[h]['google_app_contacts_in_chatgpt_copilot_sessions'] == 0)
    img_ok = ctab[GEM_IMG]['google_app_contacts_in_chatgpt_copilot_sessions'] == 0
    keep_old = np.array([category(h, 2) in ('assistant', 'image') for h in HT])   # construct_validity 'assistant' + 'image' of the Gemini class
    keep_new = keep_old & np.array([(h in specific) or (h == GEM_IMG and img_ok) for h in FHT])
    named_google = np.array([bool(h) and bool(GOOGLE_RE.search(h)) for h in HT])
    cc = dict(candidate_hosts_contact_table=ctab, gemini_specific_after_relabel=specific, image_host_kept=bool(img_ok),
              n_gemini_test_connections=int(gemT.sum()),
              share_generic_google_old=float(np.mean(named_google[gemT] & ~keep_old[gemT])),
              share_generic_google_new=float(np.mean(named_google[gemT] & ~keep_new[gemT])),
              share_unnamed=float(np.mean(FHT[gemT] == '<none>')),
              n_chatgpt_copilot_sessions=len(ac_sessions),
              n_chatgpt_copilot_sessions_with_named_google_app_connection=len({s for (s, p) in contact if p == GAPP and sid_app[s] != 2}),
              n_chatgpt_copilot_sessions_with_google_app_payload_connection=int(len(np.unique(gs))))
    for rule, keep in (('original_rule', keep_old), ('proactivebackend_relabelled', keep_new)):
        m = gemT & keep; ss = np.unique(sidT[m])
        rr = dict(n_connections=int(m.sum()), n_sessions_with_connections=int(len(ss)), n_gemini_test_sessions=int((s_app == 2).sum()),
                  sessions_with_connections_by_modality=dict(collections.Counter('image' if yT[sidT == s_][0] % 2 else 'text' for s_ in ss)),
                  hosts=dict(collections.Counter(FHT[m].tolist())))
        for K in KS:
            rk = {}
            for o in OBS:
                acc, acc_all = [], []
                for s in SEEDS:
                    vv = vote(P[(o, K, s)]['test'][m] // 2, sidT[m]); acc.append(np.mean([v == 2 for v in vv.values()]) if vv else np.nan)
                    acc_all.append(sum(v == 2 for v in vv.values()) / int((s_app == 2).sum()))
                rk[o] = dict(acc_on_sessions_with_connections=rng3(acc), acc_all_16_sessions_missing_as_error=rng3(acc_all))
            rr[f'K{K}'] = rk
        cc[rule] = rr
        print(f'\n(c) {rule}: {rr["n_connections"]} connections in {rr["n_sessions_with_connections"]}/{rr["n_gemini_test_sessions"]} Gemini test sessions {rr["sessions_with_connections_by_modality"]} hosts {rr["hosts"]}')
        for K in KS:
            print(f'   K={K} ' + '  '.join(f'{o} {rr[f"K{K}"][o]["acc_on_sessions_with_connections"]["mean"]:.1f} [{rr[f"K{K}"][o]["acc_on_sessions_with_connections"]["min"]:.0f},{rr[f"K{K}"][o]["acc_on_sessions_with_connections"]["max"]:.0f}]' for o in OBS))
    out['c'] = cc
    print('\n(c) Gemini-regex hosts and Google-app contacts in ChatGPT/Copilot sessions:')
    for h, e in ctab.items():
        print(f'   {h:45s} regex={e["gemini_regex"]!s:5s} testconn={e["n_gemini_test_connections"]:4d} gapp in ChatGPT/Copilot: gen {e["ChatGPT_Copilot_generic"]["google_app"]}/{e["ChatGPT_Copilot_generic"]["n_sessions"]} '
              f'ctl {e["ChatGPT_Copilot_controlled"]["google_app"]}/{e["ChatGPT_Copilot_controlled"]["n_sessions"]} any pkg gen {e["ChatGPT_Copilot_generic"]["any_package"]} | Gemini text {e["Gemini_text"]["google_app"]}/{e["Gemini_text"]["n_sessions"]} multi {e["Gemini_multi"]["google_app"]}/{e["Gemini_multi"]["n_sessions"]}')
    print('Gemini-specific after re-labelling:', specific, 'image host kept:', img_ok)
    print(f'generic Google share of Gemini test connections: old {100*cc["share_generic_google_old"]:.1f}% new {100*cc["share_generic_google_new"]:.1f}%  unnamed {100*cc["share_unnamed"]:.1f}%')

    json.dump(out, open(os.path.join(ROOT, 'results', 'google_app_check.json'), 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print('\nwritten results/google_app_check.json')


if __name__ == '__main__':
    main()
