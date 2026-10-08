"""Does the server-address lookup stay as good as the metadata observers on an unseen phone and as the labels age?
Training-free for the lookups; the metadata observers are read from their saved test predictions.

Rules.

Observers.
  Lookups exactly as scripts/ip_baseline.py (same functions, imported): 'backoff' (exact server address if seen in the
  labelled flows, else its /24, else its /16; the paper's address lookup), 'p24' and 'ip' (exact address only). They are
  fitted on the flows of the labelled training sessions of the split only, over the paper's task classes (GenAI 6
  app x modality classes, CCMA 9 apps); nothing is selected.
  Metadata observers from results/runs/<run>/test_preds.npz: the pre-trained encoder ('ours'), LightGBM ('lgbm') and
  the nearest neighbour on the first ten packets ('knn10'), with the run ids of scripts/robust_assistant.py and
  scripts/robust_assistant_knn.py. An observer whose five seeds are not all present is not reported (listed as missing).

(a) Unseen phone, K = 4, seeds 0-4: GenAI cd_8e_k4 (test phone Pixel) and cd_a6_k4 (Xiaomi); CCMA cd_2c_k4, cd_f4_k4,
    cd_f8_k4. Metrics as scripts/robust_assistant.py: GenAI per-connection assistant macro-F1 (pred//2 vs y//2) and the
    session vote of the assistant (majority of pred//2 over the session's test connections, ties to the lowest index);
    CCMA per-connection app macro-F1 and the session vote of the app. Mean and population s.d. over the seeds. For the
    lookups also coverage: the share of test connections whose key (exact address / /24 / any back-off level) occurs in
    the labelled flows.

(b) Label age, temporal splits temporal_k{K} (K = 2, 4, 8), seeds 0-4, both campaigns. For every test session its age =
    days from the latest labelled session of its own class to the session (the definition of scripts/drift_gaps.py,
    session timestamps of data/derived/<ds>_sessions.csv). Fixed bins: <=30, 31-90, 91-365 and >365 days. Per bin,
    pooled over K and seeds (unit = a session under one draw): the share of sessions whose assistant (GenAI) or app
    (CCMA) wins the vote, for every observer, and the exact-address coverage of the session's test connections (share
    whose exact server address occurs in the labelled flows), averaged over sessions. The number of units and of
    distinct sessions per bin is reported; bins with fewer than 20 units are reported but not interpreted.
    Cross-check: the pooled full-split numbers of the lookups must equal results/ip_baseline.json['drift'].

Writes results/addr_robust.json."""
import collections, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from ip_baseline import server_addr, keys_of, predict

SEEDS = range(5)
LOOK = ('backoff', 'p24', 'ip')
META = ('ours', 'lgbm', 'knn10')
CD = {'genai': [('Pixel', 'cd_8e_k4', {'ours': 'cd8e_k4_sslXL', 'lgbm': 'cd8e_k4_lgbm_meta64', 'knn10': 'cd8e_k4_knn10'}),
                ('Xiaomi', 'cd_a6_k4', {'ours': 'cda6_k4_sslXL', 'lgbm': 'cda6_k4_lgbm_meta64', 'knn10': 'cda6_k4_knn10'})],
      'ccma': [('2c', 'ccma_cd_2c_k4', {'ours': 'ccma_cd2c_k4_sslXL', 'lgbm': 'ccma_cd2c_k4_lgbm_meta64', 'knn10': 'ccma_cd2c_k4_knn10'}),
               ('f4', 'ccma_cd_f4_k4', {'ours': 'ccma_cdf4_k4_sslXL', 'lgbm': 'ccma_cdf4_k4_lgbm_meta64', 'knn10': 'ccma_cdf4_k4_knn10'}),
               ('f8', 'ccma_cd_f8_k4', {'ours': 'ccma_cdf8_k4_sslXL', 'lgbm': 'ccma_cdf8_k4_lgbm_meta64', 'knn10': 'ccma_cdf8_k4_knn10'})]}
DRIFT_RUN = {'ours': '{pre}fs{K}_sslXL_drift', 'lgbm': '{pre}fs{K}_drift_lgbm_meta64', 'knn10': '{pre}fs{K}_drift_knn10'}
BINS = [(0, 30, '<=30 d'), (30, 90, '31-90 d'), (90, 365, '91-365 d'), (365, 1e9, '>365 d')]
MISSING = []


def mf1(y, p): return 100 * f1_score(y, p, average='macro', zero_division=0)


def summ(v): return [round(float(np.mean(v)), 2), round(float(np.std(v)), 2)]


def load_npz(run, te, y):
    p = os.path.join(ROOT, 'results', 'runs', run, 'test_preds.npz')
    if not os.path.exists(p): MISSING.append(run); return None
    z = np.load(p); assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], y), run
    return z['pred'].astype(int)


def session_vote(pr, inv, nS, NC):
    return np.array([np.bincount(pr[inv == j], minlength=NC).argmax() for j in range(nS)])


def cross_device(D, ds, KK, ylab, NC, nact):
    out = {}
    for phone, split, runs in CD[ds]:
        st = collections.defaultdict(list)
        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json')); tr, te = idx['train'], idx['test']
            y = ylab[te]; us, inv = np.unique(D.session_id[te], return_inverse=True); nS = len(us)
            ysa = session_vote(y // nact, inv, nS, NC // nact)
            P = {}
            for o in LOOK:
                P[o], cov = predict(KK, tr, te, ylab, NC, o); st[f'{o}|coverage'].append(100 * cov.mean())
            for a in META:
                pr = load_npz(f'{runs[a]}_s{s}', te, y)
                if pr is not None: P[a] = pr
            for a, pr in P.items():
                st[f'{a}|conn_f1'].append(mf1(y // nact, pr // nact))
                st[f'{a}|sess_acc'].append(100 * float((session_vote(pr // nact, inv, nS, NC // nact) == ysa).mean()))
            st['n_test_sessions'].append(nS)
        res = {}
        for k, v in st.items():
            if k == 'n_test_sessions': continue
            a, m = k.split('|')
            if len(v) == len(SEEDS): res.setdefault(a, {})[m] = summ(v)
        out[phone] = dict(split=split, n_test_sessions=st['n_test_sessions'][0], observers=res)
        print(f'{ds} unseen phone {phone:6s} ({st["n_test_sessions"][0]} sessions): ' + '  '.join(
            f'{a} {r["conn_f1"][0]:5.1f}/{r["sess_acc"][0]:5.1f}' + (f' cov {r["coverage"][0]:4.1f}' if 'coverage' in r else '') for a, r in res.items()))
    return out


def label_age(D, ds, KK, ylab, NC, nact, pre):
    S = pd.read_csv(os.path.join(ROOT, 'data', 'derived', f'{ds}_sessions.csv'))
    ts = dict(zip(S.file, S.session_ts)); cls = dict(zip(S.file, S.class_dir if 'class_dir' in S else S.app))
    sid2file = dict(zip(D.sessions.session_id, D.sessions.file))
    units = []   # one row per (K, seed, test session)
    full = collections.defaultdict(list)
    for K in (2, 4, 8):
        for s in SEEDS:
            sp_path = os.path.join(ROOT, 'configs', 'splits', f'{pre}temporal_k{K}_s{s}.json'); sp = json.load(open(sp_path))
            idx, _ = D.split_indices(sp_path); tr, te = idx['train'], idx['test']
            y = ylab[te]; us, inv = np.unique(D.session_id[te], return_inverse=True); nS = len(us)
            ysa = session_vote(y // nact, inv, nS, NC // nact)
            last_lab = {}
            for f in sp['train']: last_lab[cls[f]] = max(last_lab.get(cls[f], -np.inf), ts[f])
            ip_seen = set(KK['ip'][tr]); exact_cov = np.array([k in ip_seen for k in KK['ip'][te]])
            P = {o: predict(KK, tr, te, ylab, NC, o)[0] for o in LOOK}
            for a in META:
                pr = load_npz(DRIFT_RUN[a].format(pre=pre, K=K) + f'_s{s}', te, y)
                if pr is not None: P[a] = pr
            votes = {a: session_vote(pr // nact, inv, nS, NC // nact) == ysa for a, pr in P.items()}
            for o in LOOK: full[(o, K)].append(100 * float(votes[o].mean()))
            for j in range(nS):
                f = sid2file[us[j]]; age = (ts[f] - last_lab[cls[f]]) / 86400.0
                units.append(dict(K=K, seed=s, session=int(us[j]), age=age, exact_cov=100 * float(exact_cov[inv == j].mean()),
                                  **{a: bool(v[j]) for a, v in votes.items()}))
    U = pd.DataFrame(units); obs = [a for a in LOOK + META if a in U]
    out = dict(age_days=dict(min=float(U.age.min()), median=float(U.age.median()), max=float(U.age.max())), bins={})
    for lo, hi, name in BINS:
        b = U[(U.age > lo) & (U.age <= hi)] if lo > 0 else U[U.age <= hi]
        if len(b) == 0: continue
        r = dict(n_units=int(len(b)), n_sessions=int(b.session.nunique()), exact_address_coverage=round(float(b.exact_cov.mean()), 1),
                 sess_acc={a: round(100 * float(b[a].mean()), 1) for a in obs if b[a].notna().all()})
        out['bins'][name] = r
        print(f'{ds} label age {name:9s} units {r["n_units"]:4d} sessions {r["n_sessions"]:3d}  exact-address coverage {r["exact_address_coverage"]:5.1f}  '
              + '  '.join(f'{a} {v:5.1f}' for a, v in r['sess_acc'].items()))
    # cross-check with results/ip_baseline.json (full-split session vote of the lookups)
    ipb = json.load(open(os.path.join(ROOT, 'results', 'ip_baseline.json')))['drift'][ds]
    key = 'sess_app_acc' if ds == 'genai' else 'sess_acc'
    chk = {}
    for (o, K), v in full.items():
        ref = ipb[f'K{K}']['observers'][o][key][0]; chk[f'{o}_K{K}'] = [round(float(np.mean(v)), 3), round(ref, 3)]
        assert abs(np.mean(v) - ref) < 1e-6, (ds, o, K, np.mean(v), ref)
    out['check_vs_ip_baseline'] = chk
    return out


def main():
    OUT = dict(rules=__doc__.split('Writes')[0].strip())
    for ds, pre in (('genai', ''), ('ccma', 'ccma_')):
        D = GenAIData(ROOT, prefix=ds); chk = collections.Counter()
        KK = keys_of([server_addr(k, chk) for k in D.index.biflow_key.values])
        ylab = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app; nact = 2 if ds == 'genai' else 1
        OUT[ds] = dict(unseen_phone=cross_device(D, ds, KK, ylab, NC, nact), label_age=label_age(D, ds, KK, ylab, NC, nact, pre))
    OUT['missing'] = MISSING
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'addr_robust.json'), 'w'), indent=1)
    print('missing:', MISSING or 'none'); print('written results/addr_robust.json')


if __name__ == '__main__':
    main()
