"""Time between labelling and attack in the temporal splits: for every class, split and seed, the gap from the latest
labelled session to each test session (days), from the session timestamps. Also, for the temporal-drift table, the difference of
the pre-trained encoder (fs{K}_sslXL_drift) over every other attacker of the table, per K-session draw and with the
95% interval of the two-level paired bootstrap of scripts/boot2.py (draws resampled with replacement, then the test
sessions; 4000 resamples; 'sig' = the interval excludes zero), from the saved test predictions. Writes results/drift_gaps.json.

The training-free nearest neighbour on the first ten packets (knn10, runs fs{K}_drift_knn10 / ccma_fs{K}_drift_knn10,
tag knn_robust, written by scripts/robust_assistant_knn.py) is one of the rows of the temporal-drift table ('knn10') and is
compared by the same procedure as every other row: one boot2.Cell per dataset and K, whose resamples are drawn at
construction from np.random.RandomState(20260926) before any observer is added, so every row is scored on the same
resamples and no row's interval depends on which other rows are present; encoder minus the row's observer; per-draw
differences, two-level 95% interval and 'sig' as above; the session-only interval alongside."""
import json, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
import boot2
DRIFT = {'lgbm': 'fs{K}_drift_lgbm_meta64', 'df': 'fs{K}_drift_dfmeta64', 'netclr': 'fs{K}_netclr_drift', 'cf': 'fs{K}_cfpub_drift',
         'knn10': 'fs{K}_drift_knn10',  # the 1-NN row (scripts/robust_assistant_knn.py)
         'scratch': 'fs{K}_sslscratchXL_drift', 'ours': 'fs{K}_sslXL_drift'}  # rows of the temporal-drift table (paper/make_tables_def_drift.py)


def drift_boot2(ds, pre):
    D = GenAIData(ROOT, prefix=ds); ylab = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app
    out = {}
    for K in (2, 4, 8):
        P, te0 = {a: [] for a in DRIFT}, None
        for s in range(5):
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}temporal_k{K}_s{s}.json')); te = idx['test']
            if te0 is None: te0 = te
            assert np.array_equal(te0, te), 'test sessions differ across draws'
            for a, fmt in DRIFT.items():
                z = np.load(os.path.join(ROOT, 'results', 'runs', f'{pre}{fmt.format(K=K)}_s{s}', 'test_preds.npz'))
                assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], ylab[te]); P[a].append(z['pred'])
        cell = boot2.Cell(ylab[te0], D.session_id[te0], NC)
        for a in DRIFT: cell.add(a, P[a])
        F1S[ds][f'K{K}'] = {a: cell.f1(a) for a in DRIFT}   # full-test macro-F1 per draw (mean, sd, min, max), a cross-check of the table means
        out[f'K{K}'] = {}
        for a in DRIFT:
            if a == 'ours': continue
            c = cell.compare('ours', a)
            out[f'K{K}'][a] = dict(diff=c['diff'], per_draw=c['per_draw'], ci=c['ci'], sig=c['sig'], ci_session_only=c['session_only']['ci'], sig_session_only=c['session_only']['sig'])
            print(f'{ds} drift K={K} ours-{a}: {c["diff"]:+.2f} per draw [{", ".join(f"{v:+.1f}" for v in c["per_draw"])}] two-level [{c["ci"][0]:+.2f},{c["ci"][1]:+.2f}]{" *" if c["sig"] else ""}')
    return out

OUT, F1S = {}, {'genai': {}, 'ccma': {}}
for ds, pre in (('genai', ''), ('ccma', 'ccma_')):
    S = pd.read_csv(os.path.join(ROOT, 'data', 'derived', f'{ds}_sessions.csv')); ts = dict(zip(S.file, S.session_ts)); cls = dict(zip(S.file, S.class_dir if 'class_dir' in S else S.app))
    gaps_min, gaps_med, span = [], [], []
    for K in (2, 4, 8):
        for s in range(5):
            sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', f'{pre}temporal_k{K}_s{s}.json')))
            for c in sorted(set(cls[f] for f in sp['test'])):
                last_lab = max(ts[f] for f in sp['train'] if cls[f] == c); tt = np.array([ts[f] for f in sp['test'] if cls[f] == c])
                g = (tt - last_lab) / 86400.0; gaps_min.append(g.min()); gaps_med.append(np.median(g))
    allts = np.array(list(ts.values()))
    OUT[ds] = dict(first_gap_days=(float(np.min(gaps_min)), float(np.max(gaps_min))), median_gap_days=(float(np.min(gaps_med)), float(np.median(gaps_med)), float(np.max(gaps_med))),
                   campaign_days=float((allts.max() - allts.min()) / 86400.0))
    print(ds, OUT[ds])
    OUT[ds]['boot2_ours_minus'] = drift_boot2(ds, pre)
    OUT[ds]['observer_f1'] = F1S[ds]
json.dump(OUT, open(os.path.join(ROOT, 'results', 'drift_gaps.json'), 'w'), indent=1)
