"""Two-level paired bootstrap for 'observer A minus observer B' under the K-session protocol, and (run as a script) the
significance verdict of every comparison listed in rule 7. Writes results/boot2.json.

Rules.
 1. Metric: test macro-F1 (%) of the task (6 app x modality classes on GenAI, 9 apps on CCMA) over the classes present
    in truth or prediction (identical to sklearn f1_score(average='macro')); for the server-address comparison
    also the GenAI assistant (3-class) macro-F1 derived from the 6-class prediction (pred // 2), as scripts/ip_baseline.py.
 2. Pairing: A and B are scored on the same five K-session draws (seeds 0-4; draw r of A and draw r of B use the same
    labelled sessions) and on the same test sessions; the test sessions are identical across draws and observers in
    every setting used here (asserted).
 3. Point estimate: the mean over the five draws of the per-draw difference A_r - B_r of full-test macro-F1 (the
    seed-paired difference printed in the tables).
 4. Resampling: B = 4000 resamples from np.random.RandomState(20260926), drawn once per setting cell (dataset, split
    family, K), in this order: the draw indices (B x 5), then the test-session multiplicities (B x n_sessions); the same
    resamples serve every comparison of the cell.
      level 1: five draw indices r*_1..r*_5, uniformly with replacement from {0,...,4};
      level 2: test-session multiplicities w ~ Multinomial(n_sessions, uniform): whole sessions, with all their connections;
      statistic: mean_j [F1(A_{r*_j}; w) - F1(B_{r*_j}; w)], F1(.; w) = macro-F1 of the session-weighted confusion
      matrix; the same w for A and B (paired) and for all five selected draws (the test sessions are shared).
 5. Interval: the 2.5th and 97.5th percentiles; a difference is significant iff the interval excludes zero.
 6. Reported alongside, never used for a mark: the per-draw differences, the number of draws with A > B, a paired t over
    the five per-draw differences (two-sided p, 95% CI), and the session-only bootstrap, which conditions on the five
    draws (test sessions only; np.random.RandomState(12345), 2000 resamples; seed-averaged macro-F1 difference), with
    its interval and verdict ('session_only').
 7. Comparisons (script mode):
    fewshot   splits [ccma_]fewshot_k{K}_s{s}, K = 1, 2, 4, 8: the pre-trained encoder (fs{K}_sslXL) minus every
              reference observer of the main comparison table (as scripts/threat_stats.py; the 1-NN is recomputed there and here);
    address   same cells: the encoder minus the address back-off lookup of scripts/ip_baseline.py (recomputed with its
              functions), on the task macro-F1 (both campaigns) and on the GenAI assistant macro-F1;
    xcorpus   same cells, K = 2, 4, 8: other-campaign-only pre-training (fs{K}_sslXLxc) minus no pre-training
              (fs{K}_sslscratchXL), and the encoder minus fs{K}_sslXLxc;
    ech       same cells, K = 4: ClientHello padding (fs4_sslXL_ech, fs4Lech_lgbm_meta64) and ClientHello plus server
              first-flight padding (fs4_sslXL_echfull, fs4Lechfull_lgbm_meta64) minus no defence, per observer, and
              full minus ClientHello-only padding;
    dilution  same cells, K = 1, 2, 4: each pre-training corpus of the pre-training-corpora table minus LightGBM (scripts/dilution_stats.py);
    drift     splits [ccma_]temporal_k{K}_s{s}, K = 2, 4, 8: the encoder (fs{K}_sslXL_drift) minus every other row of
              the temporal-drift table.
    Verdicts: 'A>B' / 'A<B' when the interval lies above / below zero, else 'ns'.
"""
import collections, json, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B_TWO, SEED_TWO = 4000, 20260926
B_OLD, SEED_OLD = 2000, 12345
R_DRAWS = 5


def macro_f1_cm(cm):  # cm [..., C, C] (true, pred); classes absent from truth and prediction are skipped
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float); denom = cm.sum(-1) + cm.sum(-2)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0); present = denom > 0
    return (f1 * present).sum(-1) / present.sum(-1)


def verdict(lo, hi):
    return 'A>B' if lo > 0 else ('A<B' if hi < 0 else 'ns')


class Cell:
    """One setting cell: fixed test connections y with their session ids, NC classes, R paired draws.
    add(name, preds) registers an observer (list of R prediction arrays aligned with y); compare(a, b) -> dict."""

    def __init__(self, y, sid, NC, R=R_DRAWS, B=B_TWO, seed=SEED_TWO):
        self.y = np.asarray(y); us, self.inv = np.unique(np.asarray(sid), return_inverse=True)
        self.nS, self.NC, self.R, self.B = len(us), int(NC), R, B
        rs = np.random.RandomState(seed)
        self.Ss = rs.randint(0, R, size=(B, R))                                          # level 1: draws
        self.Ws = rs.multinomial(self.nS, np.ones(self.nS) / self.nS, size=B)             # level 2: test sessions
        self.Wold = np.random.RandomState(SEED_OLD).multinomial(self.nS, np.ones(self.nS) / self.nS, size=B_OLD)
        self.F = {}

    def add(self, name, preds):
        assert len(preds) == self.R, (name, len(preds))
        full, bt, old = [], [], []
        for p in preds:
            p = np.asarray(p); assert p.shape == self.y.shape, name
            cm = np.zeros((self.nS, self.NC, self.NC), np.int64); np.add.at(cm, (self.inv, self.y, p), 1)
            full.append(100 * macro_f1_cm(cm.sum(0)))
            bt.append(100 * macro_f1_cm(np.tensordot(self.Ws, cm, axes=(1, 0))))
            old.append(100 * macro_f1_cm(np.tensordot(self.Wold, cm, axes=(1, 0))))
        self.F[name] = dict(full=np.array(full), boot=np.stack(bt), old=np.stack(old))

    def has(self, *names):
        return all(n in self.F for n in names)

    def f1(self, name):
        v = self.F[name]['full']; return dict(mean=float(v.mean()), sd=float(v.std()), min=float(v.min()), max=float(v.max()), per_draw=[float(x) for x in v])

    def compare(self, a, b):
        A, Bb = self.F[a], self.F[b]
        per = A['full'] - Bb['full']
        d = A['boot'] - Bb['boot']                                   # [R, B]
        D = d[self.Ss, np.arange(self.B)[:, None]].mean(1)           # D[b] = mean_j d[Ss[b, j], b]
        lo, hi = np.percentile(D, [2.5, 97.5])
        d_old = (A['old'] - Bb['old']).mean(0); lo1, hi1 = np.percentile(d_old, [2.5, 97.5])
        se = per.std(ddof=1) / np.sqrt(self.R)
        if se > 0:
            tc = stats.t.ppf(0.975, self.R - 1); tp = float(2 * stats.t.sf(abs(per.mean() / se), self.R - 1)); tci = [float(per.mean() - tc * se), float(per.mean() + tc * se)]
        else:
            tp, tci = (1.0 if per.mean() == 0 else 0.0), [float(per.mean())] * 2
        return dict(diff=float(per.mean()), per_draw=[float(x) for x in per], n_pos=int((per > 0).sum()), n_neg=int((per < 0).sum()),
                    ci=[float(lo), float(hi)], sig=bool(lo > 0 or hi < 0), verdict=verdict(lo, hi),
                    session_only=dict(ci=[float(lo1), float(hi1)], sig=bool(lo1 > 0 or hi1 < 0), verdict=verdict(lo1, hi1)),
                    t=dict(p=tp, ci=tci), n_test_sessions=int(self.nS), n_resamples=int(self.B))


# ----------------------------------------------------------------------------------------------- script mode
ATT = {'lgbm': 'fs{K}_lgbm_meta64', 'df': 'fs{K}_dfmeta64', 'tiktok': 'fs{K}_tiktok', 'tf': 'fs{K}_tf', 'cf': 'fs{K}_cfpub',
       'netclr': 'fs{K}_netclr', 'netclrtr': 'fs{K}_netclrtr', 'patch': 'meta_fewshot{K}_base', 'knn10': 'fs{K}_knn10',
       'cnn1d': 'fs{K}_paper1dcnn_512B', 'yatc': 'fs{K}_yatc_pre', 'etbert': 'fs{K}_etbert', 'scratch': 'fs{K}_sslscratchXL', 'ours': 'fs{K}_sslXL'}
EXTRA = {'xc': 'fs{K}_sslXLxc', 'ours_ech': 'fs{K}_sslXL_ech', 'ours_echfull': 'fs{K}_sslXL_echfull',
         'lgbm_ech': 'fs{K}Lech_lgbm_meta64', 'lgbm_echfull': 'fs{K}Lechfull_lgbm_meta64',
         'dilA': 'fs{K}_sslXLdilA', 'dilB': 'fs{K}_sslXLdilB', 'dilC': 'fs{K}_sslXLdilC'}
DRIFT = {'lgbm': 'fs{K}_drift_lgbm_meta64', 'df': 'fs{K}_drift_dfmeta64', 'netclr': 'fs{K}_netclr_drift', 'cf': 'fs{K}_cfpub_drift',
         'scratch': 'fs{K}_sslscratchXL_drift', 'ours': 'fs{K}_sslXL_drift'}


def rid(ds, fmt, K, s):
    f = fmt.format(K=K)
    if ds == 'ccma': f = f'ccma_fewshot{K}_base' if f.startswith('meta_fewshot') else 'ccma_' + f
    return f + f'_s{s}'


def load_pred(run, idx, y):
    p = os.path.join(ROOT, 'results', 'runs', run, 'test_preds.npz')
    if not os.path.exists(p): return None
    z = np.load(p); assert np.array_equal(z['idx'], idx) and np.array_equal(z['y'], y), run
    return z['pred']


def main():
    sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from data_mfr import GenAIData
    from trivial_baselines import feats_knn
    import ip_baseline as ipb
    COMP, F1S = [], {}

    def rec(family, ds, K, metric, cell, a, b, runs):
        if not cell.has(a, b): return
        c = cell.compare(a, b)
        key = f'{family}|{ds}|K{K}|{metric}|{a}-{b}'
        COMP.append(dict(key=key, family=family, dataset=ds, K=K, metric=metric, A=a, B=b, runs=runs, **c))
        for n in (a, b): F1S[f'{family}|{ds}|K{K}|{metric}|{n}'] = cell.f1(n)

    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); ylab = D.y_joint if ds == 'genai' else D.y_app
        NC = D.n_joint if ds == 'genai' else D.n_app; pre = '' if ds == 'genai' else 'ccma_'
        Fk = feats_knn(D.meta, D.meta_len)
        chk = collections.Counter(); KK = ipb.keys_of([ipb.server_addr(k, chk) for k in D.index.biflow_key.values])
        # ---------------- few-shot cells (the main comparison table / threat, address, xcorpus, ECH, dilution)
        for K in (1, 2, 4, 8):
            P, te0 = collections.defaultdict(list), None
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json')); tr, te = idx['train'], idx['test']
                if te0 is None: te0 = te
                assert np.array_equal(te0, te), 'test sessions differ across draws'
                p1 = np.empty(len(te), int)
                for b in range(0, len(te), 512): p1[b:b + 512] = ylab[tr][np.abs(Fk[te][b:b + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)]
                P['knn10'].append(p1)
                P['backoff'].append(ipb.predict(KK, tr, te, ylab, NC, 'backoff')[0])
                for a, fmt in list(ATT.items()) + list(EXTRA.items()):
                    if a == 'knn10': continue
                    z = load_pred(rid(ds, fmt, K, s), te, ylab[te])
                    if z is not None: P[a].append(z)
            y = ylab[te0]; sid = D.session_id[te0]
            cell = Cell(y, sid, NC)
            for a, v in P.items():
                if len(v) == 5: cell.add(a, v)
            runs = lambda a: (rid(ds, ATT.get(a) or EXTRA.get(a), K, '*') if a not in ('backoff',) else 'ip_baseline backoff lookup (recomputed)')
            for a in ATT:
                if a != 'ours': rec('fewshot', ds, K, 'task', cell, 'ours', a, [runs('ours'), runs(a)])
            rec('address', ds, K, 'task', cell, 'ours', 'backoff', [runs('ours'), runs('backoff')])
            if ds == 'genai':
                cella = Cell(y // 2, sid, NC // 2)
                for a in ('ours', 'backoff', 'lgbm', 'knn10'): cella.add(a, [p // 2 for p in P[a]])
                rec('address', ds, K, 'assistant', cella, 'ours', 'backoff', [runs('ours'), runs('backoff')])
            if K in (2, 4, 8):
                rec('xcorpus', ds, K, 'task', cell, 'xc', 'scratch', [runs('xc'), runs('scratch')])
                rec('xcorpus', ds, K, 'task', cell, 'ours', 'xc', [runs('ours'), runs('xc')])
            if K == 4:
                for a, b in (('ours_ech', 'ours'), ('ours_echfull', 'ours'), ('ours_echfull', 'ours_ech'),
                             ('lgbm_ech', 'lgbm'), ('lgbm_echfull', 'lgbm'), ('lgbm_echfull', 'lgbm_ech')):
                    rec('ech', ds, K, 'task', cell, a, b, [runs(a), runs(b)])
            if K in (1, 2, 4):
                for a in ('ours', 'dilA', 'dilB', 'dilC'):
                    rec('dilution', ds, K, 'task', cell, a, 'lgbm', [runs(a), runs('lgbm')])
            print(f'{ds} fewshot K={K}: test sessions {cell.nS}, observers {sorted(cell.F)}', flush=True)
        # ---------------- temporal drift cells (the temporal-drift table)
        for K in (2, 4, 8):
            P, te0 = collections.defaultdict(list), None
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}temporal_k{K}_s{s}.json')); te = idx['test']
                if te0 is None: te0 = te
                assert np.array_equal(te0, te)
                for a, fmt in DRIFT.items():
                    z = load_pred(f'{pre}{fmt.format(K=K)}_s{s}', te, ylab[te])
                    if z is not None: P[a].append(z)
            cell = Cell(ylab[te0], D.session_id[te0], NC)
            for a, v in P.items():
                if len(v) == 5: cell.add(a, v)
            for a in DRIFT:
                if a != 'ours': rec('drift', ds, K, 'task', cell, 'ours', a, [f'{pre}{DRIFT["ours"].format(K=K)}_s*', f'{pre}{DRIFT[a].format(K=K)}_s*'])
            print(f'{ds} drift K={K}: test sessions {cell.nS}, observers {sorted(cell.F)}', flush=True)

    # comparisons on which the session-only and the two-level interval disagree
    flips = [dict(key=c['key'], diff=round(c['diff'], 2), old=c['session_only']['verdict'], new=c['verdict'], ci_old=[round(x, 2) for x in c['session_only']['ci']],
                  ci_new=[round(x, 2) for x in c['ci']], per_draw=[round(x, 2) for x in c['per_draw']])
             for c in COMP if c['session_only']['sig'] != c['sig']]
    OUT = dict(rules=__doc__, settings=dict(B=B_TWO, seed=SEED_TWO, old_B=B_OLD, old_seed=SEED_OLD, draws=R_DRAWS),
               comparisons=COMP, observer_f1=F1S, flips_session_only_to_two_level=flips)
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'boot2.json'), 'w'), indent=1)
    print(f'\n{len(COMP)} comparisons; {len(flips)} verdicts differ (session-only vs two-level)')
    for c in COMP:
        print(f"{c['key']:52s} {c['diff']:+6.2f} draws [{', '.join(f'{x:+.1f}' for x in c['per_draw'])}] two-level [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] {c['verdict']:3s} "
              f"| session-only [{c['session_only']['ci'][0]:+.2f},{c['session_only']['ci'][1]:+.2f}] {c['session_only']['verdict']:3s} | t p={c['t']['p']:.3f}"
              + ('   <-- FLIP' if c['session_only']['sig'] != c['sig'] else ''))
    print('written results/boot2.json')


if __name__ == '__main__':
    main()
