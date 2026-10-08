"""Temporal session-scarce splits: the attacker labels the EARLIEST K sessions of each class and attacks the LATEST
sessions, so app versions, back-ends and user habits have moved on in between.
  temporal_k{K}_s{seed}.json / ccma_temporal_k{K}_s{seed}.json
Sessions are ordered by capture timestamp within each class: earliest 60% are the attacker's pool (K sampled from the
earliest half of that pool), the next 20% are validation, the latest 20% are test."""
import json, os, hashlib, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); OUT = os.path.join(ROOT, 'configs', 'splits')


def digest(d): return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


for prefix, cls_col in (('genai', 'class_dir'), ('ccma', 'app')):
    ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', f'{prefix}_sessions.csv'))
    if 'n_kept' in ses: ses = ses[ses.n_kept > 0]
    if prefix == 'genai': ses = ses[ses.part == 'generic']
    pool, val, test = [], [], []
    for c, g in ses.groupby(cls_col):
        g = g.sort_values('session_ts'); n = len(g); a, b = int(n * 0.6), int(n * 0.8)
        pool.append(g.iloc[:a]); val.append(g.iloc[a:b]); test.append(g.iloc[b:])
    pool = pd.concat(pool); val = pd.concat(val); test = pd.concat(test)
    for K in (2, 4, 8):
        for seed in range(5):
            rng = np.random.RandomState(4000 + seed); tr = []
            for c, g in pool.groupby(cls_col):
                g = g.sort_values('session_ts'); half = g.iloc[:max(K, len(g) // 2)]   # earliest half of the attacker's pool
                take = half.sample(n=min(K, len(half)), random_state=rng) if len(half) > K else half
                tr += list(take.file)
            sp = {'train': tr, 'val': list(val.file), 'test': list(test.file)}
            name = f'{"" if prefix == "genai" else "ccma_"}temporal_k{K}_s{seed}.json'
            meta = dict(dataset=prefix, unit='session', rule=f'{K} of the earliest sessions per class; test = latest 20% by capture time', seed=seed, hash=digest(sp), counts={k: len(v) for k, v in sp.items()})
            json.dump({'meta': meta, **sp}, open(os.path.join(OUT, name), 'w'), indent=1)
        print(prefix, K, 'train', len(tr), 'val', len(val), 'test', len(test))
