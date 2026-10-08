"""Temporal and K-session splits, all at capture-session granularity.

 temporal_s2026.json : sessions ordered by capture timestamp within each class_dir; earliest 60% -> train, next 20% -> val,
                       latest 20% -> test (distribution/temporal drift; app versions and backend behaviour change over
                       the Oct-2024 .. Jun-2025 collection period).
 fewshot_k{K}_s{seed}.json : from the standard session split, keep only K training sessions per class_dir (val/test
                       unchanged) -> low-label regime; K in {2, 4, 8}.
 (The leave-one-device-out splits are written by make_splits.py.)
"""
import json, os, hashlib
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'configs', 'splits'); os.makedirs(OUT, exist_ok=True)


def digest(d):
    return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


def dump(name, split, meta):
    meta = dict(meta, hash=digest(split), counts={k: len(v) for k, v in split.items()})
    json.dump({'meta': meta, **split}, open(os.path.join(OUT, name), 'w'), indent=1); print(name, meta)


ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'genai_sessions.csv'))
gen = ses[ses.part == 'generic'].copy()
# temporal split
sp = {'train': [], 'val': [], 'test': []}
for cd, g in gen.groupby('class_dir'):
    g = g.sort_values('session_ts'); f = list(g.file); n = len(f)
    n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
    sp['train'] += f[:n_tr]; sp['val'] += f[n_tr:n_tr + n_va]; sp['test'] += f[n_tr + n_va:]
ts = pd.to_datetime(gen.session_ts, unit='s')
dump('temporal_s2026.json', sp, dict(unit='session', rule='per class_dir sorted by timestamp: earliest 60% train / next 20% val / latest 20% test',
                                     span=f'{ts.min().date()}..{ts.max().date()}'))
# few-shot
base = json.load(open(os.path.join(OUT, 'session_split_s2026.json')))
file2cd = dict(zip(gen.file, gen.class_dir))
for K in [2, 4, 8]:
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.RandomState(1000 + seed)
        tr = []
        for cd in sorted(set(file2cd.values())):
            f = [x for x in base['train'] if file2cd[x] == cd]; rng.shuffle(f); tr += f[:K]
        dump(f'fewshot_k{K}_s{seed}.json', {'train': tr, 'val': base['val'], 'test': base['test']},
             dict(unit='session', rule=f'{K} training sessions per class_dir sampled from session_split_s2026 train; val/test unchanged', seed=seed))
