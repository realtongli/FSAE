"""Session-level splits for MIRAGE-COVID-CCMA-2022: stratified session split (app x device, 60/20/20) and
leave-one-device-out for each of the 3 phones (val = 20% of the training devices' sessions)."""
import json, os, hashlib
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'configs', 'splits'); os.makedirs(OUT, exist_ok=True)


def digest(d):
    return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


def dump(name, split, meta):
    meta = dict(meta, hash=digest(split), counts={k: len(v) for k, v in split.items()})
    json.dump({'meta': meta, **split}, open(os.path.join(OUT, name), 'w'), indent=1); print(name, meta)


ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'ccma_sessions.csv'))
ses = ses[ses.n_kept > 0]
rng = np.random.RandomState(2026)
sp = {'train': [], 'val': [], 'test': []}
for (app, dev), g in ses.groupby(['app', 'device']):
    f = list(g.file); rng.shuffle(f); n = len(f); nt = max(1, int(round(0.2 * n))); nv = max(1, int(round(0.2 * n)))
    sp['test'] += f[:nt]; sp['val'] += f[nt:nt + nv]; sp['train'] += f[nt + nv:]
dump('ccma_session_split_s2026.json', sp, dict(dataset='ccma', unit='session', rule='60/20/20 by session, stratified app x device', seed=2026))
for dev in sorted(ses.device.unique()):
    other = ses[ses.device != dev]; this = ses[ses.device == dev]; tr, va = [], []
    for app, g in other.groupby('app'):
        f = list(g.file); rng.shuffle(f); nv = max(1, int(round(0.2 * len(f)))); va += f[:nv]; tr += f[nv:]
    dump(f'ccma_lodo_test-{dev.replace(":", "")}_s2026.json', {'train': tr, 'val': va, 'test': list(this.file)}, dict(dataset='ccma', test_device=dev, seed=2026))
print(ses.groupby(['app', 'device']).size().unstack(fill_value=0))
