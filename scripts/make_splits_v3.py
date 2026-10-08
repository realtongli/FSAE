"""One-session and unseen-phone splits.
  fewshot_k1_s{seed}.json / ccma_fewshot_k1_s{seed}.json : one training session per class (val/test of the standard split unchanged).
  cd_{dev}_k{K}_s{seed}.json / ccma_cd_{dev}_k{K}_s{seed}.json : cross-device few-shot. From the leave-one-device-out split with test
      device {dev}, keep K training sessions per class from the other device(s); val = LODO val (other devices); test = all sessions of
      the held-out device. The attacker therefore labels K sessions on its own phone(s) and attacks a phone it has never seen."""
import json, os, hashlib, numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); OUT = os.path.join(ROOT, 'configs', 'splits')


def digest(d): return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


def dump(name, split, meta):
    meta = dict(meta, hash=digest(split), counts={k: len(v) for k, v in split.items()})
    json.dump({'meta': meta, **split}, open(os.path.join(OUT, name), 'w'), indent=1); print(name, meta['counts'])


# ---- class keys per dataset ----
gs = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'genai_sessions.csv')); g_cls = dict(zip(gs.file, gs.class_dir)) if 'class_dir' in gs else None
if g_cls is None:  # fall back: class from file path
    g_cls = {f: f.split('/')[1] if '/' in f else f for f in gs.file}
cs = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'ccma_sessions.csv')); cs = cs[cs.n_kept > 0]; c_cls = dict(zip(cs.file, cs.app))


def sample(files, cls, K, rng):
    tr = []
    for c in sorted(set(cls[f] for f in files)):
        f = [x for x in files if cls[x] == c]; rng.shuffle(f); tr += f[:K]
    return tr


# ---- K=1 ----
base = json.load(open(os.path.join(OUT, 'session_split_s2026.json')))
for seed in range(5):
    rng = np.random.RandomState(1000 + seed)
    dump(f'fewshot_k1_s{seed}.json', {'train': sample(base['train'], g_cls, 1, rng), 'val': base['val'], 'test': base['test']}, dict(unit='session', rule='1 training session per class_dir from session_split_s2026 train', seed=seed))
cbase = json.load(open(os.path.join(OUT, 'ccma_session_split_s2026.json')))
for seed in range(5):
    rng = np.random.RandomState(3000 + seed)
    dump(f'ccma_fewshot_k1_s{seed}.json', {'train': sample(cbase['train'], c_cls, 1, rng), 'val': cbase['val'], 'test': cbase['test']}, dict(dataset='ccma', rule='1 training session per app', seed=seed))

# ---- cross-device few-shot ----
K = 4
for dev in ['8e', 'a6']:
    lodo = json.load(open(os.path.join(OUT, f'lodo_test-{dev}_s2026.json')))
    for seed in range(5):
        rng = np.random.RandomState(5000 + seed)
        dump(f'cd_{dev}_k{K}_s{seed}.json', {'train': sample(lodo['train'], g_cls, K, rng), 'val': lodo['val'], 'test': lodo['test']},
             dict(unit='session', rule=f'{K} training sessions per class_dir from the non-{dev} device; test = all sessions of device {dev}', test_device=dev, seed=seed))
for dev, tag in [('2cae2bfb3a67', '2c'), ('f4428fb1642f', 'f4'), ('f8cfc5d00d9c', 'f8')]:
    lodo = json.load(open(os.path.join(OUT, f'ccma_lodo_test-{dev}_s2026.json')))
    for seed in range(5):
        rng = np.random.RandomState(6000 + seed)
        dump(f'ccma_cd_{tag}_k{K}_s{seed}.json', {'train': sample(lodo['train'], c_cls, K, rng), 'val': lodo['val'], 'test': lodo['test']},
             dict(dataset='ccma', rule=f'{K} training sessions per app from the other devices; test = all sessions of device {tag}', test_device=tag, seed=seed))
