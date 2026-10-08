"""K-session draws 5-19 for K = 1, 2 on both campaigns (20-draw primary statistics, scripts/draws20_primary.py).

Procedure: exactly that of the few-shot splits of seeds 0-4, only with more seeds.
  GenAI K=1  (scripts/make_splits_v3.py): rng = RandomState(1000 + seed); for each class_dir in sorted order of the classes
             present in session_split_s2026 train: shuffle that class's train sessions (in base-split order), keep the first K.
             Class key = genai_sessions.csv class_dir.
  GenAI K=2  (scripts/make_splits_v2.py): rng = RandomState(1000 + seed); for each class_dir in sorted order of all generic
             class_dirs: shuffle, keep the first K. Class key = class_dir of the generic sessions.
  CCMA K=1   (scripts/make_splits_v3.py): rng = RandomState(3000 + seed); apps sorted (those present in the base train),
             base = ccma_session_split_s2026.json, sessions with n_kept > 0.
  CCMA K=2   rng = RandomState(3000 + seed); apps sorted (all apps with n_kept > 0).
  val and test = those of the base session split, unchanged.

Checks (asserted before anything is written): the generator reproduces the files of seeds 0-4 exactly (train list in the
same order, val, test, hash); every draw of seeds 5-19 has the test and val sessions of seeds 0-4. The files of seeds 0-4
are only read; seeds 5-19 are written only if absent (or identical)."""
import json, os, hashlib
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); OUT = os.path.join(ROOT, 'configs', 'splits')


def digest(d): return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


gs = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'genai_sessions.csv'))
g_cls_v3 = dict(zip(gs.file, gs.class_dir))                                   # make_splits_v3
gen = gs[gs.part == 'generic']; g_cls_v2 = dict(zip(gen.file, gen.class_dir))  # make_splits_v2
cs = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'ccma_sessions.csv')); cs = cs[cs.n_kept > 0]; c_cls = dict(zip(cs.file, cs.app))
base = json.load(open(os.path.join(OUT, 'session_split_s2026.json'))); cbase = json.load(open(os.path.join(OUT, 'ccma_session_split_s2026.json')))


def draw(ds, K, seed):
    if ds == 'genai':
        rng = np.random.RandomState(1000 + seed); files = base['train']
        if K == 1:
            classes, cls = sorted(set(g_cls_v3[f] for f in files)), g_cls_v3
            meta = dict(unit='session', rule='1 training session per class_dir from session_split_s2026 train', seed=seed)
        else:
            classes, cls = sorted(set(g_cls_v2.values())), g_cls_v2
            meta = dict(unit='session', rule=f'{K} training sessions per class_dir sampled from session_split_s2026 train; val/test unchanged', seed=seed)
        b = base
    else:
        rng = np.random.RandomState(3000 + seed); files = cbase['train']
        if K == 1:
            classes = sorted(set(c_cls[f] for f in files)); meta = dict(dataset='ccma', rule='1 training session per app', seed=seed)
        else:
            classes = sorted(set(c_cls.values())); meta = dict(dataset='ccma', rule=f'{K} training sessions per app', seed=seed)
        cls, b = c_cls, cbase
    tr = []
    for c in classes:
        f = [x for x in files if cls[x] == c]; rng.shuffle(f); tr += f[:K]
    sp = {'train': tr, 'val': b['val'], 'test': b['test']}
    meta = dict(meta, hash=digest(sp), counts={k: len(v) for k, v in sp.items()})
    return {'meta': meta, **sp}


def main():
    for ds in ('genai', 'ccma'):
        pre = '' if ds == 'genai' else 'ccma_'
        for K in (1, 2):
            ref = None
            for seed in range(5):  # the generator must reproduce the files of seeds 0-4
                name = f'{pre}fewshot_k{K}_s{seed}.json'; old = json.load(open(os.path.join(OUT, name))); new = draw(ds, K, seed)
                for k in ('train', 'val', 'test'): assert old[k] == new[k], (name, k)
                assert old['meta']['hash'] == new['meta']['hash'], name
                if ref is None: ref = old
                assert old['test'] == ref['test'] and old['val'] == ref['val']
            print(f'{ds} K={K}: generator reproduces seeds 0-4 exactly', flush=True)
            trains = []
            for seed in range(5, 20):
                new = draw(ds, K, seed); assert new['test'] == ref['test'] and new['val'] == ref['val']
                name = f'{pre}fewshot_k{K}_s{seed}.json'; path = os.path.join(OUT, name)
                if os.path.exists(path):
                    old = json.load(open(path)); assert all(old[k] == new[k] for k in ('train', 'val', 'test')), f'{name} exists and differs'
                else:
                    json.dump(new, open(path, 'w'), indent=1)
                trains.append(tuple(sorted(new['train']))); print(name, new['meta']['hash'], new['meta']['counts'], flush=True)
            allt = [tuple(sorted(json.load(open(os.path.join(OUT, f'{pre}fewshot_k{K}_s{s}.json')))['train'])) for s in range(20)]
            print(f'{ds} K={K}: {len(set(allt))} distinct labelled-session sets among 20 draws', flush=True)


if __name__ == '__main__':
    main()
