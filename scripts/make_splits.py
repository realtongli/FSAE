"""Fixed session-level (capture-group) splits for MIRAGE-GenAI-2025 generic data.

Split is done on capture SESSIONS (one 15-min JSON = one group); all biflows of a session
stay in the same set. Stratified by class_dir x device so that every class and both devices
appear in train / val / test. Ratios approx 60/20/20 by session count.

Also emits leave-one-device-out (LODO) splits: train+val on one device, test on the other.

Writes configs/splits/session_split_s{seed}.json and configs/splits/lodo_test-{dev}_s{seed}.json with sha256
of the sorted session file lists, so any experiment can record the split hash.
"""
import json, os, hashlib, argparse
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'configs', 'splits')
os.makedirs(OUT, exist_ok=True)


def digest(d):
    return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


def main(seed):
    ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'genai_sessions.csv'))
    gen = ses[ses.part == 'generic']
    rng = np.random.RandomState(seed)
    split = {'train': [], 'val': [], 'test': []}
    for (cd, dev), g in gen.groupby(['class_dir', 'device']):
        files = list(g.file)
        rng.shuffle(files)
        n = len(files); n_test = max(1, int(round(0.2 * n))); n_val = max(1, int(round(0.2 * n)))
        split['test'] += files[:n_test]; split['val'] += files[n_test:n_test + n_val]; split['train'] += files[n_test + n_val:]
    meta = {'seed': seed, 'unit': 'capture session (one JSON file)', 'ratios': '60/20/20 by session count, stratified by class_dir x device',
            'hash': digest(split), 'counts': {k: len(v) for k, v in split.items()}}
    with open(os.path.join(OUT, f'session_split_s{seed}.json'), 'w') as fh:
        json.dump({'meta': meta, **split}, fh, indent=1)
    print('session split', meta)
    # LODO
    for dev in sorted(gen.device.unique()):
        other = gen[gen.device != dev]; this = gen[gen.device == dev]
        tr, va = [], []
        for cd, g in other.groupby('class_dir'):
            files = list(g.file); rng.shuffle(files)
            n_val = max(1, int(round(0.2 * len(files))))
            va += files[:n_val]; tr += files[n_val:]
        lodo = {'train': tr, 'val': va, 'test': list(this.file)}
        meta = {'seed': seed, 'test_device': dev, 'hash': digest(lodo), 'counts': {k: len(v) for k, v in lodo.items()}}
        with open(os.path.join(OUT, f'lodo_test-{dev.replace(":", "")}_s{seed}.json'), 'w') as fh:
            json.dump({'meta': meta, **lodo}, fh, indent=1)
        print('lodo', meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2026)
    main(ap.parse_args().seed)
