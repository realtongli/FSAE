"""Session-level splits for the third evaluation set: the nine MIRAGE-AppAct-2024 apps that CCMA does not contain and
that have at least 50 capture sessions (Clash Royale and KakaoTalk have fewer and are not used).

  appact_session_split_s2026.json   stratified app x device 60/20/20, exactly the procedure of scripts/make_splits_ccma.py
  appact_fewshot_k{K}_s{r}.json     K sessions per app from the training sessions, RandomState(7000 + r); the procedure of
                                    scripts/make_splits_draws20.draw(); K = 1, 2: r = 0..19, K = 4, 8: r = 0..4
  appact_cd_{d}_k4_s{r}.json        unseen phone d in {1e, 2c, f4}: classes = eligible apps (>= 10 sessions on at least two phones)
                                    with >= 10 sessions on d; test = their sessions on d; K = 4 from the other phones, RandomState(8000 + r)
  appact_cdexcl_{d}.json            'test' = every nine-app session captured on phone d (removed from that phone's pre-training corpus)

The base split and every K-session draw must reproduce the digests of the released split files (asserted before anything
is written).
Existing files are never rewritten: a file is written only if absent, and must be identical if present."""
import hashlib, json, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); OUT = os.path.join(ROOT, 'configs', 'splits')
NINE = ['Crunchyroll', 'JitsiMeet', 'Line', 'Omlet', 'Signal', 'Telegram', 'Trueconf', 'Twitch', 'WhatsApp']
N_DRAWS = {1: 20, 2: 20, 4: 5, 8: 5}
PIN_BASE, PIN_COUNTS = '7d2812b97c8659b3', {'train': 475, 'val': 163, 'test': 163}
PIN_DRAWS = {1: 'abeacafa25acfda8', 2: '29f6fe6db52b2696', 4: '8cfb7f3b2e28a643', 8: 'd54512179840b9cd'}
PIN_CD = {'1e': ['Crunchyroll', 'Line', 'Omlet', 'Signal', 'Telegram'], '2c': ['Crunchyroll', 'Line', 'Omlet', 'Signal', 'Telegram', 'WhatsApp'],
          'f4': ['Line', 'Omlet', 'Signal', 'Telegram', 'WhatsApp']}


def digest(d): return hashlib.sha256(json.dumps({k: sorted(v) for k, v in d.items()}, sort_keys=True).encode()).hexdigest()[:16]


def dump(name, split, meta):
    meta = dict(meta, hash=digest(split), counts={k: len(v) for k, v in split.items()}); new = {'meta': meta, **split}; path = os.path.join(OUT, name)
    if os.path.exists(path):
        assert json.load(open(path)) == new, f'{name} exists and differs'
    else:
        json.dump(new, open(path, 'w'), indent=1)
    return meta['hash']


def main():
    ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'appact_sessions.csv'))
    ses = ses[ses.app.isin(NINE) & (ses.n_kept > 0)]
    f2a = dict(zip(ses.file, ses.app))
    # ---- base split (scripts/make_splits_ccma.py, line for line)
    rng = np.random.RandomState(2026); sp = {'train': [], 'val': [], 'test': []}
    for (app, dev), g in ses.groupby(['app', 'device']):
        f = list(g.file); rng.shuffle(f); n = len(f); nt = max(1, int(round(0.2 * n))); nv = max(1, int(round(0.2 * n)))
        sp['test'] += f[:nt]; sp['val'] += f[nt:nt + nv]; sp['train'] += f[nt + nv:]
    assert digest(sp) == PIN_BASE and {k: len(v) for k, v in sp.items()} == PIN_COUNTS, (digest(sp), {k: len(v) for k, v in sp.items()})
    # ---- K-session draws: everything is checked against the digests before the first file is written
    draws = {}
    for K, n in N_DRAWS.items():
        hs = []
        for r in range(n):
            rg = np.random.RandomState(7000 + r); tr = []
            for ap in sorted(set(f2a[f] for f in sp['train'])):
                f = [x for x in sp['train'] if f2a[x] == ap]; rg.shuffle(f); tr += f[:K]
            d = {'train': tr, 'val': sp['val'], 'test': sp['test']}; draws[(K, r)] = d; hs.append(digest(d))
        assert hashlib.sha256(''.join(hs).encode()).hexdigest()[:16] == PIN_DRAWS[K], f'K={K}: draws do not reproduce the released digests'
        assert len(set(hs)) == n
    h = dump('appact_session_split_s2026.json', sp, dict(dataset='appact', unit='session', rule='60/20/20 by session, stratified app x device; nine apps absent from CCMA', seed=2026))
    print('appact_session_split_s2026.json', h, {k: len(v) for k, v in sp.items()})
    for (K, r), d in draws.items():
        dump(f'appact_fewshot_k{K}_s{r}.json', d, dict(dataset='appact', rule=f'{K} training sessions per app', seed=r))
    print('K-session draws written and equal to the released digests:', {K: n for K, n in N_DRAWS.items()})
    # ---- unseen phone
    tab = ses.groupby(['app', 'device']).size().unstack(fill_value=0)
    elig = [a for a in tab.index if (tab.loc[a] >= 10).sum() >= 2]
    assert elig == ['Crunchyroll', 'Line', 'Omlet', 'Signal', 'Telegram', 'WhatsApp'], elig
    held = {}
    for dev in tab.columns:
        cls = [a for a in elig if tab.loc[a, dev] >= 10]
        if len(cls) >= 3: held[dev[:2]] = (dev, cls)
    assert {t: c for t, (_, c) in held.items()} == PIN_CD, held
    for tag, (dev, cls) in held.items():
        test = list(ses[(ses.device == dev) & ses.app.isin(cls)].file); other = ses[(ses.device != dev) & ses.app.isin(cls)]
        rng = np.random.RandomState(2026); tr, va = [], []
        for app, g in other.groupby('app'):
            f = list(g.file); rng.shuffle(f); nv = max(1, int(round(0.2 * len(f)))); va += f[:nv]; tr += f[nv:]
        for r in range(5):
            rg = np.random.RandomState(8000 + r); k4 = []
            for ap in sorted(cls):
                f = [x for x in tr if f2a[x] == ap]; rg.shuffle(f); assert len(f) >= 4, (tag, ap); k4 += f[:4]
            dump(f'appact_cd_{tag}_k4_s{r}.json', {'train': k4, 'val': va, 'test': test},
                 dict(dataset='appact', rule=f'4 training sessions per app from the other phones; test = all sessions of the {len(cls)} classes on phone {tag}', test_device=tag, classes=cls, seed=r))
        dump(f'appact_cdexcl_{tag}.json', {'train': [], 'val': [], 'test': list(ses[ses.device == dev].file)},
             dict(dataset='appact', rule=f'every nine-app session captured on phone {tag}; removed from the pre-training corpus of the unseen-phone evaluation', test_device=tag))
        print(f'unseen phone {tag}: {len(cls)} classes {cls}; test sessions {len(test)}, pool train {len(tr)} val {len(va)}; sessions on the phone {int((ses.device == dev).sum())}')


if __name__ == '__main__':
    main()
