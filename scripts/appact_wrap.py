"""Run the training code of src/ and scripts/ unchanged on the third evaluation set (the nine MIRAGE-AppAct-2024 apps
absent from CCMA).

    python scripts/appact_wrap.py pretrain  <arguments of src/pretrain_meta.py>       (corpus token 'appact' is understood)
    python scripts/appact_wrap.py ft        <arguments of src/train_meta_ssl.py>      (--dataset appact is accepted)
    python scripts/appact_wrap.py lgbm      <arguments of scripts/train_baselines.py> (--dataset appact is accepted)
    python scripts/appact_wrap.py selfcheck                                           (data path only, no model, no metric)

The wrapper calls the unmodified main() of those scripts. It changes four things around them and nothing inside them:
  1. data_mfr.GenAIData knows the prefixes 'genai' and 'ccma' only; here the name resolves to AppActData for 'appact'
     (same arrays; labels = index of the app among the nine, -1 for Clash Royale and KakaoTalk, which no split contains).
  2. '--dataset' accepts 'appact'.
  3. Every append to results/metrics.jsonl and results/locked_test_metrics.jsonl goes to
     <RES>/metrics_appact.jsonl and <RES>/locked_test_metrics_appact.jsonl, and the run directories live in <RES>/runs
     (RES = environment variable APPACT_RES, default results/appact). An audit hook refuses any write to the two
     record files of GenAI and CCMA.
  4. pretrain: the corpus is that of the unmodified load_corpus for the other tokens, followed by the connections of the
     nine apps in the training and validation sessions of configs/splits/appact_session_split_s2026.json (labels unused),
     minus the sessions listed under 'test' in an --exclude appact:<json> file.
With --dataset ccma the wrapped runs give the same test predictions as the unwrapped scripts."""
import argparse, builtins, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import data_mfr

NINE = ['Crunchyroll', 'JitsiMeet', 'Line', 'Omlet', 'Signal', 'Telegram', 'Trueconf', 'Twitch', 'WhatsApp']
ELEVEN = ['ClashRoyale', 'Crunchyroll', 'JitsiMeet', 'KakaoTalk', 'Line', 'Omlet', 'Signal', 'Telegram', 'Trueconf', 'Twitch', 'WhatsApp']
BASE_SPLIT = os.path.join('configs', 'splits', 'appact_session_split_s2026.json')
RES_REL = os.environ.get('APPACT_RES', os.path.join('results', 'appact')); RES = os.path.join(ROOT, RES_REL)
LOCKED = os.path.normpath(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'))
METRICS = os.path.normpath(os.path.join(ROOT, 'results', 'metrics.jsonl'))
_OrigData = data_mfr.GenAIData


class AppActData(_OrigData):
    """The arrays of data/derived/appact_* with the interface of data_mfr.GenAIData. y_app = y_joint = 0..8 (NINE order)."""

    def __init__(self, root=ROOT, prefix='appact'):
        self.prefix = 'appact'; self.n_act = 1; self.n_app = len(NINE)
        d = np.load(os.path.join(root, 'data', 'derived', 'appact_biflows.npz'))
        self.pkt_bytes, self.pkt_len, self.pkt_sni = d['pkt_bytes'], d['pkt_len'], d['pkt_sni']
        self.flat_bytes, self.flat_len, self.flat_sni = d['flat_bytes'], d['flat_len'], d['flat_sni']
        self.meta, self.meta_len = d['meta'], d['meta_len']
        self.sessions = pd.read_csv(os.path.join(root, 'data', 'derived', 'appact_sessions.csv'))
        self.index = pd.read_csv(os.path.join(root, 'data', 'derived', 'appact_index.csv'))
        app = self.index.app.values
        assert len(app) == len(self.meta) and np.array_equal(d['y_app'], np.array([ELEVEN.index(a) for a in app])), 'index and arrays disagree'
        lab = np.array([NINE.index(a) if a in NINE else -1 for a in app], np.int64)
        self.y_app = lab; self.y_joint = lab.copy(); self.y_act = np.full(len(lab), -1, np.int64)
        self.joint_map = {i: i for i in range(len(NINE))}; self.n_joint = len(NINE)
        self.session_id, self.device_id = d['session_id'], d['device_id']
        self.file2sid = dict(zip(self.sessions.file, self.sessions.session_id))


def Data(root=ROOT, prefix='genai'):
    return AppActData(root) if prefix == 'appact' else _OrigData(root, prefix)


def _open(file, *a, **k):
    if isinstance(file, (str, os.PathLike)):
        p = os.path.normpath(os.fspath(file))
        if p == LOCKED: file = os.path.join(RES, 'locked_test_metrics_appact.jsonl')
        elif p == METRICS: file = os.path.join(RES, 'metrics_appact.jsonl')
    return builtins.open(file, *a, **k)


def _audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, os.PathLike)) and isinstance(args[1], str):
        if os.path.normpath(os.fspath(args[0])) in (LOCKED, METRICS) and any(c in args[1] for c in 'wax+'):
            raise RuntimeError(f'appact_wrap: refusing to write {args[0]}')


def _accept_appact():
    add = argparse.ArgumentParser.add_argument

    def add_argument(self, *names, **kw):
        if '--dataset' in names and kw.get('choices') is not None and 'appact' not in kw['choices']: kw['choices'] = list(kw['choices']) + ['appact']
        return add(self, *names, **kw)
    argparse.ArgumentParser.add_argument = add_argument


def appact_corpus(first_k, exclude_json=None):
    """Connections of the nine apps in the training and validation sessions of the base split, as load_corpus builds them."""
    from train_meta import features
    data = AppActData(ROOT); sp = json.load(open(os.path.join(ROOT, BASE_SPLIT))); keep = set(sp['train'] + sp['val'])
    assert not (keep & set(sp['test']))
    if exclude_json:
        drop = set(json.load(open(os.path.join(ROOT, exclude_json)))['test']); print(f'appact: excluding {len(keep & drop)} sessions of {exclude_json}', flush=True); keep -= drop
    sids = np.array([data.file2sid[f] for f in sorted(keep)])
    idx = np.where(np.isin(data.session_id, sids) & (data.y_joint >= 0))[0]
    X = features(data, first_k)[idx]; V = (np.arange(first_k)[None, :] < np.minimum(data.meta_len[idx], first_k)[:, None])
    print(f'appact: {len(idx)} flows from {len(sids)} train+val sessions', flush=True)
    return X, V, idx


def run_pretrain():
    import pretrain_meta as m
    orig = m.load_corpus

    def load_corpus(datasets, first_k, exclude=None, target_frac=1.0, frac_seed=0):
        exclude = dict(exclude or {}); ex_app = exclude.pop('appact', None); other = [d for d in datasets if d != 'appact']
        X0, V0 = orig(other, first_k, exclude, target_frac, frac_seed)
        if other == ['genai', 'ccma'] and not exclude and target_frac == 1.0: assert len(X0) == 28394, len(X0)
        if 'appact' not in datasets: return X0, V0
        Xa, Va, _ = appact_corpus(first_k, ex_app)
        if other == ['genai', 'ccma'] and not exclude and not ex_app and target_frac == 1.0: assert len(Xa) == 19614, len(Xa)
        X, V = np.concatenate([X0, Xa]), np.concatenate([V0, Va])
        assert np.array_equal(X[:len(X0)], X0) and np.array_equal(V[:len(V0)], V0)
        print(f'corpus: {len(X0)} existing + {len(Xa)} appact = {len(X)} flows', flush=True)
        return X, V
    m.load_corpus = load_corpus; m.GenAIData = Data
    name = sys.argv[sys.argv.index('--name') + 1]
    assert 'appact' in name and not os.path.exists(os.path.join(ROOT, 'weights', f'{name}.pt')), f'weights/{name}.pt: refusing to overwrite'
    m.main()


def run_ft():
    if '--out_dir' not in sys.argv: sys.argv += ['--out_dir', os.path.join(RES_REL, 'runs')]
    import train_meta_ssl as m
    m.GenAIData = Data; m.open = _open; m.main()


def run_lgbm():
    assert os.environ.get('CKPT_SELECT') == 'last' and os.environ.get('LGBM_ROUNDS') == '1600', 'fixed recipe: CKPT_SELECT=last LGBM_ROUNDS=1600'
    assert sys.argv[sys.argv.index('--which') + 1] == 'lgbm'
    import train_baselines as m

    def save_preds(name, seed, idx, y, pt):
        d = os.path.join(RES, 'runs', f'{name}_s{seed}'); os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, 'test_preds.npz'), y=y[idx['test']], pred=pt, idx=idx['test'])
    m.GenAIData = Data; m.open = _open; m.save_preds = save_preds; m.main()


def selfcheck():
    """Data path only: shapes, labels and split sizes. No model is built and no metric is computed."""
    D = AppActData(ROOT); idx, meta = D.split_indices(os.path.join(ROOT, BASE_SPLIT))
    print('flows', len(D.meta), 'nine-app flows', int((D.y_app >= 0).sum()), 'per class', np.bincount(D.y_app[D.y_app >= 0], minlength=9).tolist())
    print('base split', meta['hash'], meta['counts'], {k: int(len(v)) for k, v in idx.items()})
    assert meta['hash'] == '7d2812b97c8659b3' and [len(idx[k]) for k in ('train', 'val', 'test')] == [14856, 4758, 5044]
    for K, n in ((1, 20), (2, 20), (4, 5), (8, 5)):
        for r in range(n):
            i2, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'appact_fewshot_k{K}_s{r}.json'))
            assert np.array_equal(i2['test'], idx['test']) and np.array_equal(i2['val'], idx['val']) and set(i2['train']) <= set(idx['train'])
            assert len(np.unique(D.y_app[i2['train']])) == 9 and len(np.unique(D.session_id[i2['train']])) == 9 * K
    print('all K-session draws: nine classes, 9K sessions, val and test of the base split')


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    os.makedirs(os.path.join(RES, 'runs'), exist_ok=True)
    sys.addaudithook(_audit); _accept_appact()
    {'pretrain': run_pretrain, 'ft': run_ft, 'lgbm': run_lgbm, 'selfcheck': selfcheck}[mode]()
