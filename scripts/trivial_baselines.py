"""Trivial reference points for the session-scarce setting (K labelled sessions per class, same session-level splits):
  majority   : always predict the most frequent class of the labelled flows
  handshake  : LightGBM on the first 3 packets only (direction, IP and payload size, and the per-packet header size
               IP - payload, which separates UDP/QUIC from TCP and exposes TCP options): "is it just the transport?"
  knn10      : input-space 1-NN, L1 distance, on the first 10 packets (sizes log1p, IAT log1p(ms)), in the spirit of
               Luxemburk et al.'s input-space k-NN; also a check that session-level splits leave no near-duplicates
Records go to results/metrics.jsonl and results/locked_test_metrics.jsonl with tag 'trivial'."""
import argparse, json, os, sys, time, hashlib
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds


def feats_handshake(meta, n):
    m = meta[:, :3].copy(); k = np.minimum(n, 3)
    valid = np.arange(3)[None, :] < k[:, None]; m[~valid] = 0
    hdr = (m[..., 1] - m[..., 2]) * valid
    return np.concatenate([m[..., 0], np.log1p(m[..., 1]), np.log1p(m[..., 2]), hdr, k[:, None]], 1).astype(np.float32)


def feats_knn(meta, n, L=10):
    m = meta[:, :L].copy(); valid = np.arange(L)[None, :] < np.minimum(n, L)[:, None]
    x = np.stack([m[..., 0], np.log1p(m[..., 1]), np.log1p(m[..., 2]), np.log1p(m[..., 3] * 1000.0)], -1); x[~valid] = 0
    return x.reshape(len(x), -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--Ks', default='1,2,4,8'); ap.add_argument('--seeds', default='0,1,2,3,4'); a = ap.parse_args()
    import lightgbm as lgb
    for ds, tgt, pre in (('genai', 'joint', 'fs'), ('ccma', 'app', 'ccma_fs')):
        data = GenAIData(ROOT, prefix=ds); y = data.y_joint if tgt == 'joint' else data.y_app
        NC = data.n_joint if tgt == 'joint' else data.n_app; n_act = data.n_act if tgt == 'joint' else 1
        Fh = feats_handshake(data.meta, data.meta_len); Fk = feats_knn(data.meta, data.meta_len)
        for K in [int(k) for k in a.Ks.split(',')]:
            for s in [int(x) for x in a.seeds.split(',')]:
                sp = os.path.join('configs', 'splits', f'{"" if ds == "genai" else "ccma_"}fewshot_k{K}_s{s}.json')
                idx, sm = data.split_indices(os.path.join(ROOT, sp)); tr, iv, it = idx['train'], idx['val'], idx['test']
                preds = {}
                maj = np.bincount(y[tr], minlength=NC).argmax(); preds['majority'] = (np.full(len(iv), maj), np.full(len(it), maj))
                clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=5, random_state=s, verbose=-1)
                clf.fit(Fh[tr], y[tr]); preds['handshake'] = (clf.predict(Fh[iv]), clf.predict(Fh[it]))
                def nn1(Q):
                    out = np.empty(len(Q), int)
                    for b in range(0, len(Q), 512): out[b:b + 512] = y[tr][np.abs(Q[b:b + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)]
                    return out
                preds['knn10'] = (nn1(Fk[iv]), nn1(Fk[it]))
                for name, (pv, pt) in preds.items():
                    run = f'{pre}{K}_{name}_s{s}'; cfg = dict(model=name, split=sp, dataset=ds, target=tgt)
                    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
                    vm = metrics_from_preds(y[iv], pv, sessions=data.session_id[iv], seed=s, n_classes=NC, n_act=n_act)
                    tm = metrics_from_preds(y[it], pt, sessions=data.session_id[it], seed=s, n_classes=NC, n_act=n_act)
                    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh:
                        fh.write(json.dumps(dict(run_id=run, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag='trivial', cfg_hash=h, cfg=cfg, split_hash=sm['hash'], seed=s, val=vm)) + '\n')
                    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
                        fh.write(json.dumps(dict(run_id=run, cfg_hash=h, seed=s, tag='trivial', split_hash=sm['hash'], test=tm)) + '\n')
                    print(f'{run}: val macro_f1 {vm["macro_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
