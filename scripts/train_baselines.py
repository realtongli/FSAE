"""Reference observers on the same session-level split (standard or session-scarce).
 (1) lgbm   : LightGBM on training-set-fitted statistical features from the first 64 packets (dir/len/iat), no payload bytes.
 (2) cnn    : Paper-style 1D-CNN (Montieri et al. 2026, Comput. Netw.): first 512 raw payload bytes, two conv layers (16,32 filters,
              kernel 25) + ReLU + maxpool(3), then FC 256 with dropout 0.2.  Re-implemented from the paper description; not the authors' code.
 (3) dfmeta : Deep-Fingerprinting-style 1D-CNN (Sirinam et al., CCS 2018: 4 conv blocks 32/64/128/256 filters, kernel 8, BN, ELU,
              max-pool, dropout, 2 FC with dropout 0.7/0.5; Adamax 2e-3, batch 128, 30 epochs, glorot init, as in the official code) adapted to the 64-packet x 4-channel metadata sequence used by the metadata attacker
              (same threat model: direction, sizes, IAT; no payload). Pool size 2 instead of 8 because L=64.
The checkpoint rule is CKPT_SELECT (see below); test metrics go to results/locked_test_metrics.jsonl.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds
from train_meta import features as meta_features


def stat_features(meta, meta_len):
    N = meta.shape[0]; feats = []
    for i in range(N):
        n = int(meta_len[i]); m = meta[i, :n]
        d, ipl, pl, iat = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
        up, dn = pl[d > 0], pl[d < 0]
        f = [n, (d > 0).sum(), (d < 0).sum(), pl.sum(), up.sum(), dn.sum(), (pl > 0).sum(),
             pl.mean(), pl.std(), pl.max(), np.median(pl), ipl.mean(), ipl.std(),
             up.mean() if len(up) else 0, up.std() if len(up) else 0, up.max() if len(up) else 0,
             dn.mean() if len(dn) else 0, dn.std() if len(dn) else 0, dn.max() if len(dn) else 0,
             iat.sum(), iat.mean(), iat.std(), iat.max(), np.median(iat), np.log1p(iat).mean()]
        sig = (pl * d)[:20]; sig = np.pad(sig, (0, 20 - len(sig)))
        f += list(sig)
        feats.append(f)
    return np.array(feats, np.float32)


class PaperCNN(nn.Module):
    def __init__(self, n_cls=6, L=512):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 16, 25, padding=12), nn.ReLU(), nn.MaxPool1d(3),
                                 nn.Conv1d(16, 32, 25, padding=12), nn.ReLU(), nn.MaxPool1d(3))
        with torch.no_grad():
            d = self.net(torch.zeros(1, 1, L)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_cls))  # FC 256 + dropout 0.2 (Montieri et al. 2026, Sec. 3.3)

    def forward(self, x):
        return self.fc(self.net(x))


class DFMetaCNN(nn.Module):
    """Deep Fingerprinting (Sirinam et al. 2018) block structure on a [B, 4, 64] metadata sequence."""

    def __init__(self, n_cls=6, L=64, C=4):
        super().__init__()
        chans = [C, 32, 64, 128, 256]; blocks = []
        for i in range(4):
            act = nn.ELU() if i == 0 else nn.ReLU()
            blocks += [nn.Conv1d(chans[i], chans[i + 1], 8, padding='same'), nn.BatchNorm1d(chans[i + 1]), act,
                       nn.Conv1d(chans[i + 1], chans[i + 1], 8, padding='same'), nn.BatchNorm1d(chans[i + 1]), act,
                       nn.MaxPool1d(2), nn.Dropout(0.1)]
        self.net = nn.Sequential(*blocks)
        with torch.no_grad():
            d = self.net(torch.zeros(1, C, L)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.7),
                                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, n_cls))
        for mod in self.modules():  # glorot-uniform weights, zero biases, as in the official DF code
            if isinstance(mod, (nn.Conv1d, nn.Linear)): nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)

    def forward(self, x):
        return self.fc(self.net(x))


def save_preds(name, seed, idx, y, pt):
    d = os.path.join(ROOT, 'results', 'runs', f'{name}_s{seed}'); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, 'test_preds.npz'), y=y[idx['test']], pred=pt, idx=idx['test'])


def log_rec(name, cfg, seed, split_meta, val_m, test_m, extra, tag):
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    rec = dict(run_id=f'{name}_s{seed}', ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=split_meta['hash'], seed=seed, val=val_m, **extra)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=rec['run_id'], cfg_hash=cfg_hash, seed=seed, tag=tag, split_hash=split_meta['hash'], test=test_m)) + '\n')
    print(f'{rec["run_id"]}: val macro_f1 {val_m["macro_f1"]:.4f} ci {val_m["macro_f1_ci95_sessions"]} app {val_m["app_macro_f1"]:.4f} act {val_m["act_macro_f1"]:.4f}', flush=True)


# CKPT_SELECT=last: the torch models keep the final epoch of their fixed schedule and LightGBM is trained for exactly
# LGBM_ROUNDS boosting rounds without early stopping. The number of rounds is read from the environment variable
# LGBM_ROUNDS; its value is fixed once on CCMA validation sessions by scripts/select_lgbm_rounds.py and stored in
# configs/lgbm_rounds.json, so a run selects nothing on labelled sessions beyond its K training sessions (validation
# metrics are still computed and logged). CKPT_SELECT=val (default): the torch models keep the epoch with the best
# validation macro-F1 and LightGBM stops early on the validation sessions.
LAST = os.environ.get('CKPT_SELECT', 'val') == 'last'


def train_torch(model, X, y, idx, seed, epochs, lr, opt_name, dev='cuda', bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(dev)
    opt = torch.optim.Adamax(model.parameters(), lr=lr) if opt_name == 'adamax' else torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(); tr = idx['train']; best = (-1, None, -1); t0 = time.time()
    Xv = X[idx['val']].to(dev)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]
            if len(bi) < 2: continue  # BatchNorm needs >1 sample
            xb = X[bi].to(dev); yb = torch.from_numpy(y[bi]).to(dev)
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).cpu().numpy()
        f = f1_score(y[idx['val']], pv, average='macro')
        if (ep == epochs - 1) if LAST else (f > best[0]):
            best = (f, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pv = model(Xv).argmax(1).cpu().numpy(); pt = model(X[idx['test']].to(dev)).argmax(1).cpu().numpy()
    return pv, pt, best[2], round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--which', default='all'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--split', default='configs/splits/session_split_s2026.json'); ap.add_argument('--mask_sni', type=int, default=0)
    ap.add_argument('--dataset', default='genai', choices=['genai', 'ccma']); ap.add_argument('--target', default='joint', choices=['joint', 'app'])
    ap.add_argument('--tag', default='baseline'); ap.add_argument('--name', default='', help='run-name prefix (default: base or base_<dataset>)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--defense', default='none', choices=['none', 'pad256', 'jitter20', 'pad256_jitter20', 'front', 'tamaraw', 'ech', 'ech_full'])
    ap.add_argument('--defense_train', type=int, default=0, help='1 = adaptive attacker (defense applied to training data too); 0 = only val/test are defended')
    a = ap.parse_args()
    data = GenAIData(ROOT, prefix=a.dataset); idx, split_meta = data.split_indices(os.path.join(ROOT, a.split))
    if a.defense != 'none':  # same shaping as the metadata attacker, applied to the raw meta before any feature extraction
        from train_meta import apply_defense, NAMED_DEFENSES
        seed0 = int(a.seeds.split(',')[0])
        if a.defense in NAMED_DEFENSES:
            from wf_defenses import apply_named
            shaped, slen, ov = apply_named(a.defense, data.meta, data.meta_len, seed0); print('defense overhead', a.defense, ov, flush=True)
        else:
            shaped = apply_defense(data.meta, a.defense, seed0); slen = data.meta_len
        if a.defense_train: data.meta, data.meta_len = shaped, slen
        else:
            m = data.meta.copy(); ln = data.meta_len.copy(); di = np.concatenate([idx['val'], idx['test']])
            m[di] = shaped[di]; ln[di] = slen[di]; data.meta, data.meta_len = m, ln
    y = data.y_joint if a.target == 'joint' else data.y_app; sess = data.session_id
    NC = data.n_joint if a.target == 'joint' else data.n_app; n_act = data.n_act if a.target == 'joint' else 1
    prefix = a.name or ('base' if a.dataset == 'genai' else f'base_{a.dataset}')
    seeds = [int(s) for s in a.seeds.split(',')]
    M = lambda yy, pp, ii, seed: metrics_from_preds(yy, pp, sess[ii], seed=seed, n_classes=NC, n_act=n_act)
    which = a.which.split(',') if a.which != 'all' else ['lgbm', 'cnn', 'dfmeta']
    if 'lgbm' in which:
        import lightgbm as lgb
        X = stat_features(data.meta, data.meta_len)
        mu, sd = X[idx['train']].mean(0), X[idx['train']].std(0) + 1e-6; X = (X - mu) / sd  # fit on train only
        for seed in seeds:
            t0 = time.time()
            n_est = int(os.environ['LGBM_ROUNDS']) if LAST else 2000
            clf = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=seed, verbose=-1)
            if LAST: clf.fit(X[idx['train']], y[idx['train']])
            else: clf.fit(X[idx['train']], y[idx['train']], eval_set=[(X[idx['val']], y[idx['val']])], callbacks=[lgb.early_stopping(100, verbose=False)])
            pv = clf.predict(X[idx['val']]); pt = clf.predict(X[idx['test']]); save_preds(f'{prefix}_lgbm_meta64', seed, idx, y, pt)
            log_rec(f'{prefix}_lgbm_meta64', dict(model='lgbm', feats='stats64', dataset=a.dataset, target=a.target, split=a.split, n_est=n_est if LAST else int(clf.best_iteration_)), seed, split_meta,
                    M(y[idx['val']], pv, idx['val'], seed), M(y[idx['test']], pt, idx['test'], seed), dict(train_time_s=round(time.time() - t0, 1), best_iter=n_est if LAST else int(clf.best_iteration_)), a.tag)
    if 'cnn' in which:
        Xb = data.flat_bytes[:, :512].astype(np.float32)
        if a.mask_sni:
            for i in range(len(Xb)):
                s, e = data.flat_sni[i]
                if s >= 0 and s < 512:
                    Xb[i, s:min(e, 512)] = 0
        Xb = torch.from_numpy(Xb / 255.0).unsqueeze(1)
        for seed in seeds:
            model = PaperCNN(n_cls=NC)
            pv, pt, best_ep, tt = train_torch(model, Xb, y, idx, seed, a.epochs, 1e-3, 'adam'); save_preds(f'{prefix}_paper1dcnn_512B' + ('_masksni' if a.mask_sni else ''), seed, idx, y, pt)
            log_rec(f'{prefix}_paper1dcnn_512B' + ('_masksni' if a.mask_sni else ''), dict(model='paper_1dcnn', fc=256, input='flat512', mask_sni=a.mask_sni, epochs=a.epochs, dataset=a.dataset, target=a.target, split=a.split), seed, split_meta,
                    M(y[idx['val']], pv, idx['val'], seed), M(y[idx['test']], pt, idx['test'], seed), dict(train_time_s=tt, best_epoch=best_ep, params=sum(p.numel() for p in model.parameters())), a.tag)
    if 'dfmeta' in which:
        Xm = torch.from_numpy(meta_features(data, 64)).permute(0, 2, 1).contiguous()  # [N, 4, 64], same transform as the metadata attacker
        for seed in seeds:
            torch.manual_seed(seed); model = DFMetaCNN(n_cls=NC)  # official DF: Adamax 2e-3, batch 128, 30 epochs
            pv, pt, best_ep, tt = train_torch(model, Xm, y, idx, seed, 30, 2e-3, 'adamax', bs=128); save_preds(f'{prefix}_dfmeta64', seed, idx, y, pt)
            log_rec(f'{prefix}_dfmeta64', dict(model='df_1dcnn_meta', version='faithful', input='meta64x4', epochs=30, batch=128, dataset=a.dataset, target=a.target, split=a.split), seed, split_meta,
                    M(y[idx['val']], pv, idx['val'], seed), M(y[idx['test']], pt, idx['test'], seed), dict(train_time_s=tt, best_epoch=best_ep, params=sum(p.numel() for p in model.parameters())), a.tag)


if __name__ == '__main__':
    main()
