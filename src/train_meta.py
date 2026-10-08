"""Patch-Transformer reference observer on per-packet metadata, and the input pipeline shared by every metadata observer.

Model   : a single-granularity patch Transformer in the style of Medformer (src/carriers/medformer_compat.py), without
          augmentations.
Input   : features() turns the first k packets of a connection into (direction, log IP bytes, log payload bytes,
          log inter-arrival time in ms); apply_defense() and scripts/wf_defenses.py shape the traffic first.
Settings: --split (session / temporal / unseen-phone / K-session json), --first_k, --defense, --defense_train.
Normalisation: per-channel mean and standard deviation fitted on the training split only.
Checkpoint rule and record format as in src/train_yatc.py.
"""
import argparse, hashlib, json, os, sys, time, copy
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds, set_seed
from carriers.medformer_compat import MetaCarrier, medformer_configs


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default=''); p.add_argument('--tag', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', default='configs/splits/session_split_s2026.json')
    p.add_argument('--dataset', default='genai', choices=['genai', 'ccma'])
    p.add_argument('--target', default='joint', choices=['joint', 'app'])
    p.add_argument('--first_k', type=int, default=64, help='use only the first k packets (<=64)')
    p.add_argument('--epochs', type=int, default=60); p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--patch_len_list', default='8'); p.add_argument('--no_inter', type=int, default=1)
    p.add_argument('--d_model', type=int, default=128); p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--e_layers', type=int, default=6); p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1); p.add_argument('--augmentations', default='none')
    p.add_argument('--class_balanced', type=int, default=0, help='inverse-frequency class weights in CE')
    p.add_argument('--defense', default='none', choices=['none', 'pad256', 'jitter20', 'pad256_jitter20', 'front', 'tamaraw', 'ech', 'ech_full'])
    p.add_argument('--defense_train', type=int, default=0, help='1 = adaptive attacker (defense applied to training data too); 0 = only val/test are defended')
    p.add_argument('--adv_train', type=int, default=0, help='PGD adversarial training (padding<=adv_bytes, delay<=adv_ms per packet) on half of each batch')
    p.add_argument('--adv_bytes', type=float, default=256); p.add_argument('--adv_ms', type=float, default=50); p.add_argument('--adv_steps', type=int, default=5)
    p.add_argument('--out_dir', default='results/runs')
    return p.parse_args()


def apply_defense(m, kind, seed):
    """m: [N,L,4] raw (dir, IP bytes, payload bytes, iat seconds). Padding: payload padded up to a multiple of 256 B
    (capped at 1460, IP length grows by the same amount); jitter: an Exp(mean 20 ms) delay added to every packet's IAT."""
    m = m.copy(); rng = np.random.RandomState(10_000 + seed); has = m[..., 2] > 0
    if 'pad256' in kind:
        padded = np.minimum(np.ceil(m[..., 2] / 256.0) * 256.0, 1460.0); delta = np.where(has, padded - m[..., 2], 0.0)
        m[..., 2] += delta; m[..., 1] += delta
    if 'jitter20' in kind:
        m[..., 3] += np.where(m[..., 1] > 0, rng.exponential(0.020, size=m.shape[:2]), 0.0)
    return m


NAMED_DEFENSES = ('front', 'tamaraw', 'ech', 'ech_full')  # published defences, implemented in scripts/wf_defenses.py
DEFENSE_OVERHEAD = {}                   # filled by features() so callers can report bandwidth/latency cost


def features(data, first_k, defense='none', defend_idx=None, seed=0):
    m = data.meta[:, :first_k].copy()                # dir(+1/-1), IP bytes, payload bytes, iat
    lens = np.minimum(data.meta_len, first_k).copy()
    if defense in NAMED_DEFENSES:  # these insert packets, so the per-flow length changes too
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.join(ROOT, 'scripts'))
        from wf_defenses import apply_named
        d, dl, ov = apply_named(defense, m, lens, seed); DEFENSE_OVERHEAD[defense] = ov
        if defend_idx is None: m, lens = d, dl
        else: m[defend_idx] = d[defend_idx]; lens[defend_idx] = dl[defend_idx]
    elif defense != 'none':
        d = apply_defense(m, defense, seed)
        if defend_idx is None: m = d
        else: m[defend_idx] = d[defend_idx]
    x = np.zeros_like(m)
    x[..., 0] = m[..., 0]; x[..., 1] = np.log1p(m[..., 1]); x[..., 2] = np.log1p(m[..., 2]); x[..., 3] = np.log1p(m[..., 3] * 1000.0)
    valid = np.arange(first_k)[None, :] < lens[:, None]
    x[~valid] = 0.0
    return x.astype(np.float32)


def build_model(args):
    cfg = medformer_configs(seq_len=args.first_k, enc_in=4, num_class=args.n_classes, patch_len_list=args.patch_len_list, d_model=args.d_model, d_ff=args.d_ff,
                            e_layers=args.e_layers, n_heads=args.n_heads, dropout=args.dropout, augmentations=args.augmentations, no_inter_attn=bool(args.no_inter))
    return MetaCarrier(cfg)


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval(); out = []
    for b in range(0, len(X), bs):
        out.append(model(X[b:b + bs].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    args = get_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ('run_name', 'tag', 'out_dir', 'n_classes')}; cfg['carrier'] = 'medformer_meta'
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    run_id = args.run_name or f'meta_{cfg_hash}_s{args.seed}'
    run_dir = os.path.join(ROOT, args.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== run {run_id} cfg_hash={cfg_hash} {json.dumps(cfg)}')
    set_seed(args.seed); device = torch.device('cuda')
    data = GenAIData(ROOT, prefix=args.dataset); idx, split_meta = data.split_indices(os.path.join(ROOT, args.split))
    n_act = data.n_act if args.target == 'joint' else 1
    args.n_classes = data.n_joint if args.target == 'joint' else data.n_app; NC = args.n_classes
    defend_idx = None if (args.defense == 'none' or args.defense_train) else np.concatenate([idx['val'], idx['test']])
    X = features(data, args.first_k, args.defense, defend_idx, args.seed); y = data.y_joint if args.target == 'joint' else data.y_app
    mu = X[idx['train']].reshape(-1, 4).mean(0); sd = X[idx['train']].reshape(-1, 4).std(0) + 1e-6
    X = torch.from_numpy((X - mu) / sd)
    tr = idx['train']
    if args.adv_train:
        sys.path.insert(0, os.path.join(ROOT, 'scripts')); from adv_eval import pgd, raw_to_feat
        raw_all = torch.from_numpy(data.meta[:, :args.first_k].astype(np.float32)); mu_t = torch.tensor(mu, device=device, dtype=torch.float32); sd_t = torch.tensor(sd, device=device, dtype=torch.float32)
    log(f'split hash={split_meta["hash"]} n_train={len(tr)} n_val={len(idx["val"])} n_test={len(idx["test"])} first_k={args.first_k}')
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters()); log(f'params={n_params} patch_len_list={args.patch_len_list} no_inter={args.no_inter}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)  # Medformer default: Adam lr 1e-4
    if args.class_balanced:
        cnt = np.bincount(y[tr], minlength=NC).astype(np.float32); w = torch.tensor(cnt.sum() / (NC * np.maximum(cnt, 1)), device=device)
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w)
    yv = y[idx['val']]; best = dict(macro_f1=-1, epoch=-1); best_state = None; hist = []; rng = np.random.RandomState(args.seed)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = rng.permutation(tr); tl = 0.0; tn = 0
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            xb, yb = X[bi].to(device), torch.from_numpy(y[bi]).to(device)
            if args.adv_train and len(bi) > 1:  # replace half of the batch with PGD-perturbed traffic (defender-side padding/delay)
                h = len(bi) // 2; model.eval()
                r_adv, _, _ = pgd(lambda f, v: model(f), raw_all[bi[:h]], torch.from_numpy(y[bi[:h]]), mu_t, sd_t, args.first_k, args.adv_bytes, args.adv_ms, args.adv_steps, device)
                model.train(); xb = torch.cat([raw_to_feat(r_adv, mu_t, sd_t, args.first_k).detach(), xb[h:]], 0)
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item() * len(bi); tn += len(bi)
        pv = predict(model, X[idx['val']], device)
        f1 = float(f1_score(yv, pv, average='macro')); acc = float(accuracy_score(yv, pv))
        hist.append(dict(epoch=ep, train_loss=tl / max(tn, 1), val_macro_f1=f1, val_acc=acc))
        log(f'ep {ep:3d} loss {tl / max(tn, 1):.4f} val_f1 {f1:.4f} val_acc {acc:.4f}')
        if (ep == args.epochs - 1) if os.environ.get('CKPT_SELECT', 'val') == 'last' else (f1 > best['macro_f1']):  # CKPT_SELECT=last: final epoch of the fixed schedule
            best = dict(macro_f1=f1, epoch=ep); best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t0; peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20
    model.load_state_dict(best_state); torch.save({'model': best_state, 'args': vars(args), 'best': best, 'norm': (mu, sd)}, os.path.join(run_dir, 'best.pt'))
    m2 = build_model(args); m2.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), weights_only=False)['model']); m2.to(device)
    pv = predict(model, X[idx['val']], device); assert (pv == predict(m2, X[idx['val']], device)).mean() > 0.99
    val_m = metrics_from_preds(yv, pv, sessions=data.session_id[idx['val']], seed=args.seed, n_classes=NC, n_act=n_act)
    np.savez(os.path.join(run_dir, 'val_preds.npz'), y=yv, pred=pv, idx=idx['val'])
    xs = X[:64].to(device); model.eval()
    with torch.no_grad():
        for _ in range(5): model(xs)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(30): model(xs)
        torch.cuda.synchronize(); lat64 = (time.perf_counter() - t) / 30 * 1000 / 64
    rec = dict(run_id=run_id, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=args.tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=split_meta['hash'], seed=args.seed,
               best_epoch=best['epoch'], epochs=args.epochs, n_train=int(len(tr)), n_val=int(len(idx['val'])), params=n_params, trainable_params=n_params,
               train_time_s=round(train_time, 1), peak_mem_mb=round(peak_mem, 1), latency_ms_per_sample_b64=round(lat64, 4), val=val_m, hist=hist)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
    log(f'BEST epoch {best["epoch"]} val macro_f1 {val_m["macro_f1"]:.4f} ci95 {val_m["macro_f1_ci95_sessions"]} app_f1 {val_m["app_macro_f1"]:.4f} act_f1 {val_m["act_macro_f1"]:.4f} | params {n_params} time {train_time:.0f}s mem {peak_mem:.0f}MB lat {lat64:.3f}ms')
    yt = y[idx['test']]; pt = predict(model, X[idx['test']], device)
    test_m = metrics_from_preds(yt, pt, sessions=data.session_id[idx['test']], seed=args.seed, n_classes=NC, n_act=n_act)
    np.savez(os.path.join(run_dir, 'test_preds.npz'), y=yt, pred=pt, idx=idx['test'])
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=run_id, cfg_hash=cfg_hash, seed=args.seed, tag=args.tag, split_hash=split_meta['hash'], best_epoch=best['epoch'], test=test_m)) + '\n')
    json.dump(rec, open(os.path.join(run_dir, 'record.json'), 'w'), indent=1); logf.close()


if __name__ == '__main__':
    main()
