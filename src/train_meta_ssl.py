"""Few-shot fine-tuning of the self-supervised metadata encoder (src/pretrain_meta.py) on K-session splits.
Modes: --mode ft (fine-tune encoder + head), --mode probe (frozen encoder, linear head), --mode export (write pooled embeddings
for every flow to results/emb/<run>.npz for a downstream LightGBM). --init '' trains the same encoder from scratch (control).
Same data, splits and record format as src/train_meta.py. Checkpoint rule: --select last keeps the final epoch of the fixed
schedule (the protocol of the paper), --select val the epoch with the best validation macro-F1; the default is the value of
the environment variable CKPT_SELECT, else val."""
import argparse, hashlib, json, os, sys, time, copy
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds, set_seed
from train_meta import features
from pretrain_meta import MetaEncoder


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default=''); p.add_argument('--tag', default='ssl_fewshot')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--split', default='configs/splits/session_split_s2026.json')
    p.add_argument('--dataset', default='genai', choices=['genai', 'ccma']); p.add_argument('--target', default='joint', choices=['joint', 'app'])
    p.add_argument('--first_k', type=int, default=64); p.add_argument('--init', default='weights/meta_ssl_mpm.pt')
    p.add_argument('--mode', default='ft', choices=['ft', 'probe', 'export']); p.add_argument('--ft_run', default='', help='export: run dir of a fine-tuned classifier whose encoder to use')
    p.add_argument('--epochs', type=int, default=60); p.add_argument('--bs', type=int, default=64); p.add_argument('--lr', type=float, default=5e-4); p.add_argument('--lr_enc', type=float, default=1e-4)
    p.add_argument('--drop_head', type=float, default=0.3); p.add_argument('--out_dir', default='results/runs')
    p.add_argument('--channels', default='all', choices=['all', 'size', 'timing'], help='all | size (direction+sizes) | timing (direction+IAT)')
    p.add_argument('--defense', default='none', choices=['none', 'pad256', 'jitter20', 'pad256_jitter20', 'front', 'tamaraw', 'ech', 'ech_full']); p.add_argument('--defense_train', type=int, default=0)
    p.add_argument('--adv_train', type=int, default=0); p.add_argument('--adv_bytes', type=float, default=256); p.add_argument('--adv_ms', type=float, default=50); p.add_argument('--adv_steps', type=int, default=5)
    p.add_argument('--arch', default='64,4,4,128', help='scratch encoder d,layers,heads,ff (used only when --init is empty)')
    p.add_argument('--norm_ckpt', default='', help='scratch encoder only: standardise inputs with the mu/sd stored in this pre-training '
                                                    'checkpoint (label-free corpus statistics without the test sessions), exactly as the pre-trained attacker does')
    p.add_argument('--select', default=os.environ.get('CKPT_SELECT', 'val'), choices=['val', 'last'],
                   help='val: keep the epoch with the best validation macro-F1 (uses labelled validation sessions); '
                        'last: keep the final epoch of the fixed schedule, so no labels beyond the K training sessions are used')
    return p.parse_args()


class Classifier(nn.Module):
    def __init__(self, enc, n_cls, drop):
        super().__init__(); self.enc = enc; self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * enc.d, n_cls))

    def forward(self, x, v):
        return self.head(self.enc.pool(self.enc(x, v), v))


def predict(model, X, V, device, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs): out.append(model(X[b:b + bs].to(device), V[b:b + bs].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    args = get_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ('run_name', 'tag', 'out_dir')}; cfg['carrier'] = 'meta_ssl'
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    run_id = args.run_name or f'ssl_{cfg_hash}_s{args.seed}'; run_dir = os.path.join(ROOT, args.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')

    def log(s): print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== run {run_id} cfg_hash={cfg_hash} {json.dumps(cfg)}')
    set_seed(args.seed); device = torch.device('cuda')
    data = GenAIData(ROOT, prefix=args.dataset); idx, split_meta = data.split_indices(os.path.join(ROOT, args.split))
    n_act = data.n_act if args.target == 'joint' else 1; NC = data.n_joint if args.target == 'joint' else data.n_app
    y = data.y_joint if args.target == 'joint' else data.y_app
    defend_idx = None if (args.defense == 'none' or args.defense_train) else np.concatenate([idx['val'], idx['test']])
    Xr = features(data, args.first_k, args.defense, defend_idx, args.seed)
    V = Xr[..., 0] != 0.0  # validity after shaping: FRONT and Tamaraw insert packets, so the per-flow length changes
    if args.init:
        ck = torch.load(os.path.join(ROOT, args.init), weights_only=False); c = ck['cfg']; mu, sd = ck['mu'], ck['sd']
        enc = MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L']); enc.load_state_dict(ck['encoder']); log(f'init from {args.init} ({ck["n_flows"]} flows)')
    else:
        if 'sslscratchXL' in run_id:  # the 'no pre-training' control must be the same 2.4M encoder as the pre-trained attacker
            assert args.arch == '192,8,8,384', f'{run_id}: scratch control needs --arch 192,8,8,384, got {args.arch}'
        if args.norm_ckpt:
            ckn = torch.load(os.path.join(ROOT, args.norm_ckpt), weights_only=False); mu, sd = ckn['mu'], ckn['sd']; log(f'standardisation from {args.norm_ckpt}')
        else:  # label-free statistics of every flow outside the test sessions (never test flows)
            nt = np.ones(len(Xr), bool); nt[idx['test']] = False; Xn, Vn = Xr[nt], V[nt]
            mu = (Xn * Vn[..., None]).sum((0, 1)) / Vn.sum(); sd = np.sqrt((((Xn - mu) ** 2) * Vn[..., None]).sum((0, 1)) / Vn.sum()) + 1e-6
        d_, l_, h_, f_ = [int(t) for t in args.arch.split(',')]; enc = MetaEncoder(d_, l_, h_, f_, args.first_k); log(f'encoder from scratch arch={args.arch}')
    X = torch.from_numpy(((Xr - mu) / sd * V[..., None]).astype(np.float32)); Vt = torch.from_numpy(V)
    if args.channels != 'all':  # channel ablation: the attacker observes direction plus sizes only, or plus timing only;
        drop = [3] if args.channels == 'size' else [1, 2]  # removed channels are set to their corpus mean (0 after standardisation)
        X[..., drop] = 0.0; log(f'channels={args.channels}: dropped {drop}')
    if args.mode == 'export':
        if args.ft_run:  # use the encoder of a fine-tuned classifier instead of the raw pre-trained one
            ck2 = torch.load(os.path.join(ROOT, 'results', 'runs', args.ft_run, 'best.pt'), weights_only=False)
            tmp = Classifier(enc, NC, 0.0); tmp.load_state_dict(ck2['model']); enc = tmp.enc; log(f'encoder from fine-tuned run {args.ft_run}')
        enc.to(device).eval(); embs = []
        with torch.no_grad():
            for b in range(0, len(X), 512):
                x, v = X[b:b + 512].to(device), Vt[b:b + 512].to(device); embs.append(enc.pool(enc(x, v), v).cpu().numpy())
        os.makedirs(os.path.join(ROOT, 'results', 'emb'), exist_ok=True); out = os.path.join(ROOT, 'results', 'emb', f'{run_id}.npz')
        np.savez(out, emb=np.concatenate(embs), init=args.init, dataset=args.dataset); log(f'exported embeddings to {out}'); return
    model = Classifier(enc, NC, args.drop_head).to(device)
    if args.adv_train:
        sys.path.insert(0, os.path.join(ROOT, 'scripts')); from adv_eval import pgd, raw_to_feat
        raw_all = torch.from_numpy(data.meta[:, :args.first_k].astype(np.float32)); mu_t = torch.tensor(mu, device=device, dtype=torch.float32); sd_t = torch.tensor(sd, device=device, dtype=torch.float32)
        adv_fn = lambda f, v: model(f * v.unsqueeze(-1).float(), v)
    if args.mode == 'probe':
        for p_ in model.enc.parameters(): p_.requires_grad = False
        opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': args.lr_enc}, {'params': model.head.parameters(), 'lr': args.lr}], weight_decay=0.01)
    n_params = sum(p_.numel() for p_ in model.parameters()); log(f'params={n_params} mode={args.mode}')
    crit = nn.CrossEntropyLoss(); tr = idx['train']; yv = y[idx['val']]; best = dict(macro_f1=-1, epoch=-1); best_state = None; rng = np.random.RandomState(args.seed); val_curve = []
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if args.mode == 'probe': model.enc.eval()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            if len(bi) < 2: continue
            xb, vb = X[bi].to(device), Vt[bi].to(device)
            if args.adv_train and len(bi) > 1:  # replace half of the batch by PGD-shaped traffic (defender-side padding/delay)
                h = len(bi) // 2; model.eval()
                r_adv, _, _ = pgd(adv_fn, raw_all[bi[:h]], torch.from_numpy(y[bi[:h]]), mu_t, sd_t, args.first_k, args.adv_bytes, args.adv_ms, args.adv_steps, device)
                model.train(); xb = torch.cat([(raw_to_feat(r_adv, mu_t, sd_t, args.first_k) * vb[:h].unsqueeze(-1).float()).detach(), xb[h:]], 0)
            loss = crit(model(xb, vb), torch.from_numpy(y[bi]).to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pv = predict(model, X[idx['val']], Vt[idx['val']], device); f1 = float(f1_score(yv, pv, average='macro')); val_curve.append(round(f1, 4))
        if args.select == 'last':
            if ep == args.epochs - 1: best = dict(macro_f1=f1, epoch=ep); best_state = copy.deepcopy(model.state_dict())
        elif f1 > best['macro_f1']: best = dict(macro_f1=f1, epoch=ep); best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t0; model.load_state_dict(best_state); torch.save({'model': best_state, 'args': vars(args), 'best': best, 'norm': (mu, sd)}, os.path.join(run_dir, 'best.pt'))
    pv = predict(model, X[idx['val']], Vt[idx['val']], device); val_m = metrics_from_preds(yv, pv, sessions=data.session_id[idx['val']], seed=args.seed, n_classes=NC, n_act=n_act)
    rec = dict(run_id=run_id, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=args.tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=split_meta['hash'], seed=args.seed, best_epoch=best['epoch'],
               epochs=args.epochs, n_train=int(len(tr)), params=n_params, train_time_s=round(train_time, 1), val=val_m, select=args.select, val_curve=val_curve)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec) + '\n')
    log(f'BEST epoch {best["epoch"]} val macro_f1 {val_m["macro_f1"]:.4f} ci95 {val_m["macro_f1_ci95_sessions"]} | params {n_params} time {train_time:.0f}s')
    yt = y[idx['test']]; pt = predict(model, X[idx['test']], Vt[idx['test']], device)
    test_m = metrics_from_preds(yt, pt, sessions=data.session_id[idx['test']], seed=args.seed, n_classes=NC, n_act=n_act)
    np.savez(os.path.join(run_dir, 'test_preds.npz'), y=yt, pred=pt, idx=idx['test'])
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=run_id, cfg_hash=cfg_hash, seed=args.seed, tag=args.tag, split_hash=split_meta['hash'], best_epoch=best['epoch'], test=test_m)) + '\n')
    json.dump(rec, open(os.path.join(run_dir, 'record.json'), 'w'), indent=1); logf.close()


if __name__ == '__main__':
    main()
