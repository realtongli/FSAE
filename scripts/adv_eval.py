"""Adversarial traffic perturbation against the metadata attacker (defender's view).
The defender may only ADD payload padding (<= budget_bytes per packet, capped at 1460 total) and ADD delay
(<= budget_ms per packet) to real packets; direction and packet count are unchanged. Perturbations are found with
PGD in raw space through the attacker's differentiable feature transform (log1p + train-set standardisation).
Modes: whitebox (gradients of the evaluated attacker), transfer (gradients of a surrogate attacker checkpoint).
Usage: adv_eval.py --run <run_id> [--surrogate <run_id>] --budget_bytes 256 --budget_ms 50 --steps 20
Appends one JSON line per evaluation to results/adv_eval.jsonl (test split of the run's own split file).
"""
import argparse, json, os, sys, numpy as np, torch
from sklearn.metrics import f1_score
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_meta import build_model, apply_defense


def load_attacker(run_id, device, NC):
    """Returns (fn(feat, valid)->logits, mu, sd, args) for a train_meta run or a train_meta_ssl run (detected by 'init' in args)."""
    ck, args = load_run(run_id)
    if 'init' in ck['args']:  # pre-trained metadata encoder (src/train_meta_ssl.py)
        import train_meta_ssl
        from pretrain_meta import MetaEncoder
        if args.init:
            pre = torch.load(os.path.join(ROOT, args.init), weights_only=False); c = pre['cfg']; mu, sd = pre['mu'], pre['sd']
            enc = MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L'])
        else:
            d_, l_, h_, f_ = [int(t) for t in args.arch.split(',')]; enc = MetaEncoder(d_, l_, h_, f_, args.first_k)
            if 'norm' in ck: mu, sd = ck['norm']
            else:  # from-scratch checkpoint without a stored 'norm': recompute the dataset-wide standardisation used by train_meta_ssl
                from train_meta import features as _feat
                _d = GenAIData(ROOT, prefix=getattr(args, 'dataset', 'genai')); _X = _feat(_d, args.first_k); _V = (np.arange(args.first_k)[None, :] < np.minimum(_d.meta_len, args.first_k)[:, None])
                mu = (_X * _V[..., None]).sum((0, 1)) / _V.sum(); sd = np.sqrt((((_X - mu) ** 2) * _V[..., None]).sum((0, 1)) / _V.sum()) + 1e-6
        m = train_meta_ssl.Classifier(enc, NC, args.drop_head).to(device); m.load_state_dict(ck['model']); m.eval()
        fn = lambda f, v: m(f * v.unsqueeze(-1).float(), v)
    else:
        args.n_classes = NC; m = build_model(args).to(device); m.load_state_dict(ck['model']); m.eval(); mu, sd = ck['norm']
        fn = lambda f, v: m(f)
    return fn, torch.tensor(mu, device=device, dtype=torch.float32), torch.tensor(sd, device=device, dtype=torch.float32), args


def load_run(run_id):
    ck = torch.load(os.path.join(ROOT, 'results', 'runs', run_id, 'best.pt'), weights_only=False)
    a = argparse.Namespace(**ck['args']); a.n_classes = None
    return ck, a


def raw_to_feat(raw, mu, sd, first_k):
    # raw: [B,L,4] torch (dir, IP bytes, payload bytes, iat s)  -> standardised log features as in train_meta.features
    x = torch.stack([raw[..., 0], torch.log1p(raw[..., 1]), torch.log1p(raw[..., 2]), torch.log1p(raw[..., 3] * 1000.0)], -1)
    valid = (raw[..., 1] > 0).float().unsqueeze(-1); x = x * valid
    return (x - mu) / sd


def pgd(model_grad, raw, y, mu, sd, first_k, budget_bytes, budget_ms, steps, device):
    """Maximise the surrogate's loss; delta_size in [0,budget_bytes] (payload+IP both increase), delta_iat in [0,budget_ms/1000]."""
    raw = raw.to(device); y = y.to(device); has = (raw[..., 2] > 0).float(); real = (raw[..., 1] > 0).float()
    ds = torch.zeros_like(raw[..., 2], requires_grad=True); dt = torch.zeros_like(raw[..., 3], requires_grad=True)
    cap = torch.clamp(1460.0 - raw[..., 2], min=0.0)
    for _ in range(steps):
        r = raw.clone(); r[..., 2] = raw[..., 2] + ds * has; r[..., 1] = raw[..., 1] + ds * has; r[..., 3] = raw[..., 3] + dt * real
        loss = torch.nn.functional.cross_entropy(model_grad(raw_to_feat(r, mu, sd, first_k), raw[..., 1] > 0), y)
        g_s, g_t = torch.autograd.grad(loss, [ds, dt])
        with torch.no_grad():
            ds += (budget_bytes / 8.0) * g_s.sign(); dt += (budget_ms / 1000.0 / 8.0) * g_t.sign()
            ds.clamp_(0.0, budget_bytes); ds.copy_(torch.minimum(ds, cap)); dt.clamp_(0.0, budget_ms / 1000.0)
    with torch.no_grad():
        r = raw.clone(); r[..., 2] = raw[..., 2] + ds * has; r[..., 1] = raw[..., 1] + ds * has; r[..., 3] = raw[..., 3] + dt * real
    return r, float((ds * has).sum() / has.sum()), float((dt * real).sum() / real.sum() * 1000)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--surrogate', default='')
    ap.add_argument('--budget_bytes', type=float, default=256); ap.add_argument('--budget_ms', type=float, default=50); ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--tag', default='adv'); a = ap.parse_args()
    device = torch.device('cuda')
    ck, args = load_run(a.run); data = GenAIData(ROOT, prefix=args.dataset if hasattr(args, 'dataset') else 'genai')
    idx, meta = data.split_indices(os.path.join(ROOT, args.split)); target = getattr(args, 'target', 'joint')
    NC = data.n_joint if target == 'joint' else data.n_app; y_all = data.y_joint if target == 'joint' else data.y_app
    model, mu, sd, args = load_attacker(a.run, device, NC)
    if a.surrogate: sur, mu2, sd2, _ = load_attacker(a.surrogate, device, NC)
    te = idx['test']; raw_all = torch.from_numpy(data.meta[:, :args.first_k].astype(np.float32)); y = torch.from_numpy(y_all)
    preds_clean, preds_adv, ys = [], [], []; pad_stats, del_stats = [], []
    for b in range(0, len(te), 256):
        bi = te[b:b + 256]; raw = raw_all[bi]; yb = y[bi]; v = (raw[..., 1] > 0).to(device)
        with torch.no_grad(): preds_clean.append(model(raw_to_feat(raw.to(device), mu, sd, args.first_k), v).argmax(1).cpu().numpy())
        if a.surrogate: r_adv, pb, pm = pgd(sur, raw, yb, mu2, sd2, args.first_k, a.budget_bytes, a.budget_ms, a.steps, device)
        else: r_adv, pb, pm = pgd(model, raw, yb, mu, sd, args.first_k, a.budget_bytes, a.budget_ms, a.steps, device)
        with torch.no_grad(): preds_adv.append(model(raw_to_feat(r_adv, mu, sd, args.first_k), v).argmax(1).cpu().numpy())
        ys.append(yb.numpy()); pad_stats.append(pb); del_stats.append(pm)
    ys = np.concatenate(ys); pc = np.concatenate(preds_clean); pa = np.concatenate(preds_adv)
    rec = dict(run=a.run, surrogate=a.surrogate or 'whitebox', tag=a.tag, dataset=getattr(args, 'dataset', 'genai'), split=meta['hash'], seed=args.seed,
               budget_bytes=a.budget_bytes, budget_ms=a.budget_ms, steps=a.steps, test_f1_clean=float(f1_score(ys, pc, average='macro')),
               test_f1_adv=float(f1_score(ys, pa, average='macro')), mean_pad_bytes=float(np.mean(pad_stats)), mean_delay_ms=float(np.mean(del_stats)), n=int(len(ys)))
    with open(os.path.join(ROOT, 'results', 'adv_eval.jsonl'), 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
