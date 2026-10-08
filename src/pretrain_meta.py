"""Self-supervised pre-training of a per-packet metadata encoder (masked packet modelling).
Corpus: all biflows of the train+val sessions of the given datasets (labels unused; test sessions excluded).
Model : Linear(4->d) token embedding + learned positions + Transformer encoder (n_layers, d, heads) ; reconstruction head Linear(d->4).
Task  : mask a fraction of valid packets (replace by a learned [MASK] token), regress their 4 standardised channels (MSE on masked positions).
Output: weights/<name>.pt with encoder state, normalisation stats and config; used by src/train_meta_ssl.py.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_meta import features


class MetaEncoder(nn.Module):
    def __init__(self, d=64, n_layers=4, n_heads=4, ff=128, L=64, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(4, d); self.pos = nn.Parameter(torch.zeros(1, L, d)); self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers); self.norm = nn.LayerNorm(d); self.d = d
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, valid, mask=None):  # x [B,L,4], valid [B,L] bool, mask [B,L] bool (positions to hide)
        h = self.embed(x)
        if mask is not None: h = torch.where(mask[..., None], self.mask_token.expand(h.shape[0], h.shape[1], -1), h)
        h = self.enc(h + self.pos, src_key_padding_mask=~valid)
        return self.norm(h)  # [B,L,d]

    def pool(self, h, valid):
        v = valid[..., None].float(); mean = (h * v).sum(1) / v.sum(1).clamp(min=1)
        mx = h.masked_fill(~valid[..., None], -1e4).amax(1)
        return torch.cat([mean, mx], 1)  # [B, 2d]


def load_corpus(datasets, first_k, exclude=None, target_frac=1.0, frac_seed=0):
    """exclude: {dataset: split json}; the test sessions of that split are removed from the dataset's corpus, so an
    evaluation split whose test sessions differ from the standard split's never sees its test traffic in pre-training.
    Tokens 'genai_bg' / 'ccma_bg' add the background flows (every other package and system service captured in the same
    sessions, no app filtering) of the train+val sessions. target_frac < 1 keeps a random fraction of the target-app flows
    of 'genai' / 'ccma' (the observer sees less traffic of the apps it monitors)."""
    Xs, Vs = [], []; exclude = exclude or {}
    for ds in datasets:
        if ds.endswith('_bg'):
            base = ds[:-3]; z = np.load(os.path.join(ROOT, 'data', 'derived', f'{base}_background.npz'), allow_pickle=True)
            data = GenAIData(ROOT, prefix=base)
            sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', 'session_split_s2026.json' if base == 'genai' else 'ccma_session_split_s2026.json')))
            keep_files = set(sp['train'] + sp['val'])
            if base in exclude: keep_files -= set(json.load(open(os.path.join(ROOT, exclude[base])))['test'])
            keep_sids = np.array([data.file2sid[f] for f in sorted(keep_files)]); idx = np.where(np.isin(z['session_id'], keep_sids))[0]
            m = z['meta'][idx, :first_k]; x = np.zeros_like(m, dtype=np.float32)
            x[..., 0] = m[..., 0]; x[..., 1] = np.log1p(m[..., 1]); x[..., 2] = np.log1p(m[..., 2]); x[..., 3] = np.log1p(m[..., 3] * 1000.0)
            V = (np.arange(first_k)[None, :] < np.minimum(z['meta_len'][idx], first_k)[:, None]); x[~V] = 0.0
            Xs.append(x); Vs.append(V); print(f'{ds}: {len(idx)} background flows from {len(keep_sids)} train+val sessions', flush=True); continue
        if ds == 'm2019':  # unlabelled corpus from scripts/build_m2019_meta.py (no test sessions of the evaluation datasets involved)
            d = np.load(os.path.join(ROOT, 'data', 'derived', 'm2019_meta.npz'))
            m = d['meta'][:, :first_k]; x = np.zeros_like(m); x[..., 0] = m[..., 0]; x[..., 1] = np.log1p(m[..., 1]); x[..., 2] = np.log1p(m[..., 2]); x[..., 3] = np.log1p(m[..., 3] * 1000.0)
            V = (np.arange(first_k)[None, :] < np.minimum(d['meta_len'], first_k)[:, None]); x[~V] = 0.0
            Xs.append(x.astype(np.float32)); Vs.append(V); print(f'm2019: {len(x)} flows from {len(np.unique(d["session_id"]))} sessions', flush=True); continue
        data = GenAIData(ROOT, prefix=ds)
        sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', 'session_split_s2026.json' if ds == 'genai' else 'ccma_session_split_s2026.json')))
        keep_files = set(sp['train'] + sp['val'])
        if ds in exclude:
            drop = set(json.load(open(os.path.join(ROOT, exclude[ds])))['test']); print(f'{ds}: excluding {len(keep_files & drop)} test sessions of {exclude[ds]}', flush=True); keep_files -= drop
        keep_sids = np.array([data.file2sid[f] for f in sorted(keep_files)])
        idx = np.where(np.isin(data.session_id, keep_sids))[0]  # labels unused; includes flows with y_joint<0 (controlled) only if in those sessions
        if target_frac < 1.0:
            idx = np.sort(np.random.RandomState(frac_seed).choice(idx, int(round(target_frac * len(idx))), replace=False))
        X = features(data, first_k)[idx]; V = (np.arange(first_k)[None, :] < np.minimum(data.meta_len[idx], first_k)[:, None])
        Xs.append(X); Vs.append(V); print(f'{ds}: {len(idx)} flows from {len(keep_sids)} train+val sessions', flush=True)
    return np.concatenate(Xs), np.concatenate(Vs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='genai,ccma'); p.add_argument('--first_k', type=int, default=64)
    p.add_argument('--d', type=int, default=64); p.add_argument('--layers', type=int, default=4); p.add_argument('--heads', type=int, default=4); p.add_argument('--ff', type=int, default=128)
    p.add_argument('--mask_ratio', type=float, default=0.4); p.add_argument('--epochs', type=int, default=100); p.add_argument('--bs', type=int, default=256); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--name', default='meta_ssl_mpm')
    p.add_argument('--exclude', default='', help='dataset:split.json[,dataset:split.json] whose test sessions are removed from the corpus')
    p.add_argument('--target_frac', type=float, default=1.0, help='fraction of the target-app flows kept in the corpus')
    p.add_argument('--steps_like', type=int, default=0, help='if > 0: set the epochs so that the number of optimiser steps equals '
                                                              'that of --epochs over a corpus of this many flows (same compute for every corpus)')
    a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
    X, V = load_corpus(a.datasets.split(','), a.first_k, dict(e.split(':', 1) for e in a.exclude.split(',') if e), a.target_frac)
    if a.steps_like:
        ref_steps = a.epochs * int(np.ceil(a.steps_like / a.bs)); a.epochs = max(1, int(round(ref_steps / int(np.ceil(len(X) / a.bs)))))
        print(f'steps_like={a.steps_like}: {ref_steps} reference steps -> {a.epochs} epochs over {len(X)} flows', flush=True)
    mu = (X * V[..., None]).sum((0, 1)) / V.sum(); sd = np.sqrt((((X - mu) ** 2) * V[..., None]).sum((0, 1)) / V.sum()) + 1e-6
    Xn = ((X - mu) / sd * V[..., None]).astype(np.float32)
    Xt = torch.from_numpy(Xn); Vt = torch.from_numpy(V)
    model = MetaEncoder(a.d, a.layers, a.heads, a.ff, a.first_k).to(dev); head = nn.Linear(a.d, 4).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * int(np.ceil(len(Xt) / a.bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.RandomState(a.seed); t0 = time.time(); N = len(Xt)
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(N); tot = 0.0; cnt = 0
        for b in range(0, N, a.bs):
            bi = perm[b:b + a.bs]; x = Xt[bi].to(dev); v = Vt[bi].to(dev)
            m = (torch.rand(x.shape[:2], device=dev) < a.mask_ratio) & v
            m[:, 0] = False  # keep the first packet visible
            h = model(x, v, m); pred = head(h)
            loss = ((pred - x) ** 2)[m].mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * m.sum().item(); cnt += m.sum().item()
        if ep % 10 == 0 or ep == a.epochs - 1: print(f'ep {ep:3d} masked-MSE {tot / max(cnt, 1):.4f}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(os.path.join(ROOT, 'weights'), exist_ok=True)
    torch.save(dict(encoder=model.state_dict(), cfg=dict(d=a.d, layers=a.layers, heads=a.heads, ff=a.ff, L=a.first_k), mu=mu, sd=sd, args=vars(a), n_flows=int(N)),
               os.path.join(ROOT, 'weights', f'{a.name}.pt'))
    print('saved', f'weights/{a.name}.pt', 'flows', N)


if __name__ == '__main__':
    main()
