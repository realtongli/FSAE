"""Website-fingerprinting-style few-shot attackers re-implemented on the 64-packet x 4-channel metadata sequence (same threat
model as the proposed attacker). All are adaptations: the originals operate on Tor cell direction sequences. The methods here
share one DF backbone with global pooling; src/train_wf_faithful.py holds the variants that keep each publication's own
architecture, augmentation and training schedule.
  netclr : NetCLR (Bahramali et al., CCS 2023) - SimCLR (NT-Xent, T=0.5) pre-training of a DF backbone on UNLABELLED flows with
           traffic augmentations (burst merge/split, packet drop, size and timing jitter, shift), then fine-tuning on the K sessions.
           --backbone transformer swaps the DF backbone for the proposed encoder (pretext-task ablation).
  tf     : Triplet Fingerprinting (Sirinam et al., CCS 2019) - triplet-loss feature extractor trained on a LABELLED disjoint campaign
           (the other dataset), then k-NN over the K labelled sessions of the target campaign.
  tiktok : Tik-Tok (Rahman et al., PETS 2020) - DF on the directional-timing sequence (direction x cumulative time), supervised on K sessions.
  cf     : Contrastive Fingerprinting adapted (Xie et al., WWW 2024) - supervised contrastive learning with augmentation on the K
           sessions' flows only (DF backbone), then k-NN classification; a variant without the pre-training on a disjoint
           labelled campaign and without the linear classifier of the published pipeline (cfpub).
  cfpub  : Contrastive Fingerprinting as published - SupCon pre-training (DF backbone + MLP projection head, inject/remove
           augmentation at 10% of the flow) on a LABELLED disjoint campaign (the other dataset, as for tf), then the frozen extractor
           (projection head dropped) and a linear classifier trained on the augmented K labelled sessions of the target campaign.
           No public code; re-implemented from the paper's description (Sec. 3, Table 1).
Records follow src/train_meta_ssl.py (tag wf_baselines); test predictions are saved for session-level aggregation."""
import argparse, hashlib, json, os, sys, time, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds, set_seed
from pretrain_meta import MetaEncoder

L = 64


# ---------------- features & augmentation on raw metadata ----------------
def raw_to_feat(m, n):  # m [L,4] raw (dir, IP, payload, iat s), n valid -> [L,4] log features (as train_meta.features)
    x = np.zeros((L, 4), np.float32); k = min(n, L)
    x[:k, 0] = m[:k, 0]; x[:k, 1] = np.log1p(m[:k, 1]); x[:k, 2] = np.log1p(m[:k, 2]); x[:k, 3] = np.log1p(m[:k, 3] * 1000.0)
    return x


def tiktok_feat(m, n):  # [L,1] direction x cumulative time (s), as in Tik-Tok's directional timing
    x = np.zeros((L, 1), np.float32); k = min(n, L); x[:k, 0] = m[:k, 0] * np.cumsum(m[:k, 3]); return x


class Augmentor:
    """Traffic augmentations in the spirit of NetCLR's burst-level augmentations, applied to raw (dir, IP, payload, iat) packets."""

    def __init__(self, rng): self.rng = rng

    def __call__(self, m, n):
        m = m[:n].copy(); r = self.rng
        if n > 4 and r.rand() < 0.5:  # shift: drop the first 1-3 packets
            k = r.randint(1, 4); m = m[k:]
        if len(m) > 4 and r.rand() < 0.5:  # drop 10% of packets (never the first)
            keep = np.ones(len(m), bool); keep[1:] = r.rand(len(m) - 1) > 0.1; m = m[keep]
        if len(m) > 4 and r.rand() < 0.5:  # merge adjacent same-direction packets (burst merge)
            out = [m[0]]; i = 1
            while i < len(m):
                if m[i, 0] == out[-1][0] and r.rand() < 0.2: out[-1] = out[-1].copy(); out[-1][1:3] += m[i, 1:3]; out[-1][3] += m[i, 3]
                else: out.append(m[i])
                i += 1
            m = np.array(out)
        if r.rand() < 0.5:  # size jitter on downstream packets
            f = r.uniform(0.8, 1.2, size=len(m)); dn = m[:, 0] < 0; m[dn, 2] = np.minimum(m[dn, 2] * f[dn], 1460); m[dn, 1] = m[dn, 2] + (m[dn, 1] - m[dn, 2])
        if r.rand() < 0.5:  # timing jitter
            m[:, 3] = m[:, 3] * np.exp(r.normal(0, 0.2, size=len(m)))
        if len(m) < L and r.rand() < 0.3:  # insert an ACK-like upstream packet
            j = r.randint(1, len(m) + 1); ack = np.array([1.0, 52.0, 0.0, 0.001], np.float32); m = np.insert(m, j, ack, axis=0)
        return m, len(m)


class CFAugmentor:
    """Contrastive Fingerprinting's augmentation: 'Injecting' inserts packets of random direction at random positions and 'Removing'
    deletes packets at random positions, each on 10% of the flow. CF works on direction-only cells; on our 4-channel packets an
    injected packet copies the sizes and gap of a random packet of the same flow."""

    def __init__(self, rng, pct=0.1): self.rng = rng; self.pct = pct

    def __call__(self, m, n):
        m = m[:n].copy(); r = self.rng; k = max(1, int(round(self.pct * n)))
        for _ in range(k):  # injecting
            src = m[r.randint(len(m))].copy(); src[0] = r.choice([-1.0, 1.0]); m = np.insert(m, r.randint(1, len(m) + 1), src, axis=0)
        if len(m) > k + 1:  # removing (never the first packet)
            drop = r.choice(np.arange(1, len(m)), size=k, replace=False); m = np.delete(m, drop, axis=0)
        m = m[:L]; return m, len(m)


def load_other_campaign(dataset):
    """Labelled training sessions of the other campaign (source of tf / cfpub pre-training)."""
    other = 'ccma' if dataset == 'genai' else 'genai'; d = GenAIData(ROOT, prefix=other)
    sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', 'session_split_s2026.json' if other == 'genai' else 'ccma_session_split_s2026.json')))
    keep = np.array([d.file2sid[f] for f in sp['train']]); ii = np.where(np.isin(d.session_id, keep) & (d.y_joint >= 0))[0]
    oy = (d.y_joint if other == 'genai' else d.y_app)[ii]; oraw = d.meta[ii, :L].astype(np.float32); olen = np.minimum(d.meta_len[ii], L)
    return other, oy, oraw, olen


# ---------------- backbones ----------------
class DFBackbone(nn.Module):
    """DF block structure (Sirinam et al. 2018) on [B, C, L]; global avg+max pooling -> 512-d."""

    def __init__(self, C=4):
        super().__init__(); chans = [C, 32, 64, 128, 256]; blocks = []
        for i in range(4):
            act = nn.ELU if i == 0 else nn.ReLU
            blocks += [nn.Conv1d(chans[i], chans[i + 1], 8, padding=4), nn.BatchNorm1d(chans[i + 1]), act(), nn.Conv1d(chans[i + 1], chans[i + 1], 8, padding=4), nn.BatchNorm1d(chans[i + 1]), act(), nn.MaxPool1d(2), nn.Dropout(0.1)]
        self.net = nn.Sequential(*blocks); self.d = 512

    def forward(self, x, v=None):  # x [B,L,C]
        h = self.net(x.transpose(1, 2)); return torch.cat([h.mean(-1), h.amax(-1)], 1)


class TrBackbone(nn.Module):
    def __init__(self, d=192, layers=8, heads=8, ff=384):
        super().__init__(); self.enc = MetaEncoder(d, layers, heads, ff, L); self.d = 2 * d

    def forward(self, x, v): return self.enc.pool(self.enc(x, v), v)


def nt_xent(z1, z2, T=0.5):
    z = F.normalize(torch.cat([z1, z2], 0), dim=1); sim = z @ z.t() / T; n = z1.shape[0]
    sim.fill_diagonal_(-1e9); tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def supcon(z, y, T=0.1):
    z = F.normalize(z, dim=1); sim = z @ z.t() / T; n = len(y); eye = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9); logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye; cnt = pos.sum(1).clamp(min=1)
    return -((logp * pos).sum(1) / cnt)[pos.sum(1) > 0].mean()


def triplet(z, y, margin=0.1):  # batch-hard triplet on cosine distance
    z = F.normalize(z, dim=1); d = 1 - z @ z.t(); same = y[:, None] == y[None, :]; eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
    hardest_pos = d.masked_fill(~same | eye, -1).max(1).values; hardest_neg = d.masked_fill(same, 9).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


# ---------------- main ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['netclr', 'tf', 'tiktok', 'cf', 'cfpub']); p.add_argument('--backbone', default='df', choices=['df', 'transformer'])
    p.add_argument('--run_name', default=''); p.add_argument('--tag', default='wf_baselines'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', required=True); p.add_argument('--dataset', default='genai', choices=['genai', 'ccma']); p.add_argument('--target', default='joint', choices=['joint', 'app'])
    p.add_argument('--pre_epochs', type=int, default=100); p.add_argument('--epochs', type=int, default=60); p.add_argument('--bs', type=int, default=64)
    p.add_argument('--pre_ckpt', default='', help='netclr / cfpub: reuse/save the pre-trained backbone at this path (weights/...)')
    p.add_argument('--knn', type=int, default=5); p.add_argument('--out_dir', default='results/runs')
    p.add_argument('--defense', default='none', choices=['none', 'pad256', 'jitter20', 'pad256_jitter20', 'front', 'tamaraw'])
    p.add_argument('--defense_train', type=int, default=0, help='1 = adaptive attacker (the defence is also applied to its training traffic)')
    return p.parse_args()


def encode(bb, X, V, device, bs=512):
    bb.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs): out.append(bb(X[b:b + bs].to(device), V[b:b + bs].to(device)).cpu())
    return torch.cat(out)


def knn_predict(Ztr, ytr, Z, k):
    Ztr = F.normalize(Ztr, dim=1); Z = F.normalize(Z, dim=1); sim = Z @ Ztr.t(); nn_idx = sim.topk(min(k, len(ytr)), dim=1).indices
    votes = torch.from_numpy(ytr)[nn_idx]; return torch.mode(votes, 1).values.numpy()


def main():
    a = get_args(); set_seed(a.seed); device = torch.device('cuda'); rng = np.random.RandomState(a.seed)
    cfg = {k: v for k, v in vars(a).items() if k not in ('run_name', 'tag', 'out_dir')}; cfg['carrier'] = 'wf_baseline'
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    run_id = a.run_name or f'{a.method}_{cfg_hash}_s{a.seed}'; run_dir = os.path.join(ROOT, a.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')

    def log(s): print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== run {run_id} {json.dumps(cfg)}')
    data = GenAIData(ROOT, prefix=a.dataset); idx, sm = data.split_indices(os.path.join(ROOT, a.split))
    n_act = data.n_act if a.target == 'joint' else 1; NC = data.n_joint if a.target == 'joint' else data.n_app; y = data.y_joint if a.target == 'joint' else data.y_app
    raw = data.meta[:, :L].astype(np.float32); nlen = np.minimum(data.meta_len, L)
    def shape(r_, n_):  # same shaping as for the other attackers (train_meta / wf_defenses)
        from train_meta import apply_defense, NAMED_DEFENSES
        if a.defense in NAMED_DEFENSES:
            from wf_defenses import apply_named
            return apply_named(a.defense, r_, n_, a.seed)
        return apply_defense(r_, a.defense, a.seed), n_, None

    if a.defense != 'none':
        sraw, slen, ov = shape(raw, nlen)
        if ov is not None: log(f'defence {a.defense} overhead {ov}')
        if a.defense_train: raw, nlen = sraw, slen
        else:
            di = np.concatenate([idx['val'], idx['test']]); raw = raw.copy(); nlen = nlen.copy(); raw[di] = sraw[di]; nlen[di] = slen[di]
    C = 1 if a.method == 'tiktok' else 4; featf = tiktok_feat if a.method == 'tiktok' else raw_to_feat
    X = np.stack([featf(raw[i], nlen[i]) for i in range(len(raw))]); V = (np.arange(L)[None, :] < nlen[:, None])
    mu = (X * V[..., None]).sum((0, 1)) / V.sum(); sd = np.sqrt((((X - mu) ** 2) * V[..., None]).sum((0, 1)) / V.sum()) + 1e-6
    norm = lambda x, v: ((x - mu) / sd * v[..., None]).astype(np.float32)
    Xn = torch.from_numpy(norm(X, V)); Vt = torch.from_numpy(V); aug = Augmentor(rng)

    def aug_batch(ids):  # two augmented views (features, valid) for the given flow indices
        xs, vs = [], []
        for i in ids:
            m, n = aug(raw[i], nlen[i]); v = np.arange(L) < n; xs.append(norm(featf(m, n), v)); vs.append(v)
        return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(vs)).to(device)

    bb = (TrBackbone() if a.backbone == 'transformer' else DFBackbone(C)).to(device); t0 = time.time(); tr = idx['train']
    # ---------- representation learning ----------
    if a.method == 'netclr':
        ck = os.path.join(ROOT, a.pre_ckpt) if a.pre_ckpt else ''
        if ck and os.path.exists(ck): bb.load_state_dict(torch.load(ck, weights_only=False)['backbone']); log(f'loaded SimCLR backbone {a.pre_ckpt}')
        else:  # SimCLR on the unlabelled corpus (train+val sessions of both campaigns, labels unused; test sessions excluded)
            corp = []
            for ds in ('genai', 'ccma'):
                d = GenAIData(ROOT, prefix=ds); sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', 'session_split_s2026.json' if ds == 'genai' else 'ccma_session_split_s2026.json')))
                keep = np.array([d.file2sid[f] for f in sp['train'] + sp['val']]); ii = np.where(np.isin(d.session_id, keep))[0]
                corp.append((d.meta[ii, :L].astype(np.float32), np.minimum(d.meta_len[ii], L)))
            craw = np.concatenate([c[0] for c in corp]); clen = np.concatenate([c[1] for c in corp]); log(f'SimCLR corpus {len(craw)} flows')
            proj = nn.Sequential(nn.Linear(bb.d, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
            opt = torch.optim.Adam(list(bb.parameters()) + list(proj.parameters()), lr=1e-3); bs = 256
            for ep in range(a.pre_epochs):
                bb.train(); perm = rng.permutation(len(craw)); tot = 0; cnt = 0
                for b in range(0, len(perm), bs):
                    bi = perm[b:b + bs]
                    if len(bi) < 8: continue
                    views = []
                    for _ in range(2):
                        xs, vs = [], []
                        for i in bi:
                            m, n = aug(craw[i], clen[i]); v = np.arange(L) < n; xs.append(norm(featf(m, n), v)); vs.append(v)
                        views.append((torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(vs)).to(device)))
                    loss = nt_xent(proj(bb(*views[0])), proj(bb(*views[1]))); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); cnt += 1
                if ep % 10 == 0 or ep == a.pre_epochs - 1: log(f'simclr ep {ep} loss {tot / max(cnt, 1):.4f} {time.time() - t0:.0f}s')
            if ck: torch.save({'backbone': bb.state_dict(), 'norm': (mu, sd)}, ck); log(f'saved {a.pre_ckpt}')
    elif a.method == 'tf':  # triplet extractor on the OTHER campaign's labelled training sessions
        other, oy, oraw, olen = load_other_campaign(a.dataset)
        oX =torch.from_numpy(norm(np.stack([featf(oraw[i], olen[i]) for i in range(len(oraw))]), (np.arange(L)[None, :] < olen[:, None]))); oV = torch.from_numpy(np.arange(L)[None, :] < olen[:, None])
        opt = torch.optim.Adam(bb.parameters(), lr=1e-3); log(f'triplet extractor on {other}: {len(oy)} flows, {len(np.unique(oy))} classes')
        for ep in range(a.pre_epochs):
            bb.train(); perm = rng.permutation(len(oy)); tot = 0; cnt = 0
            for b in range(0, len(perm), 128):
                bi = perm[b:b + 128]
                if len(bi) < 8: continue
                loss = triplet(bb(oX[bi].to(device), oV[bi].to(device)), torch.from_numpy(oy[bi]).to(device)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); cnt += 1
            if ep % 20 == 0 or ep == a.pre_epochs - 1: log(f'triplet ep {ep} loss {tot / max(cnt, 1):.4f}')
    elif a.method == 'cf':  # supervised contrastive with augmentation on the K sessions' flows
        opt = torch.optim.Adam(bb.parameters(), lr=1e-3)
        for ep in range(a.pre_epochs):
            bb.train(); perm = rng.permutation(tr); tot = 0; cnt = 0
            for b in range(0, len(perm), 128):
                bi = perm[b:b + 128]
                if len(bi) < 8: continue
                x1, v1 = aug_batch(bi); x2, v2 = aug_batch(bi); yb = torch.from_numpy(np.concatenate([y[bi], y[bi]])).to(device)
                loss = supcon(torch.cat([bb(x1, v1), bb(x2, v2)]), yb); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); cnt += 1
            if ep % 20 == 0 or ep == a.pre_epochs - 1: log(f'supcon ep {ep} loss {tot / max(cnt, 1):.4f}')
    elif a.method == 'cfpub':  # SupCon + inject/remove on the OTHER campaign's labelled training sessions, MLP projection head
        other, oy, oraw, olen = load_other_campaign(a.dataset)
        if a.defense != 'none' and a.defense_train:  # an adaptive attacker also shapes its pre-training traffic (as CF does for WTF-PAD)
            oraw, olen, _ = shape(oraw, olen)
        cfaug = CFAugmentor(rng); proj = nn.Sequential(nn.Linear(bb.d, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
        opt = torch.optim.Adam(list(bb.parameters()) + list(proj.parameters()), lr=1e-3); log(f'CF pre-training on {other}: {len(oy)} flows, {len(np.unique(oy))} classes')

        def cf_views(r_, n_, ids):
            xs, vs = [], []
            for i in ids:
                m, n = cfaug(r_[i], n_[i]); v = np.arange(L) < n; xs.append(norm(featf(m, n), v)); vs.append(v)
            return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(vs)).to(device)

        # pre-training depends only on (dataset, seed, shaping of the pre-training traffic), not on K or the split: cache it
        ck = os.path.join(ROOT, a.pre_ckpt) if a.pre_ckpt else ''
        cached = bool(ck) and os.path.exists(ck)
        if cached:
            st = torch.load(ck, weights_only=False); assert np.allclose(st['norm'][0], mu) and np.allclose(st['norm'][1], sd), 'normalisation mismatch'
            bb.load_state_dict(st['backbone']); log(f'loaded CF backbone {a.pre_ckpt}')
        for ep in range(0 if cached else a.pre_epochs):
            bb.train(); perm = rng.permutation(len(oy)); tot = 0; cnt = 0
            for b in range(0, len(perm), 128):
                bi = perm[b:b + 128]
                if len(bi) < 8: continue
                x1, v1 = cf_views(oraw, olen, bi); x2, v2 = cf_views(oraw, olen, bi); yb = torch.from_numpy(np.concatenate([oy[bi], oy[bi]])).to(device)
                loss = supcon(proj(torch.cat([bb(x1, v1), bb(x2, v2)])), yb); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); cnt += 1
            if ep % 20 == 0 or ep == a.pre_epochs - 1: log(f'cf supcon ep {ep} loss {tot / max(cnt, 1):.4f}')
        if ck and not cached: torch.save({'backbone': bb.state_dict(), 'norm': (mu, sd)}, ck); log(f'saved {a.pre_ckpt}')
        set_seed(a.seed + 1); rng = np.random.RandomState(a.seed + 1); cfaug.rng = rng  # same classifier stage whether cached or not
    # ---------- classification on the K sessions ----------
    if a.method == 'cfpub':  # frozen extractor; linear classifier on the augmented K sessions, epoch selected on val
        n_aug = 10; ids = np.repeat(tr, n_aug); Za = []; bb.eval()  # CF augments the support traces before the linear classifier
        for b in range(0, len(ids), 512):
            xa, va = cf_views(raw, nlen, ids[b:b + 512])
            with torch.no_grad(): Za.append(bb(xa, va).cpu())
        Ztr = F.normalize(torch.cat([encode(bb, Xn[tr], Vt[tr], device)] + Za), dim=1).to(device); ytr = torch.from_numpy(np.concatenate([y[tr], y[ids]])).to(device)
        Zv = F.normalize(encode(bb, Xn[idx['val']], Vt[idx['val']], device), dim=1).to(device); Zt = F.normalize(encode(bb, Xn[idx['test']], Vt[idx['test']], device), dim=1).to(device)
        head = nn.Linear(bb.d, NC).to(device); opt = torch.optim.Adam(head.parameters(), lr=1e-2); yv = y[idx['val']]; best = (-1, None, -1)
        for ep in range(a.epochs):
            head.train(); perm = torch.randperm(len(ytr), device=device)
            for b in range(0, len(perm), a.bs):
                bi = perm[b:b + a.bs]; loss = F.cross_entropy(head(Ztr[bi]), ytr[bi]); opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad(): pv = head(Zv).argmax(1).cpu().numpy()
            f = f1_score(yv, pv, average='macro')
            if f > best[0]: best = (f, copy.deepcopy(head.state_dict()), ep)
        head.load_state_dict(best[1]); head.eval(); best_ep = best[2]
        with torch.no_grad(): pv = head(Zv).argmax(1).cpu().numpy(); pt = head(Zt).argmax(1).cpu().numpy()
    elif a.method in ('tf', 'cf'):  # k-NN in the learnt embedding space
        Ztr = encode(bb, Xn[tr], Vt[tr], device); pv = knn_predict(Ztr, y[tr], encode(bb, Xn[idx['val']], Vt[idx['val']], device), a.knn); pt = knn_predict(Ztr, y[tr], encode(bb, Xn[idx['test']], Vt[idx['test']], device), a.knn); best_ep = -1
    else:  # netclr / tiktok: (fine-)tune backbone + linear head with cross-entropy, select on val
        head = nn.Linear(bb.d, NC).to(device); opt = torch.optim.Adam(list(bb.parameters()) + list(head.parameters()), lr=1e-3 if a.method == 'tiktok' else 3e-4)
        crit = nn.CrossEntropyLoss(); yv = y[idx['val']]; best = (-1, None, -1)
        for ep in range(a.epochs):
            bb.train(); head.train(); perm = rng.permutation(tr)
            for b in range(0, len(perm), a.bs):
                bi = perm[b:b + a.bs]
                if len(bi) < 2: continue
                loss = crit(head(bb(Xn[bi].to(device), Vt[bi].to(device))), torch.from_numpy(y[bi]).to(device)); opt.zero_grad(); loss.backward(); opt.step()
            head.eval(); pv = head(encode(bb, Xn[idx['val']], Vt[idx['val']], device).to(device)).argmax(1).cpu().numpy(); f = f1_score(yv, pv, average='macro')
            if f > best[0]: best = (f, (copy.deepcopy(bb.state_dict()), copy.deepcopy(head.state_dict())), ep)
        bb.load_state_dict(best[1][0]); head.load_state_dict(best[1][1]); head.eval(); best_ep = best[2]
        pv = head(encode(bb, Xn[idx['val']], Vt[idx['val']], device).to(device)).argmax(1).cpu().numpy(); pt = head(encode(bb, Xn[idx['test']], Vt[idx['test']], device).to(device)).argmax(1).cpu().numpy()
    val_m = metrics_from_preds(y[idx['val']], pv, sessions=data.session_id[idx['val']], seed=a.seed, n_classes=NC, n_act=n_act)
    test_m = metrics_from_preds(y[idx['test']], pt, sessions=data.session_id[idx['test']], seed=a.seed, n_classes=NC, n_act=n_act)
    n_params = sum(p_.numel() for p_ in bb.parameters()); tt = round(time.time() - t0, 1)
    rec = dict(run_id=run_id, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=a.tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=sm['hash'], seed=a.seed, best_epoch=best_ep, params=n_params, train_time_s=tt, val=val_m)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec) + '\n')
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=run_id, cfg_hash=cfg_hash, seed=a.seed, tag=a.tag, split_hash=sm['hash'], best_epoch=best_ep, test=test_m)) + '\n')
    np.savez(os.path.join(run_dir, 'test_preds.npz'), y=y[idx['test']], pred=pt, idx=idx['test'])
    log(f'BEST epoch {best_ep} val macro_f1 {val_m["macro_f1"]:.4f} | params {n_params} time {tt}s'); logf.close()


if __name__ == '__main__':
    main()
