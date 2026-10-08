"""Website-fingerprinting few-shot attackers ported as faithfully as possible from their publications and official code to the
64-packet x 4-channel metadata sequence of our threat model. src/train_wf_baselines.py holds the adapted variants of the same
attacks (one shared DF backbone with global pooling); the variants here keep each publication's own architecture, augmentation
and training schedule.

Unavoidable deviations, shared by all methods and stated in the paper: input is 64 packets x (direction, IP size, payload size,
IAT) instead of 5000 Tor cells of direction (Tik-Tok: direction x time); DF max-pooling uses size/stride 2 instead of 8/4 because
L=64 would otherwise collapse to one position; the checkpoint is the epoch with the best validation macro-F1 or, with
CKPT_SELECT=last, the final epoch of the published schedule (never selected on the test set).

  netclr   NetCLR (Bahramali et al., CCS'23; official notebooks SPIN-UMass/Realistic-Website-Fingerprinting...):
           NetAugment (pick one of: change incoming burst sizes / merge incoming bursts / insert outgoing burst, then shift),
           SimCLR NT-Xent T=0.5, batch 256, Adam 3e-4 with cosine schedule (T_max = batches per epoch, stepped per epoch after
           epoch 10), 401 epochs, projection Linear(d,d)-BN-ReLU-Linear(d,128) on the flattened DF features; DF as in the NetCLR
           code (BN after the second conv of each block). Fine-tuning: DF + Linear(d, classes), Adam 1e-4, batch 32, 31 epochs.
           Pre-trained on the unlabelled corpus (train+val sessions of both campaigns). Burst constants kept as published
           (bursts >= 10 packets, first 20 packets untouched, shift 10); the length rule of "change burst sizes" (up below 1000,
           down above 4000 of 5000 cells) is scaled to 13 and 51 of 64 packets and follows the paper (the released code always
           up-samples). Packets added by a manipulation copy size and IAT from a random packet of the same burst / direction.
  netclrtr NetCLR pre-training (as above) on our Transformer encoder, fine-tuned with our own recipe: pretext-task ablation.
  (tf/tiktok/cf standardise with statistics of all non-test sessions of the target campaign, used unlabelled.)
  tf       Triplet Fingerprinting (Sirinam et al., CCS'19; official code triplet-fingerprinting/tf): DF -> Dense(64) embedding,
           cosine triplet loss alpha 0.1, a fixed pool of 25 flows per class drawn once from the labelled training sessions of
           the OTHER campaign, all positive pairs of the pool, negatives from the pool mined semi-hard (condition of the
           official code) with the previous epoch's model (random in epoch 0), SGD lr 1e-3 momentum 0.9 Nesterov decay 1e-6,
           batch 128, 30 epochs; classification by N-MEV (mean embedding of each class's K labelled sessions, cosine nearest).
  tiktok   Tik-Tok (Rahman et al., PETS'20; official code msrocean/Tik_Tok): DF classifier on direction x timestamp, FC dropouts
           0.5/0.7, Adamax 2e-3, batch 32, 40 epochs.
  cf       Contrastive Fingerprinting (Xie et al., WWW'24; no public code): DF -> FC 256 feature extractor, MLP projection to 128,
           supervised contrastive loss with inject/remove augmentation (10%) on the labelled training sessions of the OTHER
           campaign, 30 epochs; frozen extractor + linear classifier on the augmented K sessions.
"""
import argparse, hashlib, json, os, sys, time, copy, threading, queue, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds, set_seed
from train_wf_baselines import raw_to_feat, tiktok_feat, CFAugmentor, load_other_campaign, supcon, nt_xent, TrBackbone

L = 64
ACK = np.array([1.0, 52.0, 0.0, 0.001], np.float32)


# ---------------- NetAugment ported to packets ----------------
def bursts_of(m):
    """Consecutive runs of equal direction -> list of packet arrays."""
    out, s = [], 0
    for i in range(1, len(m) + 1):
        if i == len(m) or m[i, 0] != m[s, 0]:
            out.append(m[s:i]); s = i
    return out


class NetAugment:
    def __init__(self, rng, out_burst_sizes, large=10, up=1.0, down=0.5, n_merge=5, r_merge=0.1, r_insert=0.3, shift=10, skip=20,
                 lo=13, hi=51):
        self.r = rng; self.obs = np.asarray(out_burst_sizes); self.large = large; self.up = up; self.down = down
        self.n_merge = n_merge; self.r_merge = r_merge; self.r_insert = r_insert; self.shift_p = shift; self.skip = skip
        self.lo = lo; self.hi = hi

    def _skip(self, bs):  # bursts covering the first `skip` packets are left untouched
        i = n = 0
        while i < len(bs) and n < self.skip: n += len(bs[i]); i += 1
        return i

    def _resize(self, b, new):
        r = self.r
        if new > len(b):  # copy random packets of the same burst at random positions
            for _ in range(new - len(b)):
                b = np.insert(b, r.randint(1, len(b) + 1), b[r.randint(len(b))], axis=0)
        elif new < len(b):  # drop random packets (never the first), their gaps go to the next packet
            keep = np.ones(len(b), bool); keep[1 + r.choice(len(b) - 1, len(b) - new, replace=False)] = False
            b = b.copy(); acc = 0.0
            for j in range(len(b)):
                if not keep[j]: acc += b[j, 3]
                elif acc: b[j, 3] += acc; acc = 0.0
            b = b[keep]
        return b

    def change_content(self, bs, n):
        r = self.r; up = n < self.lo or (self.lo <= n <= self.hi and r.rand() >= 0.5)
        k = self._skip(bs); out = bs[:k]
        for b in bs[k:]:
            if b[0, 0] < 0 and len(b) >= self.large:
                f = 1 + r.rand() * self.up if up else 1 - r.rand() * self.down
                b = self._resize(b, max(1, int(len(b) * f)))
            out.append(b)
        return out

    def merge_incoming(self, bs, n):
        r = self.r; i = self._skip(bs); out = bs[:i]
        while i < len(bs):
            if bs[i][0, 0] > 0 or r.rand() >= self.r_merge:
                out.append(bs[i]); i += 1; continue
            k = r.randint(2, self.n_merge + 1); parts = []; gap = 0.0
            while i < len(bs) and k > 0:  # merge the next k incoming bursts; outgoing bursts in between are removed
                if bs[i][0, 0] < 0:
                    b = bs[i].copy(); b[0, 3] += gap; gap = 0.0; parts.append(b); k -= 1
                else: gap += bs[i][:, 3].sum()
                i += 1
            out.append(np.concatenate(parts))
        return out

    def insert_outgoing(self, bs, n, m):
        r = self.r; k = self._skip(bs); out = bs[:k]; ups = m[m[:, 0] > 0]
        for b in bs[k:]:
            if b[0, 0] > 0 or len(b) < self.large or r.rand() >= self.r_insert:
                out.append(b); continue
            s = int(self.obs[r.randint(len(self.obs))]); d = r.randint(3, len(b) - 3 + 1)
            ins = np.stack([ups[r.randint(len(ups))] if len(ups) else ACK for _ in range(s)])
            out += [b[:d], ins, b[d:]]
        return out

    def shift(self, m):
        r = self.r; s = r.randint(-self.shift_p, self.shift_p + 1)
        if s < 0: return m[min(-s, len(m) - 1):]
        if s > 0:
            pre = []
            for _ in range(s):
                d = r.choice([-1.0, 1.0]); pool = m[m[:, 0] == d]
                p = pool[r.randint(len(pool))].copy() if len(pool) else m[r.randint(len(m))].copy(); p[0] = d; pre.append(p)
            return np.concatenate([np.stack(pre), m])
        return m

    def __call__(self, m, n):
        m = m[:n].astype(np.float32); bs = bursts_of(m); c = self.r.randint(3)
        bs = self.change_content(bs, n) if c == 0 else self.merge_incoming(bs, n) if c == 1 else self.insert_outgoing(bs, n, m)
        m = self.shift(np.concatenate(bs))[:L]
        return m, len(m)


# ---------------- DF ----------------
class DFConv(nn.Module):
    """DF conv blocks (32/64/128/256 filters, kernel 8, ELU then ReLU, dropout 0.1) with max-pool 2; flattened output.
    bn_both=True: BN after both convs of a block (DF / Tik-Tok / TF code); False: after the second only (NetCLR code)."""

    def __init__(self, C=4, bn_both=True):
        super().__init__(); ch = [C, 32, 64, 128, 256]; layers = []
        for i in range(4):
            act = nn.ELU if i == 0 else nn.ReLU
            layers += [nn.Conv1d(ch[i], ch[i + 1], 8, padding='same')] + ([nn.BatchNorm1d(ch[i + 1])] if bn_both else []) + [act(),
                       nn.Conv1d(ch[i + 1], ch[i + 1], 8, padding='same'), nn.BatchNorm1d(ch[i + 1]), act(), nn.MaxPool1d(2), nn.Dropout(0.1)]
        self.net = nn.Sequential(*layers); self.d = 256 * (L // 16)
        for mod in self.modules():
            if isinstance(mod, (nn.Conv1d, nn.Linear)): nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)

    def forward(self, x, v=None): return self.net(x.transpose(1, 2)).flatten(1)


def df_head(d, NC, p1, p2):
    h = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                      nn.Dropout(p2), nn.Linear(512, NC))
    for mod in h.modules():
        if isinstance(mod, nn.Linear): nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)
    return h


class Seq(nn.Module):
    def __init__(self, bb, top): super().__init__(); self.bb = bb; self.top = top

    def forward(self, x, v): return self.top(self.bb(x, v))


# ---------------- helpers ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['netclr', 'netclrtr', 'tf', 'tiktok', 'cf'])
    p.add_argument('--run_name', default=''); p.add_argument('--tag', default='wf_faithful'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', required=True); p.add_argument('--dataset', default='genai', choices=['genai', 'ccma']); p.add_argument('--target', default='joint', choices=['joint', 'app'])
    p.add_argument('--pre_ckpt', default='', help='cache of the pre-trained backbone (netclr / netclrtr / cf)')
    p.add_argument('--pre_epochs', type=int, default=-1, help='override for smoke tests only'); p.add_argument('--epochs', type=int, default=-1)
    p.add_argument('--out_dir', default='results/runs')
    p.add_argument('--defense', default='none', choices=['none', 'pad256', 'jitter20', 'pad256_jitter20', 'front', 'tamaraw', 'ech'])
    p.add_argument('--defense_train', type=int, default=0)
    return p.parse_args()


def encode(m, X, V, device, bs=512):
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs): out.append(m(X[b:b + bs].to(device), V[b:b + bs].to(device)).cpu())
    return torch.cat(out)


def corpus(target_ds=None, split=None):
    """Unlabelled pre-training corpus: train+val sessions of both campaigns (standard split), labels discarded, minus the
    test sessions of the evaluation split (identical to the standard test sessions for the few-shot splits)."""
    cr, cl = [], []
    drop = set(json.load(open(os.path.join(ROOT, split)))['test']) if split else set()
    for ds in ('genai', 'ccma'):
        d = GenAIData(ROOT, prefix=ds); sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', 'session_split_s2026.json' if ds == 'genai' else 'ccma_session_split_s2026.json')))
        files = set(sp['train'] + sp['val']) - (drop if ds == target_ds else set())
        keep = np.array([d.file2sid[f] for f in sorted(files)]); ii = np.where(np.isin(d.session_id, keep))[0]
        cr.append(d.meta[ii, :L].astype(np.float32)); cl.append(np.minimum(d.meta_len[ii], L))
    return np.concatenate(cr), np.concatenate(cl)


def stats(X, V):
    mu = (X * V[..., None]).sum((0, 1)) / V.sum(); sd = np.sqrt((((X - mu) ** 2) * V[..., None]).sum((0, 1)) / V.sum()) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


# CKPT_SELECT=last keeps the final epoch of the published schedule instead of the best validation epoch, so no labels
# beyond the K training sessions are used.
LAST = os.environ.get('CKPT_SELECT', 'val') == 'last'


def train_classifier(model, params, lr, bs, epochs, Xn, Vt, y, tr, iv, device, rng, opt_name='adam', wd=0.0):
    """Cross-entropy training on the labelled flows; the epoch with the best validation macro-F1 is kept (with
    CKPT_SELECT=last: the final epoch)."""
    opt = {'adam': lambda: torch.optim.Adam(params, lr=lr), 'adamax': lambda: torch.optim.Adamax(params, lr=lr),
           'adamw': lambda: torch.optim.AdamW(params, weight_decay=wd)}[opt_name]()
    best = (-1, None, -1)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]
            if len(bi) < 2: continue
            loss = F.cross_entropy(model(Xn[bi].to(device), Vt[bi].to(device)), torch.from_numpy(y[bi]).to(device)); opt.zero_grad(); loss.backward(); opt.step()
        pv = encode(model, Xn[iv], Vt[iv], device).argmax(1).numpy(); f = f1_score(y[iv], pv, average='macro')
        if (ep == epochs - 1) if LAST else (f > best[0]): best = (f, copy.deepcopy(model.state_dict()), ep)
    model.load_state_dict(best[1]); return best[2]


def main():
    a = get_args(); set_seed(a.seed); device = torch.device('cuda'); rng = np.random.RandomState(a.seed)
    cfg = {k: v for k, v in vars(a).items() if k not in ('run_name', 'tag', 'out_dir')}; cfg['carrier'] = 'wf_faithful'
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    run_id = a.run_name or f'{a.method}_{cfg_hash}_s{a.seed}'; run_dir = os.path.join(ROOT, a.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'train.log'), 'a', encoding='utf-8')

    def log(s): print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== run {run_id} {json.dumps(cfg)}')
    data = GenAIData(ROOT, prefix=a.dataset); idx, sm = data.split_indices(os.path.join(ROOT, a.split))
    n_act = data.n_act if a.target == 'joint' else 1; NC = data.n_joint if a.target == 'joint' else data.n_app; y = data.y_joint if a.target == 'joint' else data.y_app
    raw = data.meta[:, :L].astype(np.float32); nlen = np.minimum(data.meta_len, L)

    def shape(r_, n_):
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
    if a.method in ('netclr', 'netclrtr'):  # one normalisation for the shared pre-training corpus and every fine-tuning set
        craw, clen = corpus(a.dataset, a.split); CX = np.stack([featf(craw[i], clen[i]) for i in range(len(craw))]); mu, sd = stats(CX, np.arange(L)[None, :] < clen[:, None])
    else:
        nt = np.setdiff1d(np.where(y >= 0)[0], idx['test']); mu, sd = stats(X[nt], V[nt])  # unlabelled traffic of all non-test sessions, never test
    norm = lambda x, v: ((x - mu) / sd * v[..., None]).astype(np.float32)
    Xn = torch.from_numpy(norm(X, V)); Vt = torch.from_numpy(V); tr, iv, it = idx['train'], idx['val'], idx['test']; t0 = time.time()

    def views(aug, r_, n_, ids):
        xs, vs = [], []
        for i in ids:
            m, n = aug(r_[i], n_[i]); v = np.arange(L) < n; xs.append(norm(featf(m, n), v)); vs.append(v)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(vs))

    ck = os.path.join(ROOT, a.pre_ckpt) if a.pre_ckpt else ''
    cached = bool(ck) and os.path.exists(ck)
    if cached:
        st = torch.load(ck, weights_only=False); assert np.allclose(st['norm'][0], mu) and np.allclose(st['norm'][1], sd), 'normalisation mismatch'

    if a.method in ('netclr', 'netclrtr'):
        bb = (TrBackbone() if a.method == 'netclrtr' else DFConv(4, bn_both=False)).to(device)
        if cached: bb.load_state_dict(st['backbone']); log(f'loaded {a.pre_ckpt}')
        else:
            obs = []  # empirical outgoing-burst sizes (packets) of 1000 random corpus flows
            for i in rng.choice(len(craw), 1000, replace=False): obs += [len(b) for b in bursts_of(craw[i][:clen[i]]) if b[0, 0] > 0]
            aug = NetAugment(rng, obs); d = bb.d
            proj = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, 128)).to(device)
            bs = 256; nb = len(craw) // bs; E = 401 if a.pre_epochs < 0 else a.pre_epochs
            opt = torch.optim.Adam(list(bb.parameters()) + list(proj.parameters()), lr=3e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nb)
            q = queue.Queue(maxsize=8)

            def producer():  # CPU augmentation overlapped with GPU training
                for ep in range(E):
                    perm = rng.permutation(len(craw))
                    for b in range(nb):
                        bi = perm[b * bs:(b + 1) * bs]; q.put((views(aug, craw, clen, bi), views(aug, craw, clen, bi)))
                q.put(None)

            threading.Thread(target=producer, daemon=True).start(); log(f'NetCLR pre-training: {len(craw)} flows, {E} epochs, {nb} batches/epoch')
            for ep in range(E):
                bb.train(); proj.train(); tot = 0.0
                for _ in range(nb):
                    (x1, v1), (x2, v2) = q.get()
                    z = proj(bb(torch.cat([x1, x2]).to(device), torch.cat([v1, v2]).to(device)))
                    loss = nt_xent(z[:len(x1)], z[len(x1):], T=0.5); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
                if ep >= 10: sch.step()
                if ep % 20 == 0 or ep == E - 1: log(f'netclr ep {ep} loss {tot / nb:.4f} lr {opt.param_groups[0]["lr"]:.2e} {time.time() - t0:.0f}s')
            if ck: torch.save({'backbone': bb.state_dict(), 'norm': (mu, sd)}, ck); log(f'saved {a.pre_ckpt}')
        set_seed(a.seed + 1); rng = np.random.RandomState(a.seed + 1)
        if a.method == 'netclr':  # official fine-tuning: DF + Linear, Adam 1e-4, batch 32, 31 epochs
            model = Seq(bb, nn.Linear(bb.d, NC)).to(device)
            best_ep = train_classifier(model, model.parameters(), 1e-4, 32, 31 if a.epochs < 0 else a.epochs, Xn, Vt, y, tr, iv, device, rng)
        else:  # our fine-tuning recipe (src/train_meta_ssl.py): AdamW, encoder 3e-4, head 1e-3, wd 0.01, head dropout 0.3
            model = Seq(bb, nn.Sequential(nn.Dropout(0.3), nn.Linear(bb.d, NC))).to(device)
            params = [{'params': bb.parameters(), 'lr': 3e-4}, {'params': model.top.parameters(), 'lr': 1e-3}]
            best_ep = train_classifier(model, params, None, 64, 60 if a.epochs < 0 else a.epochs, Xn, Vt, y, tr, iv, device, rng, 'adamw', 0.01)
        pv = encode(model, Xn[iv], Vt[iv], device).argmax(1).numpy(); pt = encode(model, Xn[it], Vt[it], device).argmax(1).numpy()

    elif a.method == 'tiktok':
        model = Seq(DFConv(1), None).to(device); model.top = df_head(model.bb.d, NC, 0.5, 0.7).to(device)
        best_ep = train_classifier(model, model.parameters(), 2e-3, 32, 40 if a.epochs < 0 else a.epochs, Xn, Vt, y, tr, iv, device, rng, 'adamax')
        pv = encode(model, Xn[iv], Vt[iv], device).argmax(1).numpy(); pt = encode(model, Xn[it], Vt[it], device).argmax(1).numpy()

    elif a.method == 'tf':
        other, oy, oraw, olen = load_other_campaign(a.dataset)
        oX = torch.from_numpy(norm(np.stack([featf(oraw[i], olen[i]) for i in range(len(oraw))]), np.arange(L)[None, :] < olen[:, None])); oV = torch.from_numpy(np.arange(L)[None, :] < olen[:, None])
        emb = Seq(DFConv(4), None).to(device); emb.top = nn.Linear(emb.bb.d, 64).to(device); nn.init.xavier_uniform_(emb.top.weight); nn.init.zeros_(emb.top.bias)
        opt = torch.optim.SGD(emb.parameters(), lr=1e-3, momentum=0.9, nesterov=True); decay = 1e-6; it_n = 0
        cls = np.unique(oy); E = 30 if a.pre_epochs < 0 else a.pre_epochs; sims = None
        # official model_training.py: a fixed training pool of 25 traces per class; all positive pairs are built once and
        # shuffled; negatives are drawn from the same pool, semi-hard w.r.t. the previous epoch's embedding (random in epoch 0)
        pool = np.concatenate([rng.choice(np.where(oy == c)[0], min(25, (oy == c).sum()), replace=False) for c in cls])
        oX, oV, oy = oX[pool], oV[pool], oy[pool]
        A, P = [], []
        for c in cls:
            ids = np.where(oy == c)[0]
            A += [ids[i] for i in range(len(ids)) for j in range(i + 1, len(ids))]; P += [ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids))]
        perm = rng.permutation(len(A)); A = np.array(A)[perm]; P = np.array(P)[perm]
        log(f'TF extractor on {other}: pool {len(oy)} flows, {len(cls)} classes, {len(A)} positive pairs')
        for ep in range(E):
            Nn = []
            for an, po in zip(A, P):
                if sims is None: Nn.append(rng.choice(np.where(oy != oy[an])[0])); continue
                cand = np.where((sims[an] + 0.1 > sims[an, po]) & (oy != oy[an]))[0]
                Nn.append(rng.choice(cand) if len(cand) else rng.choice(np.where(oy != oy[an])[0]))
            Nn = np.array(Nn); emb.train(); tot = 0.0; nb = max(1, len(A) // 128)
            for b in range(nb):
                sl = slice(b * 128, (b + 1) * 128); za, zp, zn = [F.normalize(emb(oX[s].to(device), oV[s].to(device)), dim=1) for s in (A[sl], P[sl], Nn[sl])]
                loss = F.relu((za * zn).sum(1) - (za * zp).sum(1) + 0.1).mean()
                for g in opt.param_groups: g['lr'] = 1e-3 / (1 + decay * it_n)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); it_n += 1
            E_all = F.normalize(encode(emb, oX, oV, device), dim=1).numpy(); sims = E_all @ E_all.T
            if ep % 10 == 0 or ep == E - 1: log(f'tf ep {ep} loss {tot / nb:.4f}')
        # N-MEV (the representation behind the main results of the TF paper): one mean embedding per class, cosine nearest neighbour
        Ztr = F.normalize(encode(emb, Xn[tr], Vt[tr], device), dim=1)
        cen = F.normalize(torch.stack([Ztr[torch.from_numpy(y[tr] == c)].mean(0) for c in range(NC)]), dim=1)
        pv = (F.normalize(encode(emb, Xn[iv], Vt[iv], device), dim=1) @ cen.T).argmax(1).numpy()
        pt = (F.normalize(encode(emb, Xn[it], Vt[it], device), dim=1) @ cen.T).argmax(1).numpy(); best_ep = -1

    elif a.method == 'cf':
        other, oy, oraw, olen = load_other_campaign(a.dataset)
        if a.defense != 'none' and a.defense_train: oraw, olen, _ = shape(oraw, olen)
        ext = Seq(DFConv(4), None).to(device); ext.top = nn.Sequential(nn.Linear(ext.bb.d, 256), nn.BatchNorm1d(256), nn.ReLU()).to(device)
        cfaug = CFAugmentor(rng)
        if cached: ext.load_state_dict(st['backbone']); log(f'loaded {a.pre_ckpt}')
        else:
            proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
            opt = torch.optim.Adam(list(ext.parameters()) + list(proj.parameters()), lr=1e-3); E = 30 if a.pre_epochs < 0 else a.pre_epochs
            log(f'CF pre-training on {other}: {len(oy)} flows');
            for ep in range(E):
                ext.train(); perm = rng.permutation(len(oy)); tot = 0.0; nb = 0
                for b in range(0, len(perm), 128):
                    bi = perm[b:b + 128]
                    if len(bi) < 8: continue
                    x1, v1 = views(cfaug, oraw, olen, bi); x2, v2 = views(cfaug, oraw, olen, bi)
                    z = proj(ext(torch.cat([x1, x2]).to(device), torch.cat([v1, v2]).to(device)))
                    loss = supcon(z, torch.from_numpy(np.concatenate([oy[bi], oy[bi]])).to(device)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
                if ep % 10 == 0 or ep == E - 1: log(f'cf ep {ep} loss {tot / max(nb, 1):.4f}')
            if ck: torch.save({'backbone': ext.state_dict(), 'norm': (mu, sd)}, ck); log(f'saved {a.pre_ckpt}')
        set_seed(a.seed + 1); rng = np.random.RandomState(a.seed + 1); cfaug.r = rng; cfaug.rng = rng
        ids = np.repeat(tr, 10); Za = []  # the support flows are augmented before the linear classifier
        for b in range(0, len(ids), 512):
            xa, va = views(cfaug, raw, nlen, ids[b:b + 512]); Za.append(encode(ext, xa, va, device))
        Ztr = torch.cat([encode(ext, Xn[tr], Vt[tr], device)] + Za).to(device); ytr = torch.from_numpy(np.concatenate([y[tr], y[ids]])).to(device)
        Zv = encode(ext, Xn[iv], Vt[iv], device).to(device); Zt = encode(ext, Xn[it], Vt[it], device).to(device)
        head = nn.Linear(256, NC).to(device); opt = torch.optim.Adam(head.parameters(), lr=1e-2); best = (-1, None, -1)
        E_head = 60 if a.epochs < 0 else a.epochs
        for ep in range(E_head):
            head.train(); perm = torch.randperm(len(ytr), device=device)
            for b in range(0, len(perm), 64):
                bi = perm[b:b + 64]; loss = F.cross_entropy(head(Ztr[bi]), ytr[bi]); opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad(): f = f1_score(y[iv], head(Zv).argmax(1).cpu().numpy(), average='macro')
            if (ep == E_head - 1) if LAST else (f > best[0]): best = (f, copy.deepcopy(head.state_dict()), ep)
        head.load_state_dict(best[1]); head.eval(); best_ep = best[2]
        with torch.no_grad(): pv = head(Zv).argmax(1).cpu().numpy(); pt = head(Zt).argmax(1).cpu().numpy()

    val_m = metrics_from_preds(y[iv], pv, sessions=data.session_id[iv], seed=a.seed, n_classes=NC, n_act=n_act)
    test_m = metrics_from_preds(y[it], pt, sessions=data.session_id[it], seed=a.seed, n_classes=NC, n_act=n_act)
    tt = round(time.time() - t0, 1)
    rec = dict(run_id=run_id, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=a.tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=sm['hash'], seed=a.seed, best_epoch=best_ep, train_time_s=tt, val=val_m)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec) + '\n')
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=run_id, cfg_hash=cfg_hash, seed=a.seed, tag=a.tag, split_hash=sm['hash'], best_epoch=best_ep, test=test_m)) + '\n')
    np.savez(os.path.join(run_dir, 'test_preds.npz'), y=y[it], pred=pt, idx=it)
    log(f'BEST epoch {best_ep} val macro_f1 {val_m["macro_f1"]:.4f} | time {tt}s'); logf.close()


if __name__ == '__main__':
    main()
