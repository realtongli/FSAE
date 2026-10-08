"""Open-world evaluation with held-out APPS as the unmonitored set (the standard website-fingerprinting setting).

The attacker monitors M of the campaign's apps and labels K sessions per monitored app. Everything else that the phone
sends is unmonitored: the flows of the held-out apps (which the attacker has never labelled) and the background flows of
every session (system services and other apps, scripts/build_background.py). Splits stay session-level.

Two decisions are evaluated:
  flow level   : report a flow as monitored class c when softmax confidence >= t, else discard it.
  window level : the observer watches one 15-minute capture session and reports the monitored app it believes was used,
                 or nothing. Score of class c = number of the window's flows reported as c; the window is reported as
                 argmax_c when that count >= m. Windows of held-out apps must yield no report.
Reports TPR / FPR / precision at both levels; writes results/open_world_apps.jsonl.
"""
import argparse, copy, json, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from train_meta import features as meta_features
from train_baselines import stat_features, DFMetaCNN
from pretrain_meta import MetaEncoder
import train_meta_ssl

L = 64


def load_bg(ds):
    z = np.load(os.path.join(ROOT, 'data', 'derived', f'{ds}_background.npz'), allow_pickle=True)
    m = z['meta'][:, :L]; x = np.zeros_like(m)
    x[..., 0] = m[..., 0]; x[..., 1] = np.log1p(m[..., 1]); x[..., 2] = np.log1p(m[..., 2]); x[..., 3] = np.log1p(m[..., 3] * 1000.0)
    v = (np.arange(L)[None, :] < np.minimum(z['meta_len'], L)[:, None]); x[~v] = 0.0
    return x.astype(np.float32), v, z['session_id'], z['meta'][:, :L], z['meta_len']


class WithGlobals(nn.Module):
    """Classifier of src/train_meta_ssl.py plus three explicit flow-level scalars (number of packets, log total bytes, log duration).
    The pooled token representation is length-invariant, which is harmless in closed world but makes short system-service
    flows look like app flows; these scalars restore the information a statistics-based attacker has for free."""

    def __init__(self, enc, n_cls, drop=0.3):
        super().__init__(); self.enc = enc
        self.g = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * enc.d + 32, n_cls))

    def forward(self, x, v, g):
        return self.head(torch.cat([self.enc.pool(self.enc(x, v), v), self.g(g)], 1))


def globals_of(X, V):
    """[N,3]: packet count, log(1+sum of payload-length channel), log(1+sum of the IAT channel); X already log-scaled."""
    n = V.sum(1, keepdims=True).astype(np.float32)
    return np.concatenate([np.log1p(n), np.log1p((X[..., 2] * V).sum(1, keepdims=True)), np.log1p((X[..., 3] * V).sum(1, keepdims=True))], 1).astype(np.float32)


def fit_ours(Xtr, Vtr, ytr, Xva, Vva, yva, NC, init, seed, epochs=60, use_globals=False):
    torch.manual_seed(seed); np.random.seed(seed)
    if init:
        pre = torch.load(os.path.join(ROOT, init), weights_only=False); c = pre['cfg']; enc = MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L']); enc.load_state_dict(pre['encoder'])
    else:
        enc = MetaEncoder(192, 8, 8, 384, L)
    model = (WithGlobals(enc, NC) if use_globals else train_meta_ssl.Classifier(enc, NC, 0.3)).cuda()
    head_params = [p for n_, p in model.named_parameters() if not n_.startswith('enc.')]
    opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': 3e-4}, {'params': head_params, 'lr': 1e-3}], weight_decay=0.01)
    crit = nn.CrossEntropyLoss(); rng = np.random.RandomState(seed); best = (-1, None)
    Xtr_t, Vtr_t = torch.from_numpy(Xtr), torch.from_numpy(Vtr); Xva_t, Vva_t = torch.from_numpy(Xva), torch.from_numpy(Vva)
    Gtr = torch.from_numpy(globals_of(Xtr, Vtr)) if use_globals else None; Gva = torch.from_numpy(globals_of(Xva, Vva)) if use_globals else None
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(ytr))
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]
            if len(bi) < 2: continue
            args_ = (Xtr_t[bi].cuda(), Vtr_t[bi].cuda()) + ((Gtr[bi].cuda(),) if use_globals else ())
            loss = crit(model(*args_), torch.from_numpy(ytr[bi]).cuda()); opt.zero_grad(); loss.backward(); opt.step()
        pv = predict_probs(model, Xva_t, Vva_t, G=Gva).argmax(1); f = f1_score(yva, pv, average='macro')
        if f > best[0]: best = (f, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1]); model.eval(); return model


def predict_probs(model, X, V, bs=512, G=None):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            args_ = (X[b:b + bs].cuda(), V[b:b + bs].cuda()) + ((G[b:b + bs].cuda(),) if G is not None else ())
            out.append(torch.softmax(model(*args_), 1).cpu().numpy())
    return np.concatenate(out)


def flow_metrics(conf_mon, correct_mon, conf_unm, ratios=(1, 10, 100)):
    n_mon, n_unm = len(conf_mon), len(conf_unm)
    ts = np.unique(np.concatenate([conf_unm, [0.0, 1.01]]))
    if len(ts) > 300: ts = np.quantile(ts, np.linspace(0, 1, 300))
    best_f1 = 0.0; at1 = None
    for t in ts:
        rep = conf_mon >= t; tp = int((rep & correct_mon).sum()); fp_m = int((rep & ~correct_mon).sum()); fp_u = int((conf_unm >= t).sum())
        tpr = tp / max(n_mon, 1); fpr = fp_u / max(n_unm, 1); prec = tp / max(tp + fp_m + fp_u, 1)
        f1 = 2 * prec * tpr / max(prec + tpr, 1e-9); best_f1 = max(best_f1, f1)
        if fpr <= 0.01 and (at1 is None or tpr > at1['tpr']):
            w = {str(r): r * n_mon / max(n_unm, 1) for r in ratios}
            at1 = dict(t=float(t), tpr=tpr, fpr=fpr, precision=prec, precision_at_ratio={r: tp / max(tp + fp_m + w[r] * fp_u, 1e-9) for r in w})
    return dict(n_mon=n_mon, n_unm=n_unm, best_f1=best_f1, at_fpr1=at1)


def window_metrics(P, sid, ytrue, t_grid, m_grid):
    """P: [N,C] probs of all flows of the test windows; sid: window id per flow; ytrue: true class per window (-1 unmonitored)."""
    pred = P.argmax(1); conf = P.max(1); best = None
    wins = np.unique(sid)
    for t in t_grid:
        rep = conf >= t
        counts = {w: np.bincount(pred[(sid == w) & rep], minlength=P.shape[1]) if ((sid == w) & rep).any() else np.zeros(P.shape[1], int) for w in wins}
        for m in m_grid:
            tp = fp = fn = 0
            for w in wins:
                c = counts[w]; top = int(c.argmax()); n = int(c[top]); reported = n >= m
                yt = ytrue[w]
                if yt >= 0:
                    if reported and top == yt: tp += 1
                    else: fn += 1
                    if reported and top != yt: fp += 1
                else:
                    if reported: fp += 1
            n_mon_w = int((ytrue[wins] >= 0).sum()); n_unm_w = len(wins) - n_mon_w
            tpr = tp / max(n_mon_w, 1); prec = tp / max(tp + fp, 1); fpr = sum(1 for w in wins if ytrue[w] < 0 and counts[w].max() >= m) / max(n_unm_w, 1)
            f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)
            if best is None or f1 > best['f1']: best = dict(f1=f1, tpr=tpr, fpr=fpr, precision=prec, t=float(t), m=int(m), n_mon_win=n_mon_w, n_unm_win=n_unm_w)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='ccma'); ap.add_argument('--target', default='app')
    ap.add_argument('--K', type=int, default=4); ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--n_mon', type=int, default=5)
    ap.add_argument('--attackers', default='ours,scratch,lgbm,dfmeta'); ap.add_argument('--tag', default='ow_apps')
    ap.add_argument('--init', default='weights/meta_ssl_mpm_XL.pt')
    ap.add_argument('--bg_class', type=int, default=0, help='1 = the attacker also collects unlabelled traffic of a device that does not run the monitored apps and trains an extra "unmonitored" class on it')
    a = ap.parse_args()
    ds = a.dataset; data = GenAIData(ROOT, prefix=ds); y_all = data.y_joint if a.target == 'joint' else data.y_app
    NCfull = data.n_joint if a.target == 'joint' else data.n_app
    Xall = meta_features(data, L); Vall = (np.arange(L)[None, :] < np.minimum(data.meta_len, L)[:, None])
    Fall = stat_features(data.meta, data.meta_len)
    Xbg, Vbg, bg_sid, bg_meta, bg_len = load_bg(ds); Fbg = stat_features(bg_meta, bg_len)
    if a.init:  # the encoder expects inputs standardised exactly as during pre-training
        _pre = torch.load(os.path.join(ROOT, a.init), weights_only=False); MU, SD = _pre['mu'], _pre['sd']
    else:
        MU = (Xall * Vall[..., None]).sum((0, 1)) / Vall.sum(); SD = np.sqrt((((Xall - MU) ** 2) * Vall[..., None]).sum((0, 1)) / Vall.sum()) + 1e-6
    Xall = ((Xall - MU) / SD * Vall[..., None]).astype(np.float32); Xbg = ((Xbg - MU) / SD * Vbg[..., None]).astype(np.float32)
    out = open(os.path.join(ROOT, 'results', 'open_world_apps.jsonl'), 'a', encoding='utf-8'); rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        rng = np.random.RandomState(700 + seed); mon = np.sort(rng.choice(NCfull, a.n_mon, replace=False)); remap = {int(c): i for i, c in enumerate(mon)}
        split = os.path.join(ROOT, 'configs', 'splits', (f'fewshot_k{a.K}_s{seed}.json' if ds == 'genai' else f'ccma_fewshot_k{a.K}_s{seed}.json'))
        idx, sm = data.split_indices(split); sp = json.load(open(split))
        test_sids = np.array([data.file2sid[f] for f in sp['test']])
        keep = lambda ii: ii[np.isin(y_all[ii], mon)]
        tr, va = keep(idx['train']), keep(idx['val'])
        ytr = np.array([remap[int(c)] for c in y_all[tr]]); yva = np.array([remap[int(c)] for c in y_all[va]])
        te = idx['test']; te_mon = te[np.isin(y_all[te], mon)]; te_unm_app = te[~np.isin(y_all[te], mon)]
        bg_te = np.where(np.isin(bg_sid, test_sids))[0]
        yte_mon = np.array([remap[int(c)] for c in y_all[te_mon]])
        # window ids and truth: monitored windows are test sessions of monitored apps; unmonitored windows the others
        sid_mon = data.session_id[te_mon]; sid_unm_app = data.session_id[te_unm_app]; sid_bg = bg_sid[bg_te]
        win_truth = {}
        for i, s in zip(te_mon, sid_mon): win_truth[int(s)] = remap[int(y_all[i])]
        for s in np.unique(sid_unm_app): win_truth.setdefault(int(s), -1)
        for s in np.unique(sid_bg): win_truth.setdefault(int(s), -1)
        wt = np.full(max(win_truth) + 1, -1, int)
        for s, c in win_truth.items(): wt[s] = c
        bg_tr = bg_va = np.array([], int)
        if a.bg_class:
            tr_sids = np.array([data.file2sid[f] for f in sp['train']]); va_sids = np.array([data.file2sid[f] for f in sp['val']])
            bg_tr = np.where(np.isin(bg_sid, tr_sids))[0]; bg_va = np.where(np.isin(bg_sid, va_sids))[0]
            rngb = np.random.RandomState(900 + seed)
            cap = max(len(tr), 200)  # keep the extra class comparable in size to the K labelled sessions
            if len(bg_tr) > cap: bg_tr = rngb.choice(bg_tr, cap, replace=False)
            if len(bg_va) > 2000: bg_va = rngb.choice(bg_va, 2000, replace=False)
        NCm = len(mon) + (1 if a.bg_class else 0); BGC = len(mon)
        for att in a.attackers.split(','):
            t0 = time.time()
            if att in ('ours', 'scratch', 'ourslen'):
                Xtr_, Vtr_, ytr_ = Xall[tr], Vall[tr], ytr; Xva_, Vva_, yva_ = Xall[va], Vall[va], yva
                if a.bg_class:
                    Xtr_ = np.concatenate([Xtr_, Xbg[bg_tr]]); Vtr_ = np.concatenate([Vtr_, Vbg[bg_tr]]); ytr_ = np.concatenate([ytr_, np.full(len(bg_tr), BGC)])
                    Xva_ = np.concatenate([Xva_, Xbg[bg_va]]); Vva_ = np.concatenate([Vva_, Vbg[bg_va]]); yva_ = np.concatenate([yva_, np.full(len(bg_va), BGC)])
                ug = att == 'ourslen'
                model = fit_ours(Xtr_, Vtr_, ytr_, Xva_, Vva_, yva_, NCm, '' if att == 'scratch' else a.init, seed, use_globals=ug)
                pf = lambda X, V: predict_probs(model, torch.from_numpy(X), torch.from_numpy(V), G=(torch.from_numpy(globals_of(X, V)) if ug else None))
                P_mon = pf(Xall[te_mon], Vall[te_mon]); P_ua = pf(Xall[te_unm_app], Vall[te_unm_app]); P_bg = pf(Xbg[bg_te], Vbg[bg_te])
            elif att == 'lgbm':
                import lightgbm as lgb
                mu_, sd_ = Fall[tr].mean(0), Fall[tr].std(0) + 1e-6
                Ftr_ = Fall[tr]; ytr_ = ytr; Fva_ = Fall[va]; yva_ = yva
                if a.bg_class:
                    Ftr_ = np.concatenate([Ftr_, Fbg[bg_tr]]); ytr_ = np.concatenate([ytr_, np.full(len(bg_tr), BGC)])
                    Fva_ = np.concatenate([Fva_, Fbg[bg_va]]); yva_ = np.concatenate([yva_, np.full(len(bg_va), BGC)])
                clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=seed, verbose=-1)
                clf.fit((Ftr_ - mu_) / sd_, ytr_, eval_set=[((Fva_ - mu_) / sd_, yva_)], callbacks=[lgb.early_stopping(100, verbose=False)])
                pf = lambda F: clf.predict_proba((F - mu_) / sd_)
                P_mon = pf(Fall[te_mon]); P_ua = pf(Fall[te_unm_app]); P_bg = pf(Fbg[bg_te])
            elif att == 'dfmeta':
                torch.manual_seed(seed); np.random.seed(seed)
                model = DFMetaCNN(n_cls=NCm).cuda(); opt = torch.optim.Adamax(model.parameters(), lr=2e-3); crit = nn.CrossEntropyLoss()
                Xt = torch.from_numpy(Xall).permute(0, 2, 1).contiguous(); Xb = torch.from_numpy(Xbg).permute(0, 2, 1).contiguous()
                Ttr = Xt[tr]; ytr_ = ytr; Tva = Xt[va]; yva_ = yva
                if a.bg_class:
                    Ttr = torch.cat([Ttr, Xb[bg_tr]]); ytr_ = np.concatenate([ytr_, np.full(len(bg_tr), BGC)])
                    Tva = torch.cat([Tva, Xb[bg_va]]); yva_ = np.concatenate([yva_, np.full(len(bg_va), BGC)])
                rng2 = np.random.RandomState(seed); best = (-1, None)
                for ep in range(60):
                    model.train(); perm = rng2.permutation(len(ytr_))
                    for b in range(0, len(perm), 64):
                        bi = perm[b:b + 64]
                        if len(bi) < 2: continue
                        loss = crit(model(Ttr[bi].cuda()), torch.from_numpy(ytr_[bi]).cuda()); opt.zero_grad(); loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad(): pv = model(Tva.cuda()).argmax(1).cpu().numpy()
                    f = f1_score(yva_, pv, average='macro')
                    if f > best[0]: best = (f, copy.deepcopy(model.state_dict()))
                model.load_state_dict(best[1]); model.eval()
                def pf(T):
                    with torch.no_grad(): return np.concatenate([torch.softmax(model(T[b:b + 512].cuda()), 1).cpu().numpy() for b in range(0, len(T), 512)])
                P_mon = pf(Xt[te_mon]); P_ua = pf(Xt[te_unm_app]); P_bg = pf(Xb[bg_te])
            else:
                continue
            C = len(mon)  # columns beyond C hold the optional "unmonitored" class; scoring uses the monitored columns only
            conf_mon = P_mon[:, :C].max(1); correct = P_mon[:, :C].argmax(1) == yte_mon
            P_unm = np.concatenate([P_ua, P_bg]); fm = flow_metrics(conf_mon, correct, P_unm[:, :C].max(1))
            P_all = np.concatenate([P_mon, P_ua, P_bg]); sid_all = np.concatenate([sid_mon, sid_unm_app, sid_bg])
            wm = window_metrics(P_all[:, :C], sid_all, wt, np.quantile(P_unm[:, :C].max(1), [0.5, 0.75, 0.9, 0.95, 0.99]), [1, 2, 3, 5, 10])
            rec = dict(tag=a.tag, dataset=ds, K=a.K, seed=seed, attacker=att, bg_class=int(a.bg_class), n_mon_apps=int(a.n_mon), monitored=[int(c) for c in mon],
                       closed_world_acc=float(correct.mean()), flow=fm, window=wm, time_s=round(time.time() - t0, 1))
            out.write(json.dumps(rec) + '\n'); out.flush(); rows.append(rec)
            f1v = fm['at_fpr1']
            print(f'{ds} K={a.K} s={seed} {att:8s} flow: TPR@1%FPR={f1v["tpr"]:.3f} prec={f1v["precision"]:.3f} (r=100 {f1v["precision_at_ratio"]["100"]:.3f}) | window: F1={wm["f1"]:.3f} TPR={wm["tpr"]:.3f} FPR={wm["fpr"]:.3f} ({wm["n_mon_win"]}/{wm["n_unm_win"]} windows)', flush=True)
    if rows:
        print('\n--- mean over seeds ---')
        for att in dict.fromkeys(r['attacker'] for r in rows):
            g = [r for r in rows if r['attacker'] == att]
            print(f'{att:8s} flow TPR@1%FPR {np.mean([r["flow"]["at_fpr1"]["tpr"] for r in g]):.3f}  prec {np.mean([r["flow"]["at_fpr1"]["precision"] for r in g]):.3f}  '
                  f'window F1 {np.mean([r["window"]["f1"] for r in g]):.3f}  TPR {np.mean([r["window"]["tpr"] for r in g]):.3f}  FPR {np.mean([r["window"]["fpr"] for r in g]):.3f}')


if __name__ == '__main__':
    main()
