"""Open-world attack with an explicit rejection rule, and a rigorous split of the unmonitored world.

Apps of the campaign are split three ways:
  monitored        (M apps)  the attacker labels K sessions of each,
  proxy-unmonitored(P apps)  apps the attacker also captured but does not care about; together with the background flows
                             (system services and other apps of every session) they are its *known* negatives,
  unseen-unmonitored(U apps) apps the attacker has never seen; they appear only at test time.
Training and threshold/scorer selection use monitored + known negatives; the test set contains monitored flows,
unseen-unmonitored flows and background flows of the held-out test sessions. Splits stay session-level throughout.

The class decision is always argmax of the classifier head. The report/discard decision uses a rejection score, selected
on validation among:
  msp       max softmax probability (the confidence thresholded in scripts/open_world_apps.py),
  energy    logsumexp of the logits,
  maxlogit  largest logit,
  knn       cosine similarity to the k-th nearest training embedding of the predicted class,
  maha      negative Mahalanobis distance to the class means with a shared covariance,
  bgprob    1 - probability of the explicit "unmonitored" class (only with --bg_class 1).
Writes results/open_world_reject.jsonl.
"""
import argparse, copy, json, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from train_meta import features as meta_features
from train_baselines import stat_features
from pretrain_meta import MetaEncoder
from open_world_apps import load_bg
import train_meta_ssl

L = 64
SCORERS = ['msp', 'energy', 'maxlogit', 'knn', 'maha', 'bgprob', 'ens']
ENS_PARTS = ['energy', 'maxlogit', 'knn', 'maha', 'bgprob']  # the calibrated ensemble; the maximum softmax probability is reported on its own


def add_ensemble(score_sets, ref_neg):
    """Calibrate each score by the empirical CDF of the attacker's known negatives, then average (ties broken without
    reordering, scripts/ow_metrics.ens_tiefree). ref_neg is the dict of scores on the calibration
    negatives; only those are used for calibration, never test data."""
    from ow_metrics import ens_tiefree
    if not [s for s in ENS_PARTS if s in ref_neg]: return
    ref = dict(ref_neg)
    outs = ens_tiefree(list(score_sets), ref)
    for d, e in zip(score_sets, outs): d['ens'] = e


def energy(logits):
    """Free energy of a logit vector; low energy = looks like a monitored class."""
    return -torch.logsumexp(logits, 1)


def fit(Xtr, Vtr, ytr, Xva, Vva, yva, NC, init, seed, epochs=60, Xoe=None, Voe=None, oe_lambda=0.0, m_in=-7.0, m_out=-3.0, last=False):
    torch.manual_seed(seed); np.random.seed(seed)
    if init:
        pre = torch.load(os.path.join(ROOT, init), weights_only=False); c = pre['cfg']
        enc = MetaEncoder(c['d'], c['layers'], c['heads'], c['ff'], c['L']); enc.load_state_dict(pre['encoder'])
    else:
        enc = MetaEncoder(192, 8, 8, 384, L)
    model = train_meta_ssl.Classifier(enc, NC, 0.3).cuda()
    opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': 3e-4}, {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=0.01)
    crit = nn.CrossEntropyLoss(); rng = np.random.RandomState(seed); best = (-1, None)
    Xt, Vt = torch.from_numpy(Xtr), torch.from_numpy(Vtr); Xv, Vv = torch.from_numpy(Xva), torch.from_numpy(Vva)
    use_oe = oe_lambda > 0 and Xoe is not None and len(Xoe) > 0
    if use_oe: Xo, Vo = torch.from_numpy(Xoe), torch.from_numpy(Voe)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(ytr))
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]
            if len(bi) < 2: continue
            logit_in = model(Xt[bi].cuda(), Vt[bi].cuda()); loss = crit(logit_in, torch.from_numpy(ytr[bi]).cuda())
            if use_oe:  # energy-margin (outlier-exposure) term: monitored flows below m_in, known negatives above m_out
                oi = rng.choice(len(Xo), min(len(bi), len(Xo)), replace=False)
                e_in = energy(logit_in); e_out = energy(model(Xo[oi].cuda(), Vo[oi].cuda()))
                loss = loss + oe_lambda * ((torch.relu(e_in - m_in) ** 2).mean() + (torch.relu(m_out - e_out) ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        if last:  # budget protocol: final epoch of the fixed schedule, no labelled validation sessions
            if ep == epochs - 1: best = (0.0, copy.deepcopy(model.state_dict()))
            continue
        f = f1_score(yva, forward(model, Xv, Vv)[1].argmax(1), average='macro')
        if f > best[0]: best = (f, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1]); model.eval(); return model


def forward(model, X, V, bs=512):
    """-> (embeddings [N,2d], logits [N,C])"""
    model.eval(); E, G = [], []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            x, v = X[b:b + bs].cuda(), V[b:b + bs].cuda(); e = model.enc.pool(model.enc(x, v), v)
            E.append(e.cpu().numpy()); G.append(model.head(e).cpu().numpy())
    return np.concatenate(E), np.concatenate(G)


def make_scores(E, G, ref_E, ref_y, mu_c, prec, C, k=5):
    """All rejection scores for one set of flows; higher = more likely monitored."""
    P = torch.softmax(torch.from_numpy(G[:, :C]), 1).numpy(); pred = P.argmax(1)
    s = {'msp': P.max(1), 'energy': torch.logsumexp(torch.from_numpy(G[:, :C]), 1).numpy(), 'maxlogit': G[:, :C].max(1)}
    if G.shape[1] > C:  # explicit unmonitored class
        Pf = torch.softmax(torch.from_numpy(G), 1).numpy(); s['bgprob'] = 1.0 - Pf[:, C]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); Rn = ref_E / (np.linalg.norm(ref_E, axis=1, keepdims=True) + 1e-9)
    sim = En @ Rn.T
    kk = min(k, Rn.shape[0]); s['knn'] = np.sort(sim, axis=1)[:, -kk]
    d = np.stack([np.einsum('ij,jk,ik->i', E - mu_c[c], prec, E - mu_c[c]) for c in range(C)], 1)
    s['maha'] = -d.min(1)
    return pred, s


def op_points(score_pos, correct_pos, score_neg):
    """Sweep the threshold; return (best-F1 point, point at <=1% FPR). Grid of 300 quantiles when there are more distinct
    thresholds; the reported metrics come from scripts/ow_metrics.py (exact TPR at FPR, PR-AUC) via scripts/ow_rescore.py."""
    ts = np.unique(np.concatenate([score_neg, [score_neg.min() - 1, score_neg.max() + 1]]))
    if len(ts) > 300: ts = np.quantile(ts, np.linspace(0, 1, 300))
    best = None; at1 = None
    for t in ts:
        rep = score_pos >= t; tp = int((rep & correct_pos).sum()); fp_m = int((rep & ~correct_pos).sum()); fp_u = int((score_neg >= t).sum())
        tpr = tp / max(len(score_pos), 1); fpr = fp_u / max(len(score_neg), 1); prec = tp / max(tp + fp_m + fp_u, 1)
        f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)
        p = dict(t=float(t), tpr=tpr, fpr=fpr, precision=prec, f1=f1)
        if best is None or f1 > best['f1']: best = p
        if fpr <= 0.01 and (at1 is None or tpr > at1['tpr']): at1 = p
    return best, (at1 or dict(t=float(ts[-1]), tpr=0.0, fpr=0.0, precision=0.0, f1=0.0))


def curve_metrics(score_pos, correct_pos, score_neg):
    """Threshold-free summaries of the report decision (no threshold is chosen, so nothing is selected on test data).
    pr_auc: area under the precision-recall curve of the report decision, with precision = TP / (TP + WP + FP) and
            recall = TP / #monitored test flows, where TP = monitored flow reported with its correct class, WP = monitored
            flow reported with a wrong class, FP = unmonitored flow reported; the curve therefore ends at closed-world accuracy.
    auroc:  monitored-vs-unmonitored detection AUROC of the score."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    sc = np.concatenate([score_pos, score_neg])
    lab = np.concatenate([correct_pos.astype(int), np.zeros(len(score_neg), int)])
    pr_auc = float(average_precision_score(lab, sc) * correct_pos.mean()) if correct_pos.any() else 0.0
    auroc = float(roc_auc_score(np.concatenate([np.ones(len(score_pos)), np.zeros(len(score_neg))]), sc))
    return dict(pr_auc=pr_auc, auroc=auroc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='ccma'); ap.add_argument('--target', default='app')
    ap.add_argument('--K', type=int, default=4); ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--n_mon', type=int, default=5); ap.add_argument('--n_proxy', type=int, default=2)
    ap.add_argument('--holdout_proxy', type=int, default=0, help='1 = keep the last proxy app out of the negatives used for training and use it only to pick the rejection score and threshold, so that selection sees an app the model was not trained to reject')
    ap.add_argument('--attackers', default='ours,lgbm'); ap.add_argument('--bg_class', type=int, default=1)
    ap.add_argument('--init', default='weights/meta_ssl_mpm_XL.pt'); ap.add_argument('--tag', default='ow_reject')
    ap.add_argument('--oe_lambda', type=float, default=0.0, help='weight of the energy-margin loss on known negatives during fine-tuning (0 = post-hoc rejection only)')
    ap.add_argument('--scorer', default='select', help="rejection score: 'select' picks it on validation, or fix one of SCORERS (e.g. ens)")
    ap.add_argument('--protocol', default='budget', choices=['budget', 'val'],
                    help='budget: no labelled validation sessions. Fixed schedules; the known negatives of the '
                         'K training sessions (background flows, plus the proxy apps with --bg_class 1) are split by session into a half that '
                         'trains the unmonitored class and a half that calibrates the scores; each score is thresholded at --target_fpr on that '
                         'half. val: model, calibration and threshold are selected on validation sessions.')
    ap.add_argument('--target_fpr', type=float, default=0.05); ap.add_argument('--lgbm_rounds', type=int, default=0)
    a = ap.parse_args()
    ds = a.dataset; data = GenAIData(ROOT, prefix=ds); y_all = data.y_joint if a.target == 'joint' else data.y_app
    NCfull = data.n_joint if a.target == 'joint' else data.n_app
    Xall = meta_features(data, L); Vall = (np.arange(L)[None, :] < np.minimum(data.meta_len, L)[:, None]); Fall = stat_features(data.meta, data.meta_len)
    Xbg, Vbg, bg_sid, bg_meta, bg_len = load_bg(ds); Fbg = stat_features(bg_meta, bg_len)
    # the encoder expects inputs standardised exactly as during pre-training (src/train_meta_ssl.py)
    if a.init:
        _pre = torch.load(os.path.join(ROOT, a.init), weights_only=False); MU, SD = _pre['mu'], _pre['sd']
    else:
        MU = (Xall * Vall[..., None]).sum((0, 1)) / Vall.sum(); SD = np.sqrt((((Xall - MU) ** 2) * Vall[..., None]).sum((0, 1)) / Vall.sum()) + 1e-6
    Xall = ((Xall - MU) / SD * Vall[..., None]).astype(np.float32); Xbg = ((Xbg - MU) / SD * Vbg[..., None]).astype(np.float32)
    out = open(os.path.join(ROOT, 'results', 'open_world_reject.jsonl'), 'a', encoding='utf-8'); rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        rng = np.random.RandomState(700 + seed); perm = rng.permutation(NCfull)
        mon = np.sort(perm[:a.n_mon]); proxy = np.sort(perm[a.n_mon:a.n_mon + a.n_proxy]); unseen = np.sort(perm[a.n_mon + a.n_proxy:])
        remap = {int(c): i for i, c in enumerate(mon)}; C = len(mon); BGC = C
        split = os.path.join(ROOT, 'configs', 'splits', (f'fewshot_k{a.K}_s{seed}.json' if ds == 'genai' else f'ccma_fewshot_k{a.K}_s{seed}.json'))
        idx, sm = data.split_indices(split); sp = json.load(open(split))
        sids = {k: np.array([data.file2sid[f] for f in sp[k]]) for k in ('train', 'val', 'test')}
        bgi = {k: np.where(np.isin(bg_sid, sids[k]))[0] for k in sids}
        # the attacker only holds the capture sessions it made itself: those of the monitored apps, plus those of the proxy apps
        # when it labels negatives; background flows of the other training sessions (e.g. of apps it never captured) stay out
        sess_lab = dict(zip(data.session_id.tolist(), y_all.tolist()))
        own_cls = set(int(c) for c in (np.concatenate([mon, proxy]) if a.bg_class else mon))
        own_train_sids = [s_ for s_ in sids['train'] if sess_lab.get(int(s_), -1) in own_cls]
        bg_own = np.where(np.isin(bg_sid, own_train_sids))[0]
        sel = lambda ii, cls: ii[np.isin(y_all[ii], cls)]
        tr, va = sel(idx['train'], mon), sel(idx['val'], mon)
        px_tr_apps = proxy[:-1] if (a.holdout_proxy and len(proxy) > 1) else proxy   # apps whose flows join the known negatives
        px_sel_apps = proxy[-1:] if (a.holdout_proxy and len(proxy) > 1) else proxy   # apps used to select the rejection rule
        tr_px = sel(idx['train'], px_tr_apps); va_px = sel(idx['val'], px_sel_apps)
        te_mon = sel(idx['test'], mon); te_un = sel(idx['test'], unseen)
        ytr = np.array([remap[int(c)] for c in y_all[tr]]); yva = np.array([remap[int(c)] for c in y_all[va]]); yte = np.array([remap[int(c)] for c in y_all[te_mon]])
        rb = np.random.RandomState(900 + seed)
        neg_tr = np.concatenate([Xbg[bgi['train']], Xall[tr_px]]); neg_trV = np.concatenate([Vbg[bgi['train']], Vall[tr_px]])
        cap = max(len(tr), 200)
        if len(neg_tr) > cap:
            take = rb.choice(len(neg_tr), cap, replace=False); neg_tr, neg_trV = neg_tr[take], neg_trV[take]
        neg_va = np.concatenate([Xbg[bgi['val']], Xall[va_px]]); neg_vaV = np.concatenate([Vbg[bgi['val']], Vall[va_px]])
        if len(neg_va) > 3000:
            take = rb.choice(len(neg_va), 3000, replace=False); neg_va, neg_vaV = neg_va[take], neg_vaV[take]
        va_px_raw = va_px; Fsrc_va = ('val', va_px)
        if a.protocol == 'budget':
            # the attacker's own negatives: background flows of its K training sessions, plus the proxy apps' training sessions
            # when it labels negatives; half of the sessions train the unmonitored class, the other half calibrate the scores
            pool_X = [Xbg[bg_own]]; pool_V = [Vbg[bg_own]]; pool_s = [bg_sid[bg_own]]; pool_src = [('bg', bg_own)]
            if a.bg_class:
                pool_X.append(Xall[tr_px]); pool_V.append(Vall[tr_px]); pool_s.append(data.session_id[tr_px]); pool_src.append(('px', tr_px))
            PX, PV, PS = np.concatenate(pool_X), np.concatenate(pool_V), np.concatenate(pool_s)
            us = np.unique(PS); rs = np.random.RandomState(1100 + seed); rs.shuffle(us)
            if len(us) >= 2: cal_sess = set(us[: len(us) // 2]); is_cal = np.array([x in cal_sess for x in PS])
            else: is_cal = rs.rand(len(PS)) < 0.5
            neg_va, neg_vaV = PX[is_cal], PV[is_cal]                 # calibration / threshold negatives
            fit_X, fit_V = PX[~is_cal], PV[~is_cal]
            if len(fit_X) > cap: take = rb.choice(len(fit_X), cap, replace=False); fit_X, fit_V = fit_X[take], fit_V[take]
            neg_tr, neg_trV = fit_X, fit_V                           # negatives that train the unmonitored class
            src_all = np.concatenate([np.stack([np.full(len(ii), k), ii], 1) for k, (_, ii) in enumerate(pool_src)])
            cal_src = src_all[is_cal]; fit_src_all = src_all[~is_cal]
        neg_te = np.concatenate([Xbg[bgi['test']], Xall[te_un]]); neg_teV = np.concatenate([Vbg[bgi['test']], Vall[te_un]])
        for att in a.attackers.split(','):
            t0 = time.time()
            if att in ('ours', 'scratch'):
                NCm = C + (1 if a.bg_class else 0)
                Xtr_, Vtr_, ytr_ = Xall[tr], Vall[tr], ytr; Xva_, Vva_, yva_ = Xall[va], Vall[va], yva
                if a.bg_class:
                    Xtr_ = np.concatenate([Xtr_, neg_tr]); Vtr_ = np.concatenate([Vtr_, neg_trV]); ytr_ = np.concatenate([ytr_, np.full(len(neg_tr), BGC)])
                    Xva_ = np.concatenate([Xva_, neg_va]); Vva_ = np.concatenate([Vva_, neg_vaV]); yva_ = np.concatenate([yva_, np.full(len(neg_va), BGC)])
                model = fit(Xtr_, Vtr_, ytr_, Xva_, Vva_, yva_, NCm, '' if att == 'scratch' else a.init, seed,
                            Xoe=neg_tr, Voe=neg_trV, oe_lambda=a.oe_lambda, last=a.protocol == 'budget')
                ref_E, ref_G = forward(model, torch.from_numpy(Xall[tr]), torch.from_numpy(Vall[tr]))
                mu_c = np.stack([ref_E[ytr == c].mean(0) for c in range(C)])
                cov = np.cov(np.concatenate([ref_E[ytr == c] - mu_c[c] for c in range(C)]).T) + 1e-3 * np.eye(ref_E.shape[1])
                prec = np.linalg.pinv(cov)
                sc = lambda X, V: make_scores(*forward(model, torch.from_numpy(X), torch.from_numpy(V)), ref_E, ytr, mu_c, prec, C)
                pv_pred, sv = sc(Xall[va], Vall[va]); _, sv_neg = sc(neg_va, neg_vaV)
                pt_pred, st = sc(Xall[te_mon], Vall[te_mon]); pn_pred, st_neg = sc(neg_te, neg_teV)
                cor_v = pv_pred == yva; cor_t = pt_pred == yte
            elif att == 'lgbm':
                import lightgbm as lgb
                mu_, sd_ = Fall[tr].mean(0), Fall[tr].std(0) + 1e-6; st_ = lambda F: (F - mu_) / sd_
                Fneg_tr = stat_features(np.concatenate([bg_meta[bgi['train']], data.meta[tr_px][:, :L]]), np.concatenate([bg_len[bgi['train']], np.minimum(data.meta_len[tr_px], L)]))
                Fneg_va = stat_features(np.concatenate([bg_meta[bgi['val']], data.meta[va_px][:, :L]]), np.concatenate([bg_len[bgi['val']], np.minimum(data.meta_len[va_px], L)]))
                if a.protocol == 'budget':  # same session halves as the encoder attackers
                    def stat_of(src):
                        mm = np.concatenate([bg_meta[src[src[:, 0] == 0, 1]], data.meta[src[src[:, 0] == 1, 1]][:, :L]])
                        ll = np.concatenate([bg_len[src[src[:, 0] == 0, 1]], np.minimum(data.meta_len[src[src[:, 0] == 1, 1]], L)])
                        return stat_features(mm, ll)
                    Fneg_tr, Fneg_va = stat_of(fit_src_all), stat_of(cal_src)
                Fneg_te = stat_features(np.concatenate([bg_meta[bgi['test']], data.meta[te_un][:, :L]]), np.concatenate([bg_len[bgi['test']], np.minimum(data.meta_len[te_un], L)]))
                Ftr_, ytr_ = Fall[tr], ytr; Fva_, yva_ = Fall[va], yva
                if a.bg_class:
                    if len(Fneg_tr) > cap: Fneg_tr = Fneg_tr[rb.choice(len(Fneg_tr), cap, replace=False)]
                    Ftr_ = np.concatenate([Ftr_, Fneg_tr]); ytr_ = np.concatenate([ytr_, np.full(len(Fneg_tr), BGC)])
                    Fva_ = np.concatenate([Fva_, Fneg_va]); yva_ = np.concatenate([yva_, np.full(len(Fneg_va), BGC)])
                n_est = a.lgbm_rounds if a.protocol == 'budget' else 2000
                clf = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=seed, verbose=-1)
                if a.protocol == 'budget': clf.fit(st_(Ftr_), ytr_)
                else: clf.fit(st_(Ftr_), ytr_, eval_set=[(st_(Fva_), yva_)], callbacks=[lgb.early_stopping(100, verbose=False)])
                def sc2(F):
                    P = clf.predict_proba(st_(F)); Pm = P[:, :C]; pred = Pm.argmax(1)
                    s = {'msp': Pm.max(1) / np.maximum(Pm.sum(1), 1e-9) * Pm.sum(1), 'energy': np.log(np.maximum(Pm, 1e-12)).max(1), 'maxlogit': Pm.max(1)}
                    if P.shape[1] > C: s['bgprob'] = 1.0 - P[:, C]
                    return pred, s
                pv_pred, sv = sc2(Fall[va]); _, sv_neg = sc2(Fneg_va)
                pt_pred, st2 = sc2(Fall[te_mon]); pn_pred, st_neg = sc2(Fneg_te); st = st2
                cor_v = pv_pred == yva; cor_t = pt_pred == yte
            else:
                continue
            add_ensemble([sv, sv_neg, st, st_neg], sv_neg)
            # select the scorer and threshold on validation only
            cand = [s for s in SCORERS if s in sv]

            def at_threshold(sc_name, t):
                r = st[sc_name] >= t; tp_ = int((r & cor_t).sum()); fpm = int((r & ~cor_t).sum()); fpu = int((st_neg[sc_name] >= t).sum())
                tpr_ = tp_ / max(len(cor_t), 1); fpr_ = fpu / max(len(st_neg[sc_name]), 1); pr_ = tp_ / max(tp_ + fpm + fpu, 1)
                return dict(t=float(t), tpr=tpr_, fpr=fpr_, precision=pr_, f1=2 * pr_ * tpr_ / max(pr_ + tpr_, 1e-9))
            if a.protocol == 'budget':  # threshold of every score at target_fpr on the attacker's held-out negatives
                thr = {s_: float(np.quantile(sv_neg[s_], 1.0 - a.target_fpr)) for s_ in cand}
                bestscorer = a.scorer if a.scorer in cand else 'ens'; tsel = thr[bestscorer]
                per_scorer_fixed = {s_: at_threshold(s_, thr[s_]) for s_ in cand}
            else:
                valsel = {s: op_points(sv[s], cor_v, sv_neg[s])[0] for s in cand}
                bestscorer = a.scorer if (a.scorer in valsel) else max(cand, key=lambda s: valsel[s]['f1'])
                tsel = valsel[bestscorer]['t']; per_scorer_fixed = None  # threshold chosen on validation
            rep = st[bestscorer] >= tsel; tp = int((rep & cor_t).sum()); fp_m = int((rep & ~cor_t).sum()); fp_u = int((st_neg[bestscorer] >= tsel).sum())
            tpr = tp / max(len(cor_t), 1); fpr = fp_u / max(len(st_neg[bestscorer]), 1); prec = tp / max(tp + fp_m + fp_u, 1)
            test_f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)
            per_scorer = {s: dict(best=op_points(st[s], cor_t, st_neg[s])[0], at1=op_points(st[s], cor_t, st_neg[s])[1]) for s in cand}
            per_scorer_curve = {s: curve_metrics(st[s], cor_t, st_neg[s]) for s in cand}
            sdir = os.path.join(ROOT, 'results', 'runs', 'ow_scores'); os.makedirs(sdir, exist_ok=True)
            np.savez_compressed(os.path.join(sdir, f'{a.tag}_{ds}_K{a.K}_bg{int(a.bg_class)}_{att}_s{seed}.npz'), cor_t=cor_t, pred=pt_pred, y=yte,
                                pred_neg=pn_pred, sid_mon=data.session_id[te_mon], sid_neg=np.concatenate([bg_sid[bgi['test']], data.session_id[te_un]]),
                                kind_neg=np.concatenate([np.zeros(len(bgi['test']), int), np.ones(len(te_un), int)]), yclass_neg=np.concatenate([np.full(len(bgi['test']), -1), y_all[te_un]]),
                                **{f'test_{k}': v for k, v in st.items()}, **{f'neg_{k}': v for k, v in st_neg.items()}, **{f'cal_{k}': v for k, v in sv_neg.items()})
            rec = dict(tag=a.tag, dataset=ds, K=a.K, seed=seed, attacker=att, bg_class=int(a.bg_class), oe_lambda=float(a.oe_lambda), n_mon=int(a.n_mon), n_proxy=int(a.n_proxy),
                       per_scorer_curve=per_scorer_curve, n_test_mon=int(len(cor_t)), n_test_neg=int(len(st_neg['msp'])), n_own_train_sessions=int(len(own_train_sids)),
                       monitored=[int(c) for c in mon], proxy=[int(c) for c in proxy], unseen=[int(c) for c in unseen],
                       closed_world_acc=float(cor_t.mean()), selected_scorer=bestscorer, selected_t=float(tsel),
                       test=dict(tpr=tpr, fpr=fpr, precision=prec, f1=test_f1), per_scorer_oracle=per_scorer, time_s=round(time.time() - t0, 1),
                       protocol=a.protocol, target_fpr=a.target_fpr if a.protocol == 'budget' else None, per_scorer_fixed=per_scorer_fixed,
                       n_cal_neg=int(len(neg_va)), n_fit_neg=int(len(neg_tr)))
            out.write(json.dumps(rec) + '\n'); out.flush(); rows.append(rec)
            print(f'{ds} K={a.K} s={seed} {att:7s} cw_acc={cor_t.mean():.3f} scorer={bestscorer:8s} test F1={test_f1:.3f} TPR={tpr:.3f} FPR={fpr:.3f} prec={prec:.3f} | oracle best per scorer: '
                  + ' '.join(f'{s}={per_scorer[s]["best"]["f1"]:.3f}' for s in cand), flush=True)
    if rows:
        print('\n--- mean over seeds ---')
        for att in dict.fromkeys(r['attacker'] for r in rows):
            g = [r for r in rows if r['attacker'] == att]
            print(f'{att:7s} test F1 {np.mean([r["test"]["f1"] for r in g]):.3f}  TPR {np.mean([r["test"]["tpr"] for r in g]):.3f}  FPR {np.mean([r["test"]["fpr"] for r in g]):.3f}  prec {np.mean([r["test"]["precision"] for r in g]):.3f}  '
                  + '  '.join(f'{s}:{np.mean([r["per_scorer_oracle"][s]["best"]["f1"] for r in g if s in r["per_scorer_oracle"]]):.3f}' for s in SCORERS if any(s in r['per_scorer_oracle'] for r in g)))


if __name__ == '__main__':
    main()
