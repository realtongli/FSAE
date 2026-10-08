"""Open-world metrics shared by every script.
ens_tiefree: the calibrated score ensemble of scripts/open_world_reject.py (mean over parts of the empirical CDF of the
    observer's own calibration negatives) with its ties broken, without changing the order of any two distinct ensemble
    values, by the mean of the parts' raw scores z-scored with the calibration negatives' mean and std. The step ECDF makes
    the plain ensemble a step function (all scores above the largest calibration negative equal 1), and tied blocks distort
    the area under the precision-recall curve.
pr_auc: area under the precision-recall curve of the report decision; positives are correctly classified monitored flows,
    negatives are misclassified monitored flows and unmonitored flows; recall is over all monitored flows (so the curve
    ends at closed-world accuracy); ties are handled exactly (average precision).
tpr_at_fpr: exact largest TPR (correct monitored flows reported / all monitored flows) at an unmonitored FPR <= target,
    scanning every distinct threshold; like any ROC read-off, the threshold is set on the evaluated negatives.
deploy_point: threshold at the (1 - target) quantile of the calibration negatives, applied to test traffic."""
import numpy as np
from sklearn.metrics import average_precision_score

PARTS = ['energy', 'maxlogit', 'knn', 'maha', 'bgprob']


def ens_tiefree(sets, cal):
    """sets: list of dicts {score name: array}; cal: dict of the calibration negatives' scores. Returns one array per set."""
    parts = [p for p in PARTS if p in cal]
    n = min(len(cal[p]) for p in parts); P = len(parts)
    grids = {p: np.sort(cal[p]) for p in parts}
    mu = {p: float(np.mean(cal[p])) for p in parts}; sd = {p: float(np.std(cal[p])) + 1e-9 for p in parts}
    out = []
    for d in sets:
        e = np.mean([np.searchsorted(grids[p], d[p], side='right') / len(grids[p]) for p in parts], axis=0)
        z = np.mean([(d[p] - mu[p]) / sd[p] for p in parts], axis=0)
        out.append(e + 0.49 / (n * P) * np.tanh(z))   # |tie-break| < half the smallest step between distinct ensemble values
    return out


def pr_auc(score_pos, correct_pos, score_neg):
    sc = np.concatenate([score_pos, score_neg]); lab = np.concatenate([correct_pos.astype(int), np.zeros(len(score_neg), int)])
    return float(average_precision_score(lab, sc) * correct_pos.mean()) if correct_pos.any() else 0.0


def tpr_at_fpr(score_pos, correct_pos, score_neg, target=0.01):
    ts = np.unique(np.concatenate([score_pos, score_neg]))[::-1]
    neg = np.sort(score_neg); pos_c = np.sort(score_pos[correct_pos]); N, M = len(score_neg), len(score_pos)
    fp = N - np.searchsorted(neg, ts, side='left'); tp = len(pos_c) - np.searchsorted(pos_c, ts, side='left')
    ok = fp / max(N, 1) <= target
    return float(tp[ok].max() / max(M, 1)) if ok.any() else 0.0


def deploy_point(score_pos, correct_pos, score_neg, cal_neg, target=0.05):
    t = float(np.quantile(cal_neg, 1.0 - target)); r = score_pos >= t
    tp = int((r & correct_pos).sum()); fpm = int((r & ~correct_pos).sum()); fpu = int((score_neg >= t).sum())
    tpr = tp / max(len(score_pos), 1); fpr = fpu / max(len(score_neg), 1); prec = tp / max(tp + fpm + fpu, 1)
    return dict(t=t, tpr=tpr, fpr=fpr, precision=prec)
