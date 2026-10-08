"""Window-level open world, exact (tag open_world saves the predicted class and the session of every test connection).
A window is one held-out 15-minute capture session of one device with ALL its test connections: its target-app
connections (of a monitored or a never-labelled class) and its background connections (system services, other apps);
connections are NOT grouped by app. Rule:
  for each monitored class c, s_c = mean of the 5 highest report scores (tie-free calibrated ensemble) among the window's
  connections predicted as c (fewer than 5: mean of those present; none: 0); the window's class is argmax_c s_c and its
  report score is max_c s_c.
Positive windows: held-out sessions of monitored classes (correct if the window's class is the true class); negative windows:
held-out sessions of never-labelled classes. Held-out sessions of the non-monitored classes the observer may label (proxy
classes) are not windows (their app connections are not part of the open-world test traffic).
Metrics (mean over 5 seeds): window PR-AUC (true positive = correct positive window; recall over all positive windows),
window TPR at window FPR <= 10% read off the test curve, share of positive windows named correctly, and the per-connection
PR-AUC of the same runs. Writes results/ow_windows.json."""
import glob, json, os, re, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ow_metrics import pr_auc, tpr_at_fpr
TAG = sys.argv[1] if len(sys.argv) > 1 else 'open_world'


def windows(z, C, top=5):
    score = np.concatenate([z['test_ens'], z['neg_ens']]); pred = np.concatenate([z['pred'], z['pred_neg']])
    sid = np.concatenate([z['sid_mon'], z['sid_neg']])
    pos_true = {}  # session -> true monitored class
    for s_, y_ in zip(z['sid_mon'], z['y']): pos_true[int(s_)] = int(y_)
    neg_sess = set(int(s_) for s_, k in zip(z['sid_neg'], z['kind_neg']) if k == 1)   # sessions of never-labelled classes
    out = []
    for w in sorted(set(pos_true) | neg_sess):
        m = sid == w; sc, pr = score[m], pred[m]
        s = np.zeros(C)
        for c in range(C):
            v = np.sort(sc[pr == c])[::-1][:top]
            s[c] = v.mean() if len(v) else 0.0
        c_hat = int(np.argmax(s)); out.append((w, w in pos_true, pos_true.get(w, -1) == c_hat, float(s[c_hat])))
    return out


RES = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'runs', 'ow_scores', f'{TAG}_*.npz'))):
    mm = re.search(r'_(genai|ccma)_K(\d)_bg(\d)_(ours|scratch|lgbm)_s(\d)\.npz$', f)
    ds, K, bg, att, s = mm.group(1), int(mm.group(2)), int(mm.group(3)), mm.group(4), int(mm.group(5))
    z = np.load(f); C = int(max(z['y'].max(), z['pred'].max(), z['pred_neg'].max()) + 1)
    W = windows(z, C)
    pos = np.array([w[1] for w in W]); cor = np.array([w[2] for w in W]); sc = np.array([w[3] for w in W])
    wp, wn = sc[pos], sc[~pos]; cp = cor[pos]
    RES[f'{ds}_K{K}_bg{bg}_{att}_s{s}'] = dict(win_pr_auc=pr_auc(wp, cp, wn), win_tpr_at10=tpr_at_fpr(wp, cp, wn, 0.10), win_correct=float(cp.mean()),
                                              n_pos=int(pos.sum()), n_neg=int((~pos).sum()),
                                              conn_pr_auc=pr_auc(z['test_ens'], z['cor_t'].astype(bool), z['neg_ens']))
json.dump(RES, open(os.path.join(ROOT, 'results', f'ow_windows_{TAG[3:]}.json'), 'w'), indent=1)
for ds in ('genai', 'ccma'):
    for bg in (0, 1):
        for K in (2, 4, 8):
            line = f'{ds} bg{bg} K={K}:'
            for att in ('lgbm', 'scratch', 'ours'):
                v = [RES[k] for k in (f'{ds}_K{K}_bg{bg}_{att}_s{s}' for s in range(5)) if k in RES]
                if v: line += f' | {att} winAUC {100*np.mean([x["win_pr_auc"] for x in v]):.1f}±{100*np.std([x["win_pr_auc"] for x in v]):.1f} TPR@10 {100*np.mean([x["win_tpr_at10"] for x in v]):.1f} corr {100*np.mean([x["win_correct"] for x in v]):.1f} (conn {100*np.mean([x["conn_pr_auc"] for x in v]):.1f}) n+{np.mean([x["n_pos"] for x in v]):.0f} n-{np.mean([x["n_neg"] for x in v]):.0f}'
            print(line)
print('written', f'results/ow_windows_{TAG[3:]}.json')
