"""Recompute every open-world metric of tag open_world from the saved per-flow scores (results/runs/ow_scores/open_world_*.npz) with
the shared definitions of scripts/ow_metrics.py: the tie-free calibrated ensemble, PR-AUC of the report decision, the exact
TPR at 1% FPR, and the deployable point (threshold at 5% FPR on the observer's calibration negatives).
No model is retrained. Writes results/open_world_rescored.json: {key: {scorer: {pr_auc, tpr_at1}, 'deploy': {...}, 'cw_acc', ...}}."""
import glob, json, os, re, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ow_metrics import ens_tiefree, pr_auc, tpr_at_fpr, deploy_point

OUT = {}
TAG = sys.argv[1] if len(sys.argv) > 1 else 'open_world'
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'runs', 'ow_scores', f'{TAG}_*.npz'))):
    m = re.search(r'_(genai|ccma)_K(\d)_bg(\d)_(ours|scratch|lgbm)_s(\d)\.npz$', f)
    if not m: continue
    ds, K, bg, att, s = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
    z = np.load(f); cor = z['cor_t'].astype(bool)
    names = sorted({k.split('_', 1)[1] for k in z.files if k.startswith('test_')})
    test = {n: z[f'test_{n}'] for n in names}; neg = {n: z[f'neg_{n}'] for n in names}; cal = {n: z[f'cal_{n}'] for n in names}
    test['ens'], neg['ens'], cal['ens'] = ens_tiefree([test, neg, cal], cal)
    r = {n: dict(pr_auc=pr_auc(test[n], cor, neg[n]), tpr_at1=tpr_at_fpr(test[n], cor, neg[n], 0.01)) for n in names}
    r['deploy'] = deploy_point(test['ens'], cor, neg['ens'], cal['ens'], 0.05)
    r['cw_acc'] = float(cor.mean()); r['n_mon'] = int(len(cor)); r['n_neg'] = int(len(neg['msp']))
    OUT[f'{ds}_K{K}_bg{bg}_{att}_s{s}'] = r
json.dump(OUT, open(os.path.join(ROOT, 'results', f'{TAG}_rescored.json'), 'w'), indent=1)
print(f'rescored {len(OUT)} records -> results/{TAG}_rescored.json')
for ds in ('genai', 'ccma'):
    for bg in (0, 1):
        for K in (2, 4, 8):
            line = f'{ds} bg{bg} K={K}: '
            for att in ('lgbm', 'scratch', 'ours'):
                v = [OUT[f'{ds}_K{K}_bg{bg}_{att}_s{s}'] for s in range(5) if f'{ds}_K{K}_bg{bg}_{att}_s{s}' in OUT]
                line += f'{att} AUC {100*np.mean([x["ens"]["pr_auc"] for x in v]):.1f} TPR1 {100*np.mean([x["ens"]["tpr_at1"] for x in v]):.1f} | '
            print(line)
