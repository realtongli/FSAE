"""Recompute every open-world metric of tag open_world from the saved per-flow scores (results/runs/ow_scores/open_world_*.npz) with
the shared definitions of scripts/ow_metrics.py: the tie-free calibrated ensemble, PR-AUC of the report decision, the exact
TPR at 1% FPR, and the deployable point (threshold at 5% FPR on the observer's calibration negatives).
On GenAI the background connections that belong to a monitored assistant's own service (data/derived/
genai_background_ownservice.npy, scripts/build_bg_ownservice_mask.py) are not counted as negatives of the test traffic; the
observers, their calibration and the ensemble are unchanged.
No model is retrained. Writes results/open_world_rescored.json: {key: {scorer: {pr_auc, tpr_at1}, 'deploy': {...}, 'cw_acc', ...}}."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ow_metrics import ens_tiefree, pr_auc, tpr_at_fpr, deploy_point

DER = os.path.join(ROOT, 'data', 'derived')
OWN = np.load(os.path.join(DER, 'genai_background_ownservice.npy'))
BG = {ds: np.load(os.path.join(DER, f'{ds}_background.npz'), allow_pickle=True) for ds in ('genai', 'ccma')}
F2S = {ds: (lambda t: dict(zip(t.file, t.session_id)))(pd.read_csv(os.path.join(DER, f'{ds}_sessions.csv'))) for ds in ('genai', 'ccma')}


def bg_test_rows(ds, K, s):
    """Rows of <ds>_background.npz in the test sessions of the K-session split, in the order of open_world_reject.py."""
    sp = json.load(open(os.path.join(ROOT, 'configs', 'splits', f'{"ccma_" if ds == "ccma" else ""}fewshot_k{K}_s{s}.json')))
    return np.where(np.isin(BG[ds]['session_id'], np.array([F2S[ds][f] for f in sp['test']])))[0]


OUT = {}; NBG = {}
TAG = sys.argv[1] if len(sys.argv) > 1 else 'open_world'
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'runs', 'ow_scores', f'{TAG}_*.npz'))):
    m = re.search(r'_(genai|ccma)_K(\d)_bg(\d)_(ours|scratch|lgbm)_s(\d)\.npz$', f)
    if not m: continue
    ds, K, bg, att, s = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
    z = np.load(f); cor = z['cor_t'].astype(bool)
    names = sorted({k.split('_', 1)[1] for k in z.files if k.startswith('test_')})
    test = {n: z[f'test_{n}'] for n in names}; neg = {n: z[f'neg_{n}'] for n in names}; cal = {n: z[f'cal_{n}'] for n in names}
    test['ens'], neg['ens'], cal['ens'] = ens_tiefree([test, neg, cal], cal)
    rows = bg_test_rows(ds, K, s); nb = int((z['kind_neg'] == 0).sum())
    assert nb == len(rows) and np.array_equal(z['sid_neg'][:nb], BG[ds]['session_id'][rows]), f
    keep = np.ones(len(z['kind_neg']), bool)
    if ds == 'genai': keep[:nb] = ~OWN[rows]
    neg = {n: v[keep] for n, v in neg.items()}
    NBG.setdefault(ds, set()).add((int(keep[:nb].sum()), len(set(BG[ds]['pkg'][rows][keep[:nb]]))))
    r = {n: dict(pr_auc=pr_auc(test[n], cor, neg[n]), tpr_at1=tpr_at_fpr(test[n], cor, neg[n], 0.01)) for n in names}
    r['deploy'] = deploy_point(test['ens'], cor, neg['ens'], cal['ens'], 0.05)
    r['cw_acc'] = float(cor.mean()); r['n_mon'] = int(len(cor)); r['n_neg'] = int(len(neg['msp']))
    OUT[f'{ds}_K{K}_bg{bg}_{att}_s{s}'] = r
json.dump(OUT, open(os.path.join(ROOT, 'results', f'{TAG}_rescored.json'), 'w'), indent=1)
print(f'rescored {len(OUT)} records -> results/{TAG}_rescored.json')
for ds, v in NBG.items(): print(f'{ds}: background connections of the test sessions counted as negatives, packages: {sorted(v)}')
for ds in ('genai', 'ccma'):
    for bg in (0, 1):
        for K in (2, 4, 8):
            line = f'{ds} bg{bg} K={K}: '
            for att in ('lgbm', 'scratch', 'ours'):
                v = [OUT[f'{ds}_K{K}_bg{bg}_{att}_s{s}'] for s in range(5) if f'{ds}_K{K}_bg{bg}_{att}_s{s}' in OUT]
                line += f'{att} AUC {100*np.mean([x["ens"]["pr_auc"] for x in v]):.1f} TPR1 {100*np.mean([x["ens"]["tpr_at1"] for x in v]):.1f} | '
            print(line)
