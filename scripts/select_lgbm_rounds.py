"""Fix LightGBM's number of boosting rounds once, by the same procedure that fixed the proposed attacker's fine-tuning
recipe: compare a small grid on the labelled VALIDATION sessions of CCMA K=4, seeds 0-2, keep
the best mean validation macro-F1, and use that value unchanged for every K, both campaigns and every experiment.
Only validation metrics are computed here; no test flow is touched.
Usage: python scripts/select_lgbm_rounds.py  ->  configs/lgbm_rounds.json"""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score
import lightgbm as lgb
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from train_baselines import stat_features

GRID = [50, 100, 200, 400, 800, 1600]
data = GenAIData(ROOT, prefix='ccma'); y = data.y_app; X0 = stat_features(data.meta, data.meta_len)
val = {n: [] for n in GRID}
for s in (0, 1, 2):
    idx, _ = data.split_indices(os.path.join(ROOT, 'configs', 'splits', f'ccma_fewshot_k4_s{s}.json'))
    mu, sd = X0[idx['train']].mean(0), X0[idx['train']].std(0) + 1e-6; X = (X0 - mu) / sd
    clf = lgb.LGBMClassifier(n_estimators=max(GRID), learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                             min_child_samples=5, random_state=s, verbose=-1)
    clf.fit(X[idx['train']], y[idx['train']])
    for n in GRID:
        val[n].append(f1_score(y[idx['val']], clf.predict(X[idx['val']], num_iteration=n), average='macro'))
mean = {n: float(np.mean(v)) for n, v in val.items()}
for n in GRID: print(f'rounds={n:5d}  val macro-F1 {mean[n]:.4f}  ({" ".join(f"{x:.4f}" for x in val[n])})')
best = max(GRID, key=lambda n: (mean[n], -n))
out = dict(rounds=best, grid=GRID, val_macro_f1=mean, procedure='CCMA K=4 seeds 0-2 validation sessions, lr 0.03, as for the proposed attacker\'s recipe')
json.dump(out, open(os.path.join(ROOT, 'configs', 'lgbm_rounds.json'), 'w'), indent=1)
print('selected rounds:', best)
