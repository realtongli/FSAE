"""Corpus realism (tag dilution): the pre-trained encoder with four pre-training corpora of equal compute (optimiser steps),
fine-tuned with the fixed recipe at K=1/2/4, against LightGBM on the same sessions. For each corpus, dataset and K: mean
macro-F1 over 5 seeds and the 95% interval of 'encoder minus LightGBM' from the two-level paired bootstrap of
scripts/boot2.py (as scripts/threat_stats.py: the five K-session draws resampled with replacement, then the test sessions;
4000 resamples, the same resamples for every corpus of a cell). Also stored: the per-draw differences and, for reference
only, the session-only interval conditioning on the draws (test sessions only, 2000 resamples; 'ci_session_only').
Corpora: main = target-app flows of the train+val sessions (28k; target share 100%); dilA = + all background flows of those
sessions (no app filter; 65%); dilB = dilA + 115k MIRAGE-2019 flows (18%); dilC = dilB with a quarter of the target-app
flows (5%). Writes results/dilution_stats.json and paper/tables/tab_dilution_v4.tex."""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
import boot2

CORP = [('main', 'sslXL', 'target apps', '18/82'), ('dilA', 'sslXLdilA', '+ session background', '12/53'),   # share of each campaign's apps (GenAI/CCMA) in the corpus, by flows
        ('dilB', 'sslXLdilB', '+ 115k other flows', '3.2/15'), ('dilC', 'sslXLdilC', '1/4 of targets', '0.9/4.2')]


def macro_f1_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float); denom = cm.sum(-1) + cm.sum(-2)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0); present = denom > 0
    return (f1 * present).sum(-1) / present.sum(-1)


def load(run):
    p = os.path.join(ROOT, 'results', 'runs', run, 'test_preds.npz')
    return np.load(p) if os.path.exists(p) else None


OUT = {}
for ds, pre in (('genai', ''), ('ccma', 'ccma_')):
    D = GenAIData(ROOT, prefix=ds); NC = D.n_joint if ds == 'genai' else D.n_app
    for K in (1, 2, 4):
        runs = {'lgbm': [load(f'{pre}fs{K}_lgbm_meta64_s{s}') for s in range(5)]}
        for key, suf, _, _ in CORP: runs[key] = [load(f'{pre}fs{K}_{suf}_s{s}') for s in range(5)]
        ref = runs['lgbm'][0]; y = ref['y']; sid = D.session_id[ref['idx']]; us, inv = np.unique(sid, return_inverse=True)
        cell = boot2.Cell(y, sid, NC)  # two-level resamples of this cell (scripts/boot2.py)
        mean = {}
        for key, rr in runs.items():
            if any(r is None for r in rr): continue
            f = []
            for r in rr:
                assert np.array_equal(r['idx'], ref['idx'])
                f.append(100 * f1_score(y, r['pred'], average='macro'))
            cell.add(key, [r['pred'] for r in rr]); mean[key] = (float(np.mean(f)), float(np.std(f)))
        res = {'lgbm': mean.get('lgbm')}
        for key, *_ in CORP:
            if key not in mean: continue
            c = cell.compare(key, 'lgbm')
            res[key] = dict(f1=mean[key], diff=mean[key][0] - mean['lgbm'][0], ci=tuple(c['ci']), sig=c['sig'], per_draw=c['per_draw'],
                            ci_session_only=tuple(c['session_only']['ci']), sig_session_only=c['session_only']['sig'])
        OUT[f'{ds}_K{K}'] = res
        print(ds, K, 'LGBM %.1f' % mean['lgbm'][0], ' | '.join(f'{k} {v["f1"][0]:.1f} ({v["diff"]:+.1f} [{v["ci"][0]:+.1f},{v["ci"][1]:+.1f}]{"*" if v["sig"] else ""})' for k, v in res.items() if k != 'lgbm' and v))
json.dump(OUT, open(os.path.join(ROOT, 'results', 'dilution_stats.json'), 'w'), indent=1)
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{1.0pt}\\caption{Pre-training corpora at equal optimiser steps: test macro-F1 (\\%, 5 seeds) of the pre-trained encoder and, in parentheses, its difference from LightGBM ($^*$: two-level bootstrap interval excludes zero). '
     'Share: \\% of the corpus flows from GenAI\'s / CCMA\'s monitored apps; the other flows are session background and, in the last two rows, MIRAGE-2019, which holds 3.5k Messenger flows of 2019 (counted as CCMA target: 17\\% and 7\\%).}\\label{tab:dilution}',
     '\\begin{tabular}{lrccc|ccc}\\toprule & & \\multicolumn{3}{c|}{GenAI} & \\multicolumn{3}{c}{CCMA}\\\\ Corpus & share & $K$=1 & $K$=2 & $K$=4 & $K$=1 & $K$=2 & $K$=4\\\\\\midrule']
L.append('LightGBM (no corpus) & -- & ' + ' & '.join(f'{OUT[f"{ds}_K{K}"]["lgbm"][0]:.1f}' for ds in ('genai', 'ccma') for K in (1, 2, 4)) + '\\\\\\midrule')
for key, _, lab, share in CORP:
    cells = []
    for ds in ('genai', 'ccma'):
        for K in (1, 2, 4):
            v = OUT[f'{ds}_K{K}'].get(key)
            cells.append(f'{v["f1"][0]:.1f} ({v["diff"]:+.1f}{"$^*$" if v["sig"] else ""})' if v else 'n/a')
    L.append(f'{lab} & {share} & ' + ' & '.join(cells) + '\\\\')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(ROOT, 'paper', 'tables', 'tab_dilution_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L)); print('written tab_dilution_v4.tex')
