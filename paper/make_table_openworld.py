"""Open-world tables from results/open_world_reject.jsonl, tag open_world:
  tab_openworld_v4.tex  main table: area under the precision-recall curve of the report decision (PR-AUC), a
                        threshold-free summary of the whole trade-off, same rule for every attacker;
  tab_reject_v4.tex     the same summary per rejection score (pre-trained encoder, CCMA, labelled negatives);
  tab_owdeploy_v4.tex   the deployable view: every attacker thresholds its score at 5% FPR on its own calibration negatives
                        (half of the negatives of its own capture sessions, split by session); TPR, realised FPR, precision.
No number here involves a threshold chosen on test data. TPR at 1% FPR (a point read off the test ROC curve) is printed
for the text. The attacker's negatives are the background flows of the capture sessions it made itself (monitored apps,
plus the non-monitored apps it labels when it has labelled negatives)."""
import json, os, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); P = os.path.join(ROOT, 'paper')
TAG = 'open_world'
# every metric recomputed from the saved per-flow scores with scripts/ow_metrics.py (tie-free ensemble, exact TPR@1%FPR)
RS = json.load(open(os.path.join(ROOT, 'results', f'{TAG}_rescored.json')))   # scripts/ow_rescore.py


def pick(ds, K, bg, att, get):
    v = [get(RS[k]) for k in (f'{ds}_K{K}_bg{bg}_{att}_s{s}' for s in range(5)) if k in RS]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


AUC = lambda s: (lambda r: r[s]['pr_auc'])
ROWS = [('LightGBM, packet statistics', 'lgbm'), ('Our encoder, no pre-training', 'scratch'), ('Pre-trained metadata encoder (ours)', 'ours')]
ratio = {}
for ds in ('genai', 'ccma'):
    rr = [r['n_neg'] / r['n_mon'] for k, r in RS.items() if k.startswith(ds)]
    ratio[ds] = (min(rr), max(rr)) if rr else (np.nan, np.nan)
L = ['\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\\caption{Open world: PR-AUC of the report decision (\\%, 5 seeds). The observer monitors 5 of the 9 CCMA apps or 3 of the 6 GenAI classes; '
     f'test traffic adds the never-labelled classes and the held-out sessions\' background ({ratio["genai"][0]:.1f}--{ratio["genai"][1]:.1f} unmonitored per monitored connection on GenAI, {ratio["ccma"][0]:.1f}--{ratio["ccma"][1]:.1f} on CCMA). '
     'True positive: a monitored connection reported with its correct class. Left: negatives are the observer\'s own background; right: it also labels $K$ sessions of each non-monitored class. Bold: largest area.}\\label{tab:ow}',
     '\\begin{tabular}{lcccccc|cccccc}\\toprule',
     ' & \\multicolumn{6}{c|}{negatives: own background flows} & \\multicolumn{6}{c}{plus labelled non-monitored classes}\\\\',
     ' & \\multicolumn{3}{c}{GenAI} & \\multicolumn{3}{c|}{CCMA} & \\multicolumn{3}{c}{GenAI} & \\multicolumn{3}{c}{CCMA}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}',
     'Attacker & ' + ' & '.join(['$K$=2', '$K$=4', '$K$=8'] * 4) + '\\\\\\midrule']
best = {}
for bg in (0, 1):
    for ds in ('genai', 'ccma'):
        for K in (2, 4, 8):
            best[(bg, ds, K)] = max(pick(ds, K, bg, att, AUC('ens'))[0] for _, att in ROWS)
md = ['| attacker | ' + ' | '.join(f'{b} {d} K={K}' for b in ('noneg', 'withneg') for d in ('GenAI', 'CCMA') for K in (2, 4, 8)) + ' |', '|---|' + '---|' * 12]
for lab, att in ROWS:
    cells, mdc = [], []
    for bg in (0, 1):
        for ds in ('genai', 'ccma'):
            for K in (2, 4, 8):
                m, sd, n = pick(ds, K, bg, att, AUC('ens'))
                c = f'{100*m:.1f}' if n else 'n/a'
                cells.append(f'\\textbf{{{c}}}' if n and abs(m - best[(bg, ds, K)]) < 1e-9 else c)
                t1 = pick(ds, K, bg, att, lambda r: r['ens']['tpr_at1'])[0]
                mdc.append(f'{100*m:.1f}±{100*sd:.1f} (n={n}; TPR@1%FPR on the test curve {100*t1:.1f})' if n else 'n/a')
    L.append(lab + ' & ' + ' & '.join(cells) + '\\\\'); md.append(f'| {lab} | ' + ' | '.join(mdc) + ' |')
L += ['\\bottomrule\\end{tabular}\\end{table*}']
open(os.path.join(P, 'tables', 'tab_openworld_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(md))

SC = [('max softmax', 'msp'), ('energy', 'energy'), ('max logit', 'maxlogit'), ('$k$-NN in embedding space', 'knn'), ('Mahalanobis', 'maha'), ('unmonitored-class probability', 'bgprob'), ('calibrated ensemble', 'ens')]
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\\caption{Rejection scores of the pre-trained encoder (CCMA, labelled negatives; PR-AUC of the report decision, \\%, 5 seeds; same rule as Table~\\ref{tab:ow}). '
     'The maximum softmax probability, which closed-world attacks use implicitly, is shown first.}\\label{tab:reject}',
     '\\begin{tabular}{lccc}\\toprule Rejection score & $K$=2 & $K$=4 & $K$=8\\\\\\midrule']
print()
for lab, s in SC:
    cells = []
    for K in (2, 4, 8):
        m, sd, n = pick('ccma', K, 1, 'ours', AUC(s)); cells.append(f'{100*m:.1f}' if n and not np.isnan(m) else 'n/a')
    L.append(lab + ' & ' + ' & '.join(cells) + '\\\\'); print(f'| {lab} | ' + ' | '.join(cells) + ' |')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(P, 'tables', 'tab_reject_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L))

# deployable view: threshold at 5% FPR on the attacker's own calibration negatives
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\caption{Open world at a deployable threshold ($K$=4, 5 seeds): each attacker sets its threshold at 5\\% FPR on the calibration half of its own negatives. '
     'TPR / realised FPR on the test traffic (\\%). The test traffic contains classes absent from the attacker\'s negatives, so the realised FPR departs from the 5\\% target.}\\label{tab:owdeploy}',
     '\\begin{tabular}{lcccc}\\toprule & \\multicolumn{2}{c}{own background only} & \\multicolumn{2}{c}{plus labelled non-monitored}\\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
     'Attacker & GenAI & CCMA & GenAI & CCMA\\\\\\midrule']
print()
for lab, att in ROWS:
    cells = []
    for bg in (0, 1):
        for ds in ('genai', 'ccma'):
            tp = pick(ds, 4, bg, att, lambda r: r['deploy']['tpr'])[0]; fp = pick(ds, 4, bg, att, lambda r: r['deploy']['fpr'])[0]
            pr = pick(ds, 4, bg, att, lambda r: r['deploy']['precision'])[0]
            cells.append(f'{100*tp:.1f} / {100*fp:.1f}'); print(f'{lab:32s} bg={bg} {ds}: TPR {100*tp:.1f} FPR {100*fp:.1f} prec {100*pr:.1f}')
    L.append(lab + ' & ' + ' & '.join(cells) + '\\\\')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(P, 'tables', 'tab_owdeploy_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L))
print('\nwritten tab_openworld_v4.tex, tab_reject_v4.tex, tab_owdeploy_v4.tex')

# window-level decision (scripts/ow_windows_exact.py): one device's 15-minute capture session with all its test connections
WD = json.load(open(os.path.join(ROOT, 'results', f'ow_windows_{TAG[3:]}.json')))
wpick = lambda ds, K, bg, att, key: [WD[k][key] for k in (f'{ds}_K{K}_bg{bg}_{att}_s{s}' for s in range(5)) if k in WD]
npos_g, nneg_g = np.mean(wpick('genai', 4, 0, 'ours', 'n_pos')), np.mean(wpick('genai', 4, 0, 'ours', 'n_neg'))
npos_c, nneg_c = np.mean(wpick('ccma', 4, 0, 'ours', 'n_pos')), np.mean(wpick('ccma', 4, 0, 'ours', 'n_neg'))
rng_c = lambda key: (min(wpick('ccma', 4, 0, 'ours', key)), max(wpick('ccma', 4, 0, 'ours', key)))
L = [r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{1.8pt}\caption{Open world per device window (one held-out 15-minute session with all its connections, no grouping): window PR-AUC (\%, 5 seeds) and, in parentheses, the share of monitored windows whose class (on GenAI app and modality) is named correctly. '
     r'The window takes the class whose five highest report scores have the largest mean. '
     + f'Per seed, {npos_g:.0f} monitored and {nneg_g:.0f} never-labelled windows on GenAI, {npos_c:.0f} and {nneg_c:.0f} on average on CCMA. LGB: LightGBM; scratch: no pre-training.' + r'}\label{tab:owwindow}',
     r'\begin{tabular}{llcccccc}\toprule',
     r' & & \multicolumn{3}{c}{GenAI} & \multicolumn{3}{c}{CCMA}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
     'Negatives & Observer & ' + ' & '.join(['$K$=2', '$K$=4', '$K$=8'] * 2) + r'\\\midrule']
SHORT = {'lgbm': 'LGB', 'scratch': 'scratch', 'ours': 'ours'}
from decimal import Decimal, ROUND_HALF_UP
hu = lambda x: str(Decimal(repr(round(float(x), 9))).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
print()
for bg, blab in ((0, 'own background'), (1, '+ labelled')):
    for i, (lab, att) in enumerate(ROWS):
        cells = []
        for ds in ('genai', 'ccma'):
            for K in (2, 4, 8):
                a = wpick(ds, K, bg, att, 'win_pr_auc'); c = wpick(ds, K, bg, att, 'win_correct')
                cells.append(f'{hu(100*np.mean(a))} ({hu(100*np.mean(c))})' if a else 'n/a')   # round half up, as in the text
        L.append((blab if i == 0 else '') + ' & ' + SHORT[att] + ' & ' + ' & '.join(cells) + r'\\'); print(bg, att, cells)
    if bg == 0: L.append(r'\midrule')
L += [r'\bottomrule\end{tabular}\end{table}']
open(os.path.join(P, 'tables', 'tab_owwindow_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L)); print('written tab_owwindow_v4.tex')
