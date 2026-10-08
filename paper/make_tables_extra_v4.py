"""Tables for the cross-device session-scarce attack (tab_crossdevice_v4.tex) and the pre-training corpus (tab_xcorpus_v4.tex)."""
import json, os, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); P = os.path.join(ROOT, 'paper')
T = {}
for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if not str(j.get('tag', '')).startswith('smoke'): T[j['run_id']] = j['test']['macro_f1']


def seeds(prefix, n=5): return [T[f'{prefix}_s{s}'] for s in range(n) if f'{prefix}_s{s}' in T]


def ms(v): return f'{100*np.mean(v):.1f}$\\pm${100*np.std(v):.1f}' if len(v) else 'n/a'


def m(v): return f'{100*np.mean(v):.1f}' if len(v) else 'n/a'


cd_rows = [('LightGBM, packet statistics', 'lgbm_meta64'), ('DF-style CNN', 'dfmeta64'),
           ('Nearest neighbour, first 10 packets', 'knn10'),   # runs cd8e_k4_knn10, ..., ccma_cdf8_k4_knn10 (scripts/robust_assistant_knn.py)
           ('$\\bullet$ YaTC, pre-trained', 'yatc_pre'),
           ('Our encoder, no pre-training', 'sslscratchXL'), ('Pre-trained metadata encoder (ours)', 'sslXL')]
cd_sets = [('GenAI $\\to$ Pixel', 'cd8e_k4'), ('GenAI $\\to$ Xiaomi', 'cda6_k4'), ('CCMA $\\to$ 2c', 'ccma_cd2c_k4'), ('CCMA $\\to$ f4', 'ccma_cdf4_k4'), ('CCMA $\\to$ f8', 'ccma_cdf8_k4')]
L = ['\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{6pt}\\caption{Cross-device session-scarce attack: $K$=4 sessions per class labelled on the attacker\'s phone(s), test on a phone the attacker never saw (test macro-F1 \\%, 5 seeds). $\\bullet$: reads payload bytes.}\\label{tab:cd}',
     '\\begin{tabular}{l' + 'c' * len(cd_sets) + '}\\toprule Attacker & ' + ' & '.join(n for n, _ in cd_sets) + '\\\\\\midrule']
for lab, suf in cd_rows: L.append(lab + ' & ' + ' & '.join(ms(seeds(f'{p}_{suf}')) for _, p in cd_sets) + '\\\\')
L += ['\\bottomrule\\end{tabular}\\end{table*}']
open(os.path.join(P, 'tables', 'tab_crossdevice_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L))

xc = [('none (same encoder trained from scratch)', 'fs{K}_sslscratchXL', 'ccma_fs{K}_sslscratchXL'), ('other campaign only', 'fs{K}_sslXLxc', 'ccma_fs{K}_sslXLxc'), ('both campaigns (ours)', 'fs{K}_sslXL', 'ccma_fs{K}_sslXL')]
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\caption{Pre-training corpus (test macro-F1 \\%, 5 seeds). ``Other campaign only\'\' pre-trains on the unlabelled flows of the other dataset: the attacker has never seen traffic of the target apps before fine-tuning.}\\label{tab:xcorpus}',
     '\\begin{tabular}{lcccccc}\\toprule & \\multicolumn{3}{c}{GenAI} & \\multicolumn{3}{c}{CCMA}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7} Pre-training corpus & $K$=2 & $K$=4 & $K$=8 & $K$=2 & $K$=4 & $K$=8\\\\\\midrule']
for lab, g, c in xc: L.append(lab + ' & ' + ' & '.join([m(seeds(g.format(K=K))) for K in (2, 4, 8)] + [m(seeds(c.format(K=K))) for K in (2, 4, 8)]) + '\\\\')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(P, 'tables', 'tab_xcorpus_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L)); print('extra tables written')
