"""Summarise the session-scarce comparison: reference attackers vs the proposed pre-trained metadata encoder, K=1/2/4/8, 5 seeds.
Writes paper/tables/tab_sota_fewshot.tex and prints a markdown table. Test macro-F1 (%), mean +- std over seeds; for each
reference attacker the seed-paired difference "ours - reference", marked when its two-level bootstrap interval
(results/threat_stats.json) excludes zero."""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
T = {}
for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
    r = json.loads(l)
    if r['tag'].startswith('smoke'): continue
    T[r['run_id']] = r['test']['macro_f1']  # last record wins for a run_id
KS = (1, 2, 4, 8)
# two-level paired bootstrap intervals of 'ours - attacker' (K-session draws, then test sessions; scripts/boot2.py via
# scripts/threat_stats.py); '*' marks an interval that excludes zero
STATS = json.load(open(os.path.join(ROOT, 'results', 'threat_stats.json'))) if os.path.exists(os.path.join(ROOT, 'results', 'threat_stats.json')) else {}
KEY = {'fs{K}_lgbm_meta64': 'lgbm', 'fs{K}_dfmeta64': 'df', 'fs{K}_tiktok': 'tiktok', 'fs{K}_tf': 'tf', 'fs{K}_cfpub': 'cf', 'fs{K}_netclr': 'netclr',
       'fs{K}_netclrtr': 'netclrtr', 'meta_fewshot{K}_base': 'patch', 'fs{K}_knn10': 'knn10', 'fs{K}_paper1dcnn_512B': 'cnn1d', 'fs{K}_yatc_pre': 'yatc',
       'fs{K}_etbert': 'etbert', 'fs{K}_sslscratchXL': 'scratch'}


def seeds(prefix, n=5):
    return {s: T.get(f'{prefix}_s{s}') for s in range(n)}


ROWS = [  # (label, genai run prefix fmt, ccma run prefix fmt, reads payload?)
    ('LightGBM, packet statistics', 'fs{K}_lgbm_meta64', 'ccma_fs{K}_lgbm_meta64', False),
    ('DF-style CNN~\\cite{sirinam2018df}', 'fs{K}_dfmeta64', 'ccma_fs{K}_dfmeta64', False),
    ('Tik-Tok~\\cite{rahman2020tiktok}', 'fs{K}_tiktok', 'ccma_fs{K}_tiktok', False),
    ('Triplet Fingerprinting~\\cite{sirinam2019tf}', 'fs{K}_tf', 'ccma_fs{K}_tf', False),
    ('Contrastive Fingerprinting~\\cite{xie2024cf}', 'fs{K}_cfpub', 'ccma_fs{K}_cfpub', False),
    ('NetCLR~\\cite{bahramali2023netclr}, DF backbone', 'fs{K}_netclr', 'ccma_fs{K}_netclr', False),
    ('NetCLR objective, our encoder', 'fs{K}_netclrtr', 'ccma_fs{K}_netclrtr', False),
    ('Patch-Transformer~\\cite{wang2024medformer}', 'meta_fewshot{K}_base', 'ccma_fewshot{K}_base', False),
    ('1-NN on the first 10 packets (input space)', 'fs{K}_knn10', 'ccma_fs{K}_knn10', False),
    ('1D-CNN, 512 payload bytes~\\cite{montieri2026prompts}', 'fs{K}_paper1dcnn_512B', 'ccma_fs{K}_paper1dcnn_512B', True),
    ('YaTC, pre-trained~\\cite{zhao2023yatc}', 'fs{K}_yatc_pre', 'ccma_fs{K}_yatc_pre', True),
    ('ET-BERT, pre-trained~\\cite{lin2022etbert}', 'fs{K}_etbert', None, True),
    ('Our encoder, no pre-training', 'fs{K}_sslscratchXL', 'ccma_fs{K}_sslscratchXL', False),
]
OURS = ('Pre-trained metadata encoder (ours)', 'fs{K}_sslXL', 'ccma_fs{K}_sslXL', False)


def mean_of(prefix):
    v = [x for x in seeds(prefix).values() if x is not None] if prefix else []
    return round(100 * np.mean(v), 1) if v else None


def col_best(col):  # best rounded mean among metadata-only attackers (ours included) in column col = (dataset idx, K)
    di, K = col; ms = [mean_of(r[1 + di].format(K=K) if r[1 + di] else None) for r in ROWS if not r[3]] + [mean_of(OURS[1 + di].format(K=K))]
    return max(m for m in ms if m is not None)


def cell(ref_prefix, ours_prefix, best=None, key=None):
    if ref_prefix is None: return '--'
    a = seeds(ref_prefix); b = seeds(ours_prefix)
    va = [v for v in a.values() if v is not None]
    if not va: return 'n/a'
    d = [b[s] - a[s] for s in a if a[s] is not None and b[s] is not None]
    mu = round(100 * np.mean(va), 1); txt = f'\\textbf{{{mu:.1f}}}' if best is not None and mu >= best else f'{mu:.1f}'
    if d:
        sig = key is not None and STATS.get(key[0], {}).get('attackers', {}).get(key[1], {}).get('sig', False)
        txt += f' ({100*np.mean(d):+.1f}{"$^*$" if sig else ""})'
    return txt


def sd_range(di):
    sds = []
    for r in ROWS + [OURS]:
        fmt = r[1 + di]
        if not fmt: continue
        for K in KS:
            v = [x for x in seeds(fmt.format(K=K)).values() if x is not None]
            if len(v) > 1: sds.append(100 * np.std(v))
    return min(sds), max(sds)
SDG, SDC = sd_range(0), sd_range(1)
md = ['| Attacker | payload? | ' + ' | '.join(f'GenAI K={K}' for K in KS) + ' | ' + ' | '.join(f'CCMA K={K}' for K in KS) + ' |', '|---|---|' + '---|' * 8]
L = ['\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\\caption{Session-scarce attackers under identical session-level splits: test macro-F1 (\\%, mean over 5 seeds; standard deviations over the $K$-session draws are ' + f'{SDC[0]:.1f}--{SDC[1]:.1f} points on CCMA and {SDG[0]:.1f}--{SDG[1]:.1f} on GenAI' + ') with $K$ training sessions per class. In parentheses: seed-paired difference of the pre-trained encoder over that attacker (points); $^*$: the 95\\% interval of a two-level paired bootstrap, which resamples the five $K$-session draws and then the test sessions, excludes zero (Section~\\ref{sec:scarce} gives the $K{=}1,2$ comparisons with LightGBM, the nearest neighbour and the encoder without pre-training over 20 draws). Bold: the best metadata-only attacker in each column. Attackers marked $\\bullet$ read payload bytes and are outside the metadata-only threat model. WF-style attackers are our re-implementations on the 64-packet metadata sequence (Appendix~\\ref{app:ports}).}\\label{tab:sota}',
     '\\begin{tabular}{lcccccccc}\\toprule',
     ' & \\multicolumn{4}{c}{MIRAGE-GenAI-2025 (6 classes)} & \\multicolumn{4}{c}{MIRAGE-COVID-CCMA-2022 (9 apps)}\\\\\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
     'Attacker & ' + ' & '.join(f'$K$={K}' for K in KS) + ' & ' + ' & '.join(f'$K$={K}' for K in KS) + '\\\\\\midrule']
BEST = {(di, K): col_best((di, K)) for di in (0, 1) for K in KS}
for label, g, c, payload in ROWS:
    cells = [cell(g.format(K=K), OURS[1].format(K=K), None if payload else BEST[(0, K)], (f'genai_K{K}', KEY[g])) for K in KS] + \
            [cell(c.format(K=K) if c else None, OURS[2].format(K=K), None if payload else BEST[(1, K)], (f'ccma_K{K}', KEY[g])) for K in KS]
    L.append(('$\\bullet$ ' if payload else '') + label + ' & ' + ' & '.join(cells) + '\\\\')
    md.append(f'| {label.split("~")[0]} | {"yes" if payload else "no"} | ' + ' | '.join(cells) + ' |')
L.append('\\midrule')
cells = []
for di, fmt in enumerate((OURS[1], OURS[2])):
    for K in KS:
        v = [x for x in seeds(fmt.format(K=K)).values() if x is not None]; mu = round(100 * np.mean(v), 1) if v else None
        cells.append('n/a' if mu is None else (f'\\textbf{{{mu:.1f}}}' if mu >= BEST[(di, K)] else f'{mu:.1f}') + f'$\\pm${100*np.std(v):.1f}')
L.append(OURS[0] + ' & ' + ' & '.join(cells) + '\\\\')
md.append(f'| **{OURS[0]}** | no | ' + ' | '.join(x.replace('$\\pm$', '±').replace('\\textbf{', '').replace('}', '') for x in cells) + ' |')
L += ['\\bottomrule\\end{tabular}\\end{table*}']
open(os.path.join(ROOT, 'paper', 'tables', 'tab_sota_fewshot.tex'), 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(md)); print('written tab_sota_fewshot.tex')
