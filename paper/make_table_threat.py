"""Main-text table of what each metadata observer learns on MIRAGE-GenAI-2025 (tab_threat_v4.tex), from
results/threat_stats.json (scripts/threat_stats.py): the assistant (3-class) and modality (2-class) macro-F1 of the
per-connection decision, and the share of test sessions whose assistant a majority vote over the session's connections names."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); P = os.path.join(ROOT, 'paper')
S = json.load(open(os.path.join(ROOT, 'results', 'threat_stats.json')))
ROWS = [('LightGBM, packet statistics', 'lgbm', False), ('1-NN, first 10 packets', 'knn10', False), ('NetCLR, DF backbone', 'netclr', False),
        ('Pre-trained metadata encoder (ours)', 'ours', False), ('YaTC, pre-trained', 'yatc', True)]
g = lambda K, a, k: S[f'genai_K{K}']['attackers'][a][k][0]
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.4pt}\\caption{What the observer learns on MIRAGE-GenAI-2025 from $K$ labelled sessions per app$\\times$modality class (\\%, 5 seeds; '
     f'{S["genai_K4"]["n_test_sessions"]} test sessions). Per connection: macro-F1 of the assistant (3 classes, chance 33) and of the modality (text or image generation, chance 50). '
     'Per session: share of test sessions whose assistant the majority vote over the session\'s connections names correctly. $\\bullet$: reads payload bytes.}\\label{tab:threat}',
     '\\begin{tabular}{lcccc|cc|cc}\\toprule',
     ' & \\multicolumn{4}{c|}{assistant, per connection} & \\multicolumn{2}{c|}{modality} & \\multicolumn{2}{c}{assistant, session}\\\\',
     'Observer & $K$=1 & $K$=2 & $K$=4 & $K$=8 & $K$=4 & $K$=8 & $K$=1 & $K$=4\\\\\\midrule']
for lab, a, pay in ROWS:
    cells = [f'{g(K, a, "app"):.1f}' for K in (1, 2, 4, 8)] + [f'{g(K, a, "act"):.1f}' for K in (4, 8)] + [f'{g(K, a, "sess_app_acc"):.1f}' for K in (1, 4)]
    if pay: L.append('\\midrule')
    L.append(('$\\bullet$ ' if pay else '') + lab + ' & ' + ' & '.join(cells) + '\\\\'); print(f'{lab:40s}', ' '.join(cells))
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(P, 'tables', 'tab_threat_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L)); print('written tab_threat_v4.tex')
