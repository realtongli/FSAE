"""Tables for the published defences (tab_named_v4.tex) and for temporal drift (tab_drift_v4.tex)."""
import json, os, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); P = os.path.join(ROOT, 'paper')
T = {}
for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if not str(j.get('tag', '')).startswith('smoke'): T[j['run_id']] = j['test']['macro_f1']


def m(p, n=5):
    v = [T[f'{p}_s{s}'] for s in range(n) if f'{p}_s{s}' in T]
    return f'{100*np.mean(v):.1f}' if v else 'n/a'


DS = json.load(open(os.path.join(ROOT, 'results', 'defence_stats.json')))   # scripts/defence_stats.py
DG = json.load(open(os.path.join(ROOT, 'results', 'drift_gaps.json')))       # scripts/drift_gaps.py
fr = lambda k: (DS['genai']['front'][k], DS['ccma']['front'][k]); ta = lambda k: (DS['genai']['tamaraw'][k], DS['ccma']['tamaraw'][k])
TP = DS['genai']['tamaraw']['params']; assert TP == DS['ccma']['tamaraw']['params'], 'Tamaraw parameters differ between campaigns'
ATT = [('LightGBM, packet statistics', 'lgbm_meta64'), ('DF-style CNN', 'dfmeta64'), ('NetCLR', 'netclr'), ('Contrastive Fingerprinting', 'cfpub'),
       ('Nearest neighbour, first 10 packets', 'knn10'),   # runs fs4_knn10, fs4_{front,tamaraw}_adv{0,1}_knn10 (scripts/robust_assistant_knn.py)
       ('Pre-trained metadata encoder', 'sslXL')]
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\\caption{FRONT and Tamaraw over the first 64 packets ($K$=4, test macro-F1 \\%, 5 seeds). '
     f'FRONT adds {min(fr("dummies_per_real_packet")):.2f}--{max(fr("dummies_per_real_packet")):.2f} size-copying dummies per real packet ({min(fr("dummy_bytes_over_real")):.2f}--{max(fr("dummy_bytes_over_real")):.2f}$\\times$ extra bytes, no delay), displacing {100*min(fr("real_pushed_out")):.0f}--{100*max(fr("real_pushed_out")):.0f}\\% of the real packets from the window; '
     # Tamaraw as published (scripts/wf_defenses.tamaraw, causal; parameters and costs read from defence_stats.json)
     f'Tamaraw at its published rates ({int(TP["packet_bytes"])}-B packets every {TP["rho_up_ms"]:.0f}/{TP["rho_down_ms"]:.0f}~ms up/down, '
     f'each direction padded to a multiple of {TP["pad_multiple"]}; '
     f'{min(ta("extra_bytes_aggregate")):.1f}--{max(ta("extra_bytes_aggregate")):.1f}$\\times$ extra bytes, mean queueing delay {ta("delay_ms")[0]:.0f}~ms on GenAI and {ta("delay_ms")[1]:.0f}~ms on CCMA)'
     + (' sends the same first 64 packets for every connection. ' if max(ta('distinct_inputs')) == 1 else f' leaves at most {max(ta("distinct_inputs"))} distinct inputs. ') +
     'Unaware/adaptive: labelled sessions unshaped/shaped. Uniform guess: 15.9 (GenAI), 10.7 (CCMA).}\\label{tab:named}',
     '\\begin{tabular}{llccccc}\\toprule',
     ' & & & \\multicolumn{2}{c}{FRONT} & \\multicolumn{2}{c}{Tamaraw}\\\\\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
     'Data & Attacker & none & unaware & adaptive & unaware & adaptive\\\\\\midrule']
print('| data | attacker | none | FRONT unaware | FRONT adaptive | Tamaraw unaware | Tamaraw adaptive |')
for ds, pre, dsl in (('GenAI', 'fs4', 'GenAI'), ('CCMA', 'ccma_fs4', 'CCMA')):
    for i, (lab, suf) in enumerate(ATT):
        cells = [m(f'{pre}_{suf}')]
        for d in ('front', 'tamaraw'):
            for adv in (0, 1):
                cells.append(m(f'{pre}_{suf}_{d}_adv{adv}' if suf in ('sslXL', 'netclr', 'cfpub') else f'{pre}_{d}_adv{adv}_{suf}'))
        L.append((f'\\multirow{{{len(ATT)}}}{{*}}{{{dsl}}} & ' if i == 0 else ' & ') + lab + ' & ' + ' & '.join(cells) + '\\\\')
        print(f'| {dsl} | {lab} | ' + ' | '.join(cells) + ' |')
    L.append('\\midrule' if ds == 'GenAI' else '')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(P, 'tables', 'tab_named_v4.tex'), 'w', encoding='utf-8').write('\n'.join(x for x in L if x))

ROWS = [('LightGBM, packet statistics', 'drift_lgbm_meta64', 'lgbm_meta64'), ('DF-style CNN', 'drift_dfmeta64', 'dfmeta64'), ('NetCLR', 'netclr_drift', 'netclr'), ('Contrastive Fingerprinting', 'cfpub_drift', 'cfpub'),
        ('Nearest neighbour, first 10 packets', 'drift_knn10', 'knn10'),   # runs fs{K}_drift_knn10 (scripts/robust_assistant_knn.py)
        ('Our encoder, no pre-training', 'sslscratchXL_drift', 'sslscratchXL'), ('Pre-trained metadata encoder', 'sslXL_drift', 'sslXL')]
g, c = DG['genai'], DG['ccma']
BKEY = {'drift_lgbm_meta64': 'lgbm', 'drift_dfmeta64': 'df', 'netclr_drift': 'netclr', 'cfpub_drift': 'cf', 'drift_knn10': 'knn10', 'sslscratchXL_drift': 'scratch'}  # scripts/drift_gaps.py (two-level bootstrap, scripts/boot2.py)
L = ['\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{5pt}\\caption{Temporal drift: the attacker labels $K$ of the earliest sessions of each class and attacks the latest 20\\% of the sessions of each class '
     f'(test macro-F1 \\%, 5 seeds). The first attacked session follows the last labelled one by {g["first_gap_days"][0]:.0f}--{g["first_gap_days"][1]:.0f} days on GenAI and {c["first_gap_days"][0]:.0f}--{c["first_gap_days"][1]:.0f} days on CCMA; '
     'the pre-training corpus excludes the attacked sessions. In parentheses: encoder minus attacker (points); $^*$: two-level bootstrap interval excludes zero.}\\label{tab:drift}',
     '\\begin{tabular}{lcccccc}\\toprule & \\multicolumn{3}{c}{GenAI} & \\multicolumn{3}{c}{CCMA}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7} Attacker & $K$=2 & $K$=4 & $K$=8 & $K$=2 & $K$=4 & $K$=8\\\\\\midrule']
print()
for lab, dsuf, rsuf in ROWS:
    cells = []
    for pre, dsk in (('fs', 'genai'), ('ccma_fs', 'ccma')):
        for K in (2, 4, 8):
            txt = m(f"{pre}{K}_{dsuf}")
            if dsuf in BKEY:
                b = DG[dsk]['boot2_ours_minus'][f'K{K}'][BKEY[dsuf]]; txt += f' ({b["diff"]:+.1f}{"$^*$" if b["sig"] else ""})'
            cells.append(txt)
    L.append(lab + ' & ' + ' & '.join(cells) + '\\\\'); print(f'| {lab} | ' + ' | '.join(cells) + ' |')
L += ['\\bottomrule\\end{tabular}\\end{table*}']  # full width: the differences in parentheses do not fit one column
open(os.path.join(P, 'tables', 'tab_drift_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L))
print('\nwritten tab_named_v4.tex, tab_drift_v4.tex')
