"""Assistant-level robustness and defence tables with all three observers (LightGBM, the nearest neighbour on the first ten
packets, the pre-trained encoder). Cells are recomputed unrounded from results/runs/<run>/test_preds.npz with the rule of
scripts/robust_assistant.py (assistant macro-F1 read off the task prediction; session vote over the test connections, ties to
the lowest index) and printed half up; they are asserted to agree with results/robust_assistant_knn.json (whose means are
stored rounded to 2 decimals, so they are not formatted directly: 87.946 would print as 88.0).
Writes paper/tables/tab_robust3_v4.tex (main text: no defence, drift, unseen phones; label tab:robust_assistant) and
paper/tables/tab_defence3_v4.tex (appendix: no defence and the defences over the first 64 packets; label tab:def_assistant).
scripts/robust_assistant.py writes the same conditions for LightGBM and the encoder only, as the single table
tab_robust_assistant_v4.tex; the condition list CONDS is read from that script."""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
K = json.load(open(os.path.join(ROOT, 'results', 'robust_assistant_knn.json')))
A = json.load(open(os.path.join(ROOT, 'results', 'robust_assistant.json')))
for ds in ('genai', 'ccma'):
    for c in A[ds]:
        for att in ('lgbm', 'ours'):
            for m in ('conn_asst_f1', 'sess_asst_acc'):
                if m in A[ds][c][att]: assert abs(A[ds][c][att][m][0] - K[ds][c][att][m]['mean']) < 0.01, (ds, c, att, m)
_src = open(os.path.join(ROOT, 'scripts', 'robust_assistant.py'), encoding='utf-8').read()
_ns = {}; exec(_src[_src.index('CONDS = ['):_src.index('GROUP_BREAK')], _ns); CONDS = _ns['CONDS']
RES = {}
for ds in ('genai', 'ccma'):
    D = GenAIData(ROOT, prefix=ds); ylab = D.y_joint if ds == 'genai' else D.y_app; nact = 2 if ds == 'genai' else 1
    NC = D.n_joint if ds == 'genai' else D.n_app; NA = NC // nact
    for key, _, g, c in CONDS:
        spec = g if ds == 'genai' else c
        if spec is None: continue
        rid_ours, rid_lgbm, split = spec
        rid_knn = 'fs4_knn10' if key == 'none' and ds == 'genai' else ('ccma_fs4_knn10' if key == 'none' else rid_lgbm.replace('_lgbm_meta64', '_knn10'))
        rid_knn = rid_knn.replace('fs4Lechfull_', 'fs4_echfull_').replace('fs4Lech_', 'fs4_ech_')   # runs of scripts/robust_assistant_knn.py
        for att, rid in (('lgbm', rid_lgbm), ('knn', rid_knn), ('ours', rid_ours)):
            ca, sa = [], []
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json'))
                it = idx['test']; y = ylab[it]; ya = y // nact
                us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
                ysa = np.array([np.bincount(y[inv == j], minlength=NC).argmax() for j in range(nS)]) // nact
                z = np.load(os.path.join(ROOT, 'results', 'runs', f'{rid}_s{s}', 'test_preds.npz')); assert np.array_equal(z['idx'], it), (rid, s)
                pa = z['pred'].astype(int) // nact
                psa = np.array([np.bincount(pa[inv == j], minlength=NA).argmax() for j in range(nS)])
                ca.append(100 * f1_score(ya, pa, average='macro', zero_division=0)); sa.append(100 * accuracy_score(ysa, psa))
            RES[(ds, key, att)] = {'conn_asst_f1': float(np.mean(ca)), 'sess_asst_acc': float(np.mean(sa))}
            for m in ('conn_asst_f1', 'sess_asst_acc'):
                if m in K[ds][key][att]: assert abs(RES[(ds, key, att)][m] - K[ds][key][att][m]['mean']) < 0.006, (ds, key, att, m)
ATT = ('lgbm', 'knn', 'ours')


def hu(x): return f'{int(x * 10 + 0.5) / 10:.1f}'   # half up, one decimal, on unrounded means


def cells(ds, c, m):
    return [hu(RES[(ds, c, a)][m]) if (ds, c, a) in RES else '--' for a in ATT]


HEAD = ['\\begin{tabular}{l@{\\hspace{4pt}}ccc@{\\hspace{5pt}}ccc@{\\hspace{5pt}}ccc}\\toprule',
        ' & \\multicolumn{6}{c}{GenAI: assistant} & \\multicolumn{3}{c}{CCMA: app}\\\\\\cmidrule(lr){2-7}\\cmidrule(lr){8-10}',
        ' & \\multicolumn{3}{c}{conn.\\ F1} & \\multicolumn{3}{c}{session acc.} & \\multicolumn{3}{c}{session acc.}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
        'Condition & LGB & NN & ours & LGB & NN & ours & LGB & NN & ours\\\\\\midrule']
CAPC = ('LightGBM (LGB), the nearest neighbour on the first ten packets (NN) and the pre-trained encoder, $K$=4 unless given (\\%, 5 seeds). '
        'Conn.\\ F1: assistant macro-F1 read off the six-class prediction; session acc.: test sessions whose assistant (app) wins the vote over '
        'their test connections (ties go to the lowest class index). ')


def table(rows, cap, label, star=False):
    L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}', '\\caption{' + cap + '}\\label{' + label + '}'] + HEAD
    for c, lab, brk in rows:
        if brk: L.append('\\midrule')
        L.append(lab + ' & ' + ' & '.join(cells('genai', c, 'conn_asst_f1') + cells('genai', c, 'sess_asst_acc') + cells('ccma', c, 'sess_asst_acc')) + '\\\\')
    return '\n'.join(L + ['\\bottomrule\\end{tabular}\\end{table}']) + '\n'


ROB = [('none', 'No defence', False), ('drift_k2', 'Drift, $K$=2', True), ('drift_k4', 'Drift, $K$=4', False), ('drift_k8', 'Drift, $K$=8', False),
       ('phone1', 'Unseen phone 1', True), ('phone2', 'Unseen phone 2', False), ('phone3', 'Unseen phone 3', False)]
DEF = [('none', 'No defence', False), ('ech', 'ECH, ClientHello padded', True), ('echfull', 'ECH, server padded too', False),
       ('pj_unaware', 'Pad.+jitter, unaware', True), ('pj_adaptive', 'Pad.+jitter, adaptive', False),
       ('front_unaware', 'FRONT, unaware', True), ('front_adaptive', 'FRONT, adaptive', False),
       ('tamaraw_unaware', 'Tamaraw, unaware', True), ('tamaraw_adaptive', 'Tamaraw, adaptive', False)]
open(os.path.join(ROOT, 'paper', 'tables', 'tab_robust3_v4.tex'), 'w', encoding='utf-8').write(table(
    ROB, 'The assistant (GenAI) and the app (CCMA) under drift and on an unseen phone: ' + CAPC +
    'Unseen phones: GenAI Pixel, Xiaomi; CCMA 2c, f4, f8. Random guessing: 33\\% (GenAI) and 11\\% (CCMA) session acc.', 'tab:robust_assistant'))
open(os.path.join(ROOT, 'paper', 'tables', 'tab_defence3_v4.tex'), 'w', encoding='utf-8').write(table(
    DEF, 'The assistant (GenAI) and the app (CCMA) under defences over the first 64 packets: ' + CAPC +
    'Unaware/adaptive: trained on unshaped/shaped traffic. Random guessing: 33\\% (GenAI) and 11\\% (CCMA) session acc.; '
    'the most frequent app has 18\\% of the CCMA test sessions.', 'tab:def_assistant'))
print(open(os.path.join(ROOT, 'paper', 'tables', 'tab_robust3_v4.tex'), encoding='utf-8').read())
print(open(os.path.join(ROOT, 'paper', 'tables', 'tab_defence3_v4.tex'), encoding='utf-8').read())
