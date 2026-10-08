"""Is the assistant (GenAI) or the app (CCMA) still identified under drift, an unseen phone and the defences?
Training-free: reads the saved test predictions (results/runs/<run_id>/test_preds.npz) of the pre-trained metadata encoder
('ours', *sslXL*) and of LightGBM on packet statistics ('lgbm', *lgbm_meta64*), both trained on the paper's task
(6 app x modality classes on GenAI, 9 apps on CCMA).

Analysis rule:
  GenAI (y_joint = 2*app + modality, app 0 ChatGPT, 1 Copilot, 2 Gemini):
    (a) per connection: macro-F1 of the assistant, pred//2 against y//2 over all test connections (sklearn macro average
        over the labels present in truth or prediction, zero_division=0);
    (b) per session: the assistant of a test session is the majority vote of pred//2 over that session's test connections
        (np.bincount(...).argmax(), so ties go to the lowest class index), compared with the majority true label of the
        session (every connection of a session carries the session label); reported as accuracy over the test sessions,
        exactly as in scripts/threat_stats.py.
  CCMA: the task is already the app, so (b) is computed on the 9-app prediction itself (per-connection app macro-F1 is
        kept in the JSON only).
  Every number is computed per seed (0-4) and summarised as mean and population standard deviation over the seeds;
  a cell is reported only when all five seeds exist (missing prediction files are listed, never imputed).
  Conditions: no defence K=4 (fewshot_k4, reference); temporal drift K=2/4/8 (temporal_k{K}); unseen phone (GenAI cd_8e =
  Pixel, cd_a6 = Xiaomi; CCMA cd_2c, cd_f4, cd_f8); ECH proxy (ClientHello padded in training and test); padding+jitter,
  FRONT, Tamaraw, each unaware (adv0: trained on unshaped traffic) and adaptive (adv1: trained on shaped traffic).
  References (no model): a uniform random guess (session accuracy 1/C; per-connection macro-F1 approximated by
  mean_c 2 p_c (1/C) / (p_c + 1/C) with p_c the test prevalence) and a constant guess of the most frequent class of the
  test sessions (its session accuracy).
Sanity checks: each npz's idx must equal the test indices of its split for that seed and y must equal the dataset label;
the paper-task macro-F1 recomputed from the npz is compared with the last non-smoke record of the run in
results/locked_test_metrics.jsonl (the source of the paper tables).
The Tamaraw rows use the causal Tamaraw simulation at its published configuration (scripts/wf_defenses.tamaraw).
Writes results/robust_assistant.json and paper/tables/tab_robust_assistant_v4.tex."""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData

SEEDS = range(5)
# key, table label, GenAI (ours run, lgbm run, split) or None, CCMA (ours run, lgbm run, split) or None; {s} = seed
CONDS = [
    ('none', 'No defence', ('fs4_sslXL', 'fs4_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL', 'ccma_fs4_lgbm_meta64', 'ccma_fewshot_k4')),
    ('drift_k2', 'Drift, $K$=2', ('fs2_sslXL_drift', 'fs2_drift_lgbm_meta64', 'temporal_k2'), ('ccma_fs2_sslXL_drift', 'ccma_fs2_drift_lgbm_meta64', 'ccma_temporal_k2')),
    ('drift_k4', 'Drift, $K$=4', ('fs4_sslXL_drift', 'fs4_drift_lgbm_meta64', 'temporal_k4'), ('ccma_fs4_sslXL_drift', 'ccma_fs4_drift_lgbm_meta64', 'ccma_temporal_k4')),
    ('drift_k8', 'Drift, $K$=8', ('fs8_sslXL_drift', 'fs8_drift_lgbm_meta64', 'temporal_k8'), ('ccma_fs8_sslXL_drift', 'ccma_fs8_drift_lgbm_meta64', 'ccma_temporal_k8')),
    ('phone1', 'Unseen phone 1', ('cd8e_k4_sslXL', 'cd8e_k4_lgbm_meta64', 'cd_8e_k4'), ('ccma_cd2c_k4_sslXL', 'ccma_cd2c_k4_lgbm_meta64', 'ccma_cd_2c_k4')),
    ('phone2', 'Unseen phone 2', ('cda6_k4_sslXL', 'cda6_k4_lgbm_meta64', 'cd_a6_k4'), ('ccma_cdf4_k4_sslXL', 'ccma_cdf4_k4_lgbm_meta64', 'ccma_cd_f4_k4')),
    ('phone3', 'Unseen phone 3', None, ('ccma_cdf8_k4_sslXL', 'ccma_cdf8_k4_lgbm_meta64', 'ccma_cd_f8_k4')),
    ('ech', 'ECH, ClientHello padded', ('fs4_sslXL_ech', 'fs4Lech_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_ech', 'ccma_fs4Lech_lgbm_meta64', 'ccma_fewshot_k4')),
    ('echfull', 'ECH, server padded too', ('fs4_sslXL_echfull', 'fs4Lechfull_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_echfull', 'ccma_fs4Lechfull_lgbm_meta64', 'ccma_fewshot_k4')),
    ('pj_unaware', 'Pad.+jitter, unaware', ('fs4_sslXL_pj_adv0', 'fs4_pj_adv0_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_pj_adv0', 'ccma_fs4_pj_adv0_lgbm_meta64', 'ccma_fewshot_k4')),
    ('pj_adaptive', 'Pad.+jitter, adaptive', ('fs4_sslXL_pj_adv1', 'fs4_pj_adv1_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_pj_adv1', 'ccma_fs4_pj_adv1_lgbm_meta64', 'ccma_fewshot_k4')),
    ('front_unaware', 'FRONT, unaware', ('fs4_sslXL_front_adv0', 'fs4_front_adv0_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_front_adv0', 'ccma_fs4_front_adv0_lgbm_meta64', 'ccma_fewshot_k4')),
    ('front_adaptive', 'FRONT, adaptive', ('fs4_sslXL_front_adv1', 'fs4_front_adv1_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_front_adv1', 'ccma_fs4_front_adv1_lgbm_meta64', 'ccma_fewshot_k4')),
    ('tamaraw_unaware', 'Tamaraw, unaware', ('fs4_sslXL_tamaraw_adv0', 'fs4_tamaraw_adv0_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_tamaraw_adv0', 'ccma_fs4_tamaraw_adv0_lgbm_meta64', 'ccma_fewshot_k4')),
    ('tamaraw_adaptive', 'Tamaraw, adaptive', ('fs4_sslXL_tamaraw_adv1', 'fs4_tamaraw_adv1_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_tamaraw_adv1', 'ccma_fs4_tamaraw_adv1_lgbm_meta64', 'ccma_fewshot_k4')),
]
GROUP_BREAK = {'drift_k2', 'phone1', 'ech', 'pj_unaware', 'front_unaware', 'tamaraw_unaware'}   # \midrule before these rows
DEVICE = {'genai': {'phone1': 'Pixel (8e)', 'phone2': 'Xiaomi (a6)'}, 'ccma': {'phone1': '2c', 'phone2': 'f4', 'phone3': 'f8'}}

LOCKED = {}
for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if not str(j.get('tag', '')).startswith('smoke'): LOCKED[j['run_id']] = j['test'].get('macro_f1')


def mf1(y, p): return 100 * f1_score(y, p, average='macro', zero_division=0)


def summ(v): return [float(np.mean(v)), float(np.std(v))]   # kept unrounded in memory; the table formats these (no double rounding)


def rnd(o):  # rounds floats to 2 decimals for the JSON file only
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v) for v in o]
    return o


def uniform_f1(y, C):  # expected-count approximation of a uniform random guesser's macro-F1 over the classes present in y
    p = np.bincount(y, minlength=C) / len(y); p = p[p > 0]
    return 100 * float(np.mean(2 * p / C / (p + 1 / C)))


OUT = dict(rule=__doc__.split('Analysis rule:')[1].split('Sanity checks')[0].strip(),
           seeds=list(SEEDS), missing=[], consistency=dict(max_abs_diff_vs_locked_record_pp=0.0, mismatches=[], no_record=[]))
for ds in ('genai', 'ccma'):
    D = GenAIData(ROOT, prefix=ds); ylab = D.y_joint if ds == 'genai' else D.y_app; nact = 2 if ds == 'genai' else 1
    NC = D.n_joint if ds == 'genai' else D.n_app; NA = NC // nact
    if ds == 'genai':
        lab = ylab >= 0; assert np.array_equal(ylab[lab] // 2, D.y_app[lab]) and np.array_equal(ylab[lab] % 2, D.y_act[lab]), 'y_joint != 2*app+act'
    OUT[ds] = {}
    for key, label, g, c in CONDS:
        spec = g if ds == 'genai' else c
        if spec is None: continue
        rid_ours, rid_lgbm, split = spec
        ref = dict(n_test_sessions=[], n_test_flows=[], sess_majority_acc=[], sess_uniform_acc=round(100 / NA, 2), conn_uniform_f1=[])
        res = {'ours': {}, 'lgbm': {}}
        for s in SEEDS:
            sp = os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json')
            if not os.path.exists(sp): OUT['missing'].append(f'split {split}_s{s}'); continue
            idx, _ = D.split_indices(sp); it = idx['test']; y = ylab[it]; ya = y // nact
            us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
            ys = np.array([np.bincount(y[inv == j], minlength=NC).argmax() for j in range(nS)]); ysa = ys // nact
            ref['n_test_sessions'].append(int(nS)); ref['n_test_flows'].append(int(len(it)))
            ref['sess_majority_acc'].append(100 * float(np.bincount(ysa, minlength=NA).max() / nS)); ref['conn_uniform_f1'].append(uniform_f1(ya, NA))
            for att, rid in (('ours', rid_ours), ('lgbm', rid_lgbm)):
                run = f'{rid}_s{s}'; p = os.path.join(ROOT, 'results', 'runs', run, 'test_preds.npz')
                if not os.path.exists(p): OUT['missing'].append(run); continue
                z = np.load(p); pr = z['pred'].astype(int)
                assert np.array_equal(z['idx'], it), f'{run}: idx differs from the test indices of {split}_s{s}'
                assert np.array_equal(z['y'], y), f'{run}: stored labels differ from the dataset labels'
                task_f1 = mf1(y, pr)
                if LOCKED.get(run) is None: OUT['consistency']['no_record'].append(run)
                else:
                    d = abs(task_f1 - 100 * LOCKED[run]); OUT['consistency']['max_abs_diff_vs_locked_record_pp'] = max(OUT['consistency']['max_abs_diff_vs_locked_record_pp'], round(d, 6))
                    if d > 1e-6: OUT['consistency']['mismatches'].append(dict(run=run, npz=round(task_f1, 4), record=round(100 * LOCKED[run], 4)))
                pa = pr // nact
                psa = np.array([np.bincount(pa[inv == j], minlength=NA).argmax() for j in range(nS)])   # majority vote, ties -> lowest index
                r = res[att]
                r.setdefault('conn_task_f1', []).append(task_f1)                       # GenAI 6-class / CCMA 9-app macro-F1 (paper tables)
                if ds == 'genai':
                    r.setdefault('conn_asst_f1', []).append(mf1(ya, pa))
                    r.setdefault('conn_mod_f1', []).append(mf1(y % nact, pr % nact))
                r.setdefault('sess_asst_acc', []).append(100 * accuracy_score(ysa, psa))
                r.setdefault('sess_asst_f1', []).append(mf1(ysa, psa))
                r.setdefault('sess_n_correct', []).append(int((ysa == psa).sum()))
        entry = dict(label=label, runs=dict(ours=rid_ours + '_s{0-4}', lgbm=rid_lgbm + '_s{0-4}'), split=split + '_s{0-4}',
                     n_test_sessions=ref['n_test_sessions'], n_test_flows=ref['n_test_flows'],
                     reference=dict(sess_uniform_acc=ref['sess_uniform_acc'], sess_majority_acc=summ(ref['sess_majority_acc']) if ref['sess_majority_acc'] else None,
                                    conn_uniform_f1=summ(ref['conn_uniform_f1']) if (ds == 'genai' and ref['conn_uniform_f1']) else None))
        if key in DEVICE[ds]: entry['test_phone'] = DEVICE[ds][key]
        for att in ('ours', 'lgbm'):
            r = res[att]; nseed = len(r.get('sess_asst_acc', []))
            entry[att] = dict(n_seeds=nseed, per_seed={k: [round(float(x), 2) for x in v] for k, v in r.items()})
            if nseed == len(SEEDS):
                entry[att].update({k: summ(v) for k, v in r.items() if k != 'sess_n_correct'})
        if all(entry[a]['n_seeds'] == len(SEEDS) for a in ('ours', 'lgbm')):
            dd = {}
            for k in ('conn_asst_f1', 'sess_asst_acc'):
                if k in res['ours']:
                    diff = np.array(res['ours'][k]) - np.array(res['lgbm'][k])
                    dd[k] = dict(mean=round(float(diff.mean()), 2), ours_higher=int((diff > 0).sum()), equal=int((diff == 0).sum()), lgbm_higher=int((diff < 0).sum()))
            entry['ours_minus_lgbm'] = dd
        OUT[ds][key] = entry
        o, lg = entry['ours'], entry['lgbm']
        line = f'{ds:5s} {key:17s} sess={ref["n_test_sessions"]} '
        if ds == 'genai' and 'conn_asst_f1' in o and 'conn_asst_f1' in lg:
            line += f'| task F1 L {lg["conn_task_f1"][0]:5.1f} O {o["conn_task_f1"][0]:5.1f} | asstF1 L {lg["conn_asst_f1"][0]:5.1f} O {o["conn_asst_f1"][0]:5.1f} '
        elif 'conn_task_f1' in o and 'conn_task_f1' in lg:
            line += f'| app F1 L {lg["conn_task_f1"][0]:5.1f} O {o["conn_task_f1"][0]:5.1f} '
        if 'sess_asst_acc' in o and 'sess_asst_acc' in lg:
            line += f'| sessAcc L {lg["sess_asst_acc"][0]:5.1f}±{lg["sess_asst_acc"][1]:4.1f} O {o["sess_asst_acc"][0]:5.1f}±{o["sess_asst_acc"][1]:4.1f} | majority {entry["reference"]["sess_majority_acc"][0]:5.1f}'
        print(line, flush=True)

# consistency with scripts/threat_stats.py for the no-defence reference (same runs, same rule)
ts_p = os.path.join(ROOT, 'results', 'threat_stats.json')
if os.path.exists(ts_p):
    ts = json.load(open(ts_p)); chk = {}
    for att in ('ours', 'lgbm'):
        a = ts.get('genai_K4', {}).get('attackers', {}).get(att, {})
        if 'app' in a: chk[f'genai_{att}_conn_asst_f1'] = [round(a['app'][0], 2), OUT['genai']['none'][att].get('conn_asst_f1', [None])[0]]
        if 'sess_app_acc' in a: chk[f'genai_{att}_sess_asst_acc'] = [round(a['sess_app_acc'][0], 2), OUT['genai']['none'][att].get('sess_asst_acc', [None])[0]]
        b = ts.get('ccma_K4', {}).get('attackers', {}).get(att, {})
        if 'sess_acc' in b: chk[f'ccma_{att}_sess_app_acc'] = [round(b['sess_acc'][0], 2), OUT['ccma']['none'][att].get('sess_asst_acc', [None])[0]]
    OUT['consistency']['threat_stats_K4_[threat_stats, here]'] = chk
json.dump(rnd(OUT), open(os.path.join(ROOT, 'results', 'robust_assistant.json'), 'w'), indent=1)
print('missing:', OUT['missing'] or 'none'); print('consistency:', json.dumps(OUT['consistency']))


# ---------------- LaTeX table (single column) ----------------
def cell(ds, key, att, metric):
    e = OUT[ds].get(key)
    if e is None or metric not in e[att]: return '--'
    return f'{e[att][metric][0]:.1f}'


def rng(vals, fmt='{:.0f}'):
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else fmt.format(lo) + '--' + fmt.format(hi)


gmaj = [e['reference']['sess_majority_acc'][0] for e in OUT['genai'].values() if e['reference']['sess_majority_acc']]
cmaj = [e['reference']['sess_majority_acc'][0] for e in OUT['ccma'].values() if e['reference']['sess_majority_acc']]
gunif = [e['reference']['conn_uniform_f1'][0] for e in OUT['genai'].values() if e['reference']['conn_uniform_f1']]
gns = [n for e in OUT['genai'].values() for n in e['n_test_sessions']]; cns = [n for e in OUT['ccma'].values() for n in e['n_test_sessions']]
cap = ('The assistant (GenAI) and the app (CCMA) under drift, on an unseen phone and under defences: LightGBM (LGB) and the pre-trained encoder, '
       '$K$=4 unless given (\\%, 5 seeds). Conn.\\ F1: assistant macro-F1 read off the six-class prediction; session acc.: test sessions whose '
       'assistant (app) wins the vote over their test connections (ties go to the lowest class index). Unseen phones: GenAI Pixel, Xiaomi; CCMA 2c, f4, f8. '
       f'Unaware/adaptive: trained on unshaped/shaped traffic. Random guessing: {100 / 3:.0f}\\% (GenAI) and {100 / 9:.0f}\\% (CCMA) session acc.')
L = ['\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3.5pt}',
     '\\caption{' + cap + '}\\label{tab:robust_assistant}',
     '\\begin{tabular}{lcccccc}\\toprule',
     ' & \\multicolumn{4}{c}{GenAI: assistant} & \\multicolumn{2}{c}{CCMA: app}\\\\\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}',
     ' & \\multicolumn{2}{c}{conn.\\ F1} & \\multicolumn{2}{c}{session acc.} & \\multicolumn{2}{c}{session acc.}\\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
     'Condition & LGB & ours & LGB & ours & LGB & ours\\\\\\midrule']
for key, label, _, _ in CONDS:
    if key in GROUP_BREAK: L.append('\\midrule')
    L.append(label + ' & ' + ' & '.join([cell('genai', key, 'lgbm', 'conn_asst_f1'), cell('genai', key, 'ours', 'conn_asst_f1'),
                                         cell('genai', key, 'lgbm', 'sess_asst_acc'), cell('genai', key, 'ours', 'sess_asst_acc'),
                                         cell('ccma', key, 'lgbm', 'sess_asst_acc'), cell('ccma', key, 'ours', 'sess_asst_acc')]) + '\\\\')
L += ['\\bottomrule\\end{tabular}\\end{table}']
open(os.path.join(ROOT, 'paper', 'tables', 'tab_robust_assistant_v4.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('written results/robust_assistant.json, paper/tables/tab_robust_assistant_v4.tex')
