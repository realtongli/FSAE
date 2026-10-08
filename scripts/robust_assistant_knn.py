"""The training-free nearest neighbour (knn10) on the drift, unseen-phone and defence settings of the assistant-level robustness table,
and its six-class (GenAI) / nine-app (CCMA) macro-F1 for Tables drift, crossdevice, named and mitig.

Stage 'run' produces the runs, stage 'analyse' scores them; default: both.

Observer (identical to the knn10 of the main tables, scripts/trivial_baselines.py): features = trivial_baselines.feats_knn
(first 10 packets: direction, log1p IP bytes, log1p payload bytes, log1p IAT in ms, zero beyond the flow length);
1-NN under L1 distance against the connections of the K labelled training sessions of the split only (idx['train']);
ties go to the lowest training index (np.argmin), batch 512, float32, exactly the nn1() of trivial_baselines.main.
Nothing is trained, tuned or selected; the validation sessions are not used.

Defences are shaped exactly as the LightGBM runs of the table (scripts/train_baselines.py --defense X --defense_train D
--seeds s, one seed per call): the shaping function is applied to the full meta array of the dataset with the run's
seed (wf_defenses.apply_named for front/tamaraw/ech/ech_full, which also changes the per-flow length;
train_meta.apply_defense for pad256_jitter20); D=0 (unaware): the labelled sessions stay unshaped and the test
sessions are shaped; D=1 (adaptive): both shaped. ECH (ClientHello padded) and ECH-full (server padded too) are D=1,
as their LightGBM runs are. Tamaraw is the causal simulation at its published configuration (wf_defenses.tamaraw:
1500-B packets, 40/12 ms, L = 100). Run ids follow the LightGBM prefixes with '_knn10' in place of '_lgbm_meta64'
(fs2_drift_knn10_s0, cd8e_k4_knn10_s0, fs4_front_adv0_knn10_s0, fs4_ech_knn10_s0, ...; CCMA with 'ccma_').
Each run appends one record to results/locked_test_metrics.jsonl (tag knn_robust, same fields as
train_baselines.log_rec) and saves results/runs/<run_id>/test_preds.npz (idx, y, pred). The no-defence reference is
the existing fs4_knn10 / ccma_fs4_knn10 run (tag trivial): it is recomputed with the same code, checked to equal its
locked record, and only its prediction file is written (no new record).

Analysis rule:
  the rule of scripts/robust_assistant.py, unchanged, applied to the 1-NN (and, for paired reference only, recomputed
  for LightGBM and the pre-trained encoder from their saved predictions, whose run ids are read from the CONDS list of
  scripts/robust_assistant.py itself):
    GenAI: (a) per connection, assistant macro-F1 of pred//2 against y//2 over all test connections (sklearn macro over
    the labels present in truth or prediction, zero_division=0); (b) per session, the assistant of a test session is
    the majority vote of pred//2 over its test connections (np.bincount(...).argmax(), ties to the lowest index),
    compared with the session's majority true label, reported as accuracy over the test sessions. CCMA: (b) on the
    9-app prediction itself. The task macro-F1 (6 classes / 9 apps) is the paper-table number.
  Every number is computed per seed (0-4) and summarised as mean, population standard deviation and seed range
  (min, max); a cell is reported only when all five seeds exist. Paired differences (1-NN minus LightGBM, 1-NN minus
  encoder) are reported per seed with their mean and the count of seeds on each side; no significance test is claimed.
  All conditions of the assistant-level robustness table are reported, favourable or not.
Writes results/robust_assistant_knn.json.
Run log (key 'run_log' of the JSON): run_log.overheads is keyed by dataset (run_log.overheads.genai / .ccma) and then by
'<defence>_s<seed>'. No record is appended when the identical record (same tag, cfg_hash and macro-F1) is already the
current record of the run (run_log.skipped_existing); run_log.locked_records_with_tag counts the records of the tag in
the locked file.
Options: --only restricts stage 'run' to a comma list of conditions (keys of KNN; default: all) and --tag sets the tag of
the appended records (default knn_robust). The last record of a run id in the locked file is its current record, and
the analysis reads the current record of every run. When stage 'run' is restricted by --only or not executed, the run log
stored in the output file (its 'run_log_previous' if present, else its 'run_log') is carried over under
'run_log_previous'."""
import argparse, ast, hashlib, json, os, sys, time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds
from trivial_baselines import feats_knn          # the knn10 feature map of the main tables

SEEDS = list(range(5))
TAG = 'knn_robust'
LOCKED_PATH = os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl')

# key -> GenAI (knn run prefix, split, defence, defense_train) or None, CCMA (...) or None.  Same keys, splits and
# defence settings as the LightGBM runs listed in scripts/robust_assistant.py.
KNN = {
    'none': (('fs4', 'fewshot_k4', 'none', 0), ('ccma_fs4', 'ccma_fewshot_k4', 'none', 0)),
    'drift_k2': (('fs2_drift', 'temporal_k2', 'none', 0), ('ccma_fs2_drift', 'ccma_temporal_k2', 'none', 0)),
    'drift_k4': (('fs4_drift', 'temporal_k4', 'none', 0), ('ccma_fs4_drift', 'ccma_temporal_k4', 'none', 0)),
    'drift_k8': (('fs8_drift', 'temporal_k8', 'none', 0), ('ccma_fs8_drift', 'ccma_temporal_k8', 'none', 0)),
    'phone1': (('cd8e_k4', 'cd_8e_k4', 'none', 0), ('ccma_cd2c_k4', 'ccma_cd_2c_k4', 'none', 0)),
    'phone2': (('cda6_k4', 'cd_a6_k4', 'none', 0), ('ccma_cdf4_k4', 'ccma_cd_f4_k4', 'none', 0)),
    'phone3': (None, ('ccma_cdf8_k4', 'ccma_cd_f8_k4', 'none', 0)),
    'ech': (('fs4_ech', 'fewshot_k4', 'ech', 1), ('ccma_fs4_ech', 'ccma_fewshot_k4', 'ech', 1)),
    'echfull': (('fs4_echfull', 'fewshot_k4', 'ech_full', 1), ('ccma_fs4_echfull', 'ccma_fewshot_k4', 'ech_full', 1)),
    'pj_unaware': (('fs4_pj_adv0', 'fewshot_k4', 'pad256_jitter20', 0), ('ccma_fs4_pj_adv0', 'ccma_fewshot_k4', 'pad256_jitter20', 0)),
    'pj_adaptive': (('fs4_pj_adv1', 'fewshot_k4', 'pad256_jitter20', 1), ('ccma_fs4_pj_adv1', 'ccma_fewshot_k4', 'pad256_jitter20', 1)),
    'front_unaware': (('fs4_front_adv0', 'fewshot_k4', 'front', 0), ('ccma_fs4_front_adv0', 'ccma_fewshot_k4', 'front', 0)),
    'front_adaptive': (('fs4_front_adv1', 'fewshot_k4', 'front', 1), ('ccma_fs4_front_adv1', 'ccma_fewshot_k4', 'front', 1)),
    'tamaraw_unaware': (('fs4_tamaraw_adv0', 'fewshot_k4', 'tamaraw', 0), ('ccma_fs4_tamaraw_adv0', 'ccma_fewshot_k4', 'tamaraw', 0)),
    'tamaraw_adaptive': (('fs4_tamaraw_adv1', 'fewshot_k4', 'tamaraw', 1), ('ccma_fs4_tamaraw_adv1', 'ccma_fewshot_k4', 'tamaraw', 1)),
}


def robust_conds():
    """CONDS of scripts/robust_assistant.py, read with ast (importing it would execute it and rewrite its outputs)."""
    src = open(os.path.join(ROOT, 'scripts', 'robust_assistant.py'), encoding='utf-8').read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'CONDS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('CONDS not found in robust_assistant.py')


def read_locked():
    L = {}
    for l in open(LOCKED_PATH, encoding='utf-8'):
        j = json.loads(l)
        if not str(j.get('tag', '')).startswith('smoke'): L[j['run_id']] = j
    return L


def nn1(Ftr, ytr, Q):  # verbatim nn1 of scripts/trivial_baselines.py
    out = np.empty(len(Q), int)
    for b in range(0, len(Q), 512): out[b:b + 512] = ytr[np.abs(Q[b:b + 512, None, :] - Ftr[None]).sum(-1).argmin(1)]
    return out


def shaped_meta(D, ds, defense, seed, cache, overheads):
    """Shaping as scripts/train_baselines.py: on the full meta array, with the run's seed. The overhead that
    apply_named reports is logged under overheads[ds]['<defence>_s<seed>'] (keyed by dataset, then by defence and
    seed)."""
    key = (defense, seed)
    if key not in cache:
        from train_meta import apply_defense, NAMED_DEFENSES
        if defense in NAMED_DEFENSES:
            from wf_defenses import apply_named
            m, ln, ov = apply_named(defense, D.meta, D.meta_len, seed); overheads.setdefault(ds, {})[f'{defense}_s{seed}'] = ov
        else:
            m = apply_defense(D.meta, defense, seed); ln = D.meta_len
        cache.clear(); cache[key] = (m, ln)   # keep one shaped copy in memory (unaware and adaptive share it)
    return cache[key]


def stage_run(seeds, only=None, tag=None):
    """only: optional list of conditions (KNN keys) to run; tag: tag of the appended records (default TAG)."""
    TAG = tag or globals()['TAG']
    LOCKED = read_locked(); log = dict(overheads={}, reproduced_reference=[], appended=[], skipped_existing=[], only=only, tag=TAG)
    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); tgt = 'joint' if ds == 'genai' else 'app'
        y = D.y_joint if tgt == 'joint' else D.y_app
        NC = D.n_joint if tgt == 'joint' else D.n_app; n_act = D.n_act if tgt == 'joint' else 1
        cache = {}
        order = sorted([(k, v[0 if ds == 'genai' else 1]) for k, v in KNN.items() if v[0 if ds == 'genai' else 1] is not None and (not only or k in only)],
                       key=lambda kv: (kv[1][2], kv[0]))   # group by defence so each shaped copy is built once per seed
        for key, (prefix, split, defense, dtrain) in order:
            for s in seeds:
                sp = os.path.join('configs', 'splits', f'{split}_s{s}.json')
                idx, sm = D.split_indices(os.path.join(ROOT, sp)); tr, it = idx['train'], idx['test']
                t0 = time.time()
                if defense == 'none':
                    Mtr, Ltr, Mte, Lte = D.meta, D.meta_len, D.meta, D.meta_len
                else:
                    ms, ls = shaped_meta(D, ds, defense, s, cache, log['overheads'])
                    Mte, Lte = ms, ls
                    Mtr, Ltr = (ms, ls) if dtrain else (D.meta, D.meta_len)
                Ftr = feats_knn(Mtr[tr], Ltr[tr]); Fte = feats_knn(Mte[it], Lte[it])
                pt = nn1(Ftr, y[tr], Fte)
                run = f'{prefix}_knn10_s{s}'
                tm = metrics_from_preds(y[it], pt, sessions=D.session_id[it], seed=s, n_classes=NC, n_act=n_act)
                d = os.path.join(ROOT, 'results', 'runs', run); os.makedirs(d, exist_ok=True)
                np.savez(os.path.join(d, 'test_preds.npz'), y=y[it], pred=pt, idx=it)
                if key == 'none':   # existing reference run: must reproduce its locked record exactly; no new record
                    rec = LOCKED.get(run)
                    assert rec is not None, f'{run}: no locked record'
                    assert rec['split_hash'] == sm['hash'], f'{run}: split hash differs'
                    diff = abs(rec['test']['macro_f1'] - tm['macro_f1'])
                    assert diff < 1e-12, f'{run}: recomputed macro-F1 {tm["macro_f1"]} != record {rec["test"]["macro_f1"]}'
                    log['reproduced_reference'].append(dict(run=run, macro_f1=tm['macro_f1'], abs_diff=diff))
                else:
                    cfg = dict(model='knn10', feats='first10_log_L1_1nn', dataset=ds, target=tgt, split=sp, defense=defense, defense_train=dtrain)
                    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
                    prev = LOCKED.get(run)
                    if prev is not None and prev.get('tag') == TAG and prev.get('cfg_hash') == h and abs(prev['test']['macro_f1'] - tm['macro_f1']) < 1e-12:
                        log['skipped_existing'].append(run)   # idempotent: the identical record is already the current one
                    else:
                        with open(LOCKED_PATH, 'a', encoding='utf-8') as fh:
                            fh.write(json.dumps(dict(run_id=run, cfg_hash=h, seed=s, tag=TAG, split_hash=sm['hash'], test=tm)) + '\n')
                        log['appended'].append(run)
                print(f'{run:32s} {defense:15s} D={dtrain} train {len(tr):5d} test {len(it):6d}  macro_f1 {100 * tm["macro_f1"]:.2f}  ({time.time() - t0:.1f}s)', flush=True)
    # state of the locked file after this call (nothing is appended when every record is already current)
    L = read_locked(); log['locked_records_with_tag'] = sum(1 for r in L.values() if r.get('tag') == TAG)
    return log


# ------------------------------------------------------------ analysis (robust_assistant.py rule) ----
def mf1(y, p): return 100 * f1_score(y, p, average='macro', zero_division=0)


def summ(v): return dict(mean=float(np.mean(v)), std=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)))


def rnd(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v) for v in o]
    return o


def uniform_f1(y, C):
    p = np.bincount(y, minlength=C) / len(y); p = p[p > 0]
    return 100 * float(np.mean(2 * p / C / (p + 1 / C)))


def stage_analyse(run_log):
    LOCKED = read_locked(); CONDS = robust_conds()
    OUT = dict(rule=__doc__.split('Analysis rule:')[1].split('Writes results')[0].strip(),
               observer='knn10: 1-NN, L1, first 10 packets (scripts/trivial_baselines.py), reference set = the K labelled sessions per class',
               seeds=SEEDS, tag=TAG, run_log=run_log, missing=[],
               consistency=dict(npz_vs_locked_max_abs_diff_pp=0.0, mismatches=[], no_record=[], vs_robust_assistant_json=[]))
    prev_p = os.path.join(ROOT, 'results', 'robust_assistant_knn.json')
    if os.path.exists(prev_p) and (not run_log or run_log.get('only')):   # partial or analyse-only call: carry over the run log stored in the output file
        prev = json.load(open(prev_p)); OUT['run_log_previous'] = prev.get('run_log_previous') or prev.get('run_log')
    OUT['current_tags'] = sorted({str(LOCKED[r].get('tag')) for r in LOCKED if r.endswith(tuple(f'_knn10_s{s}' for s in SEEDS)) and any(r.startswith(k[0] + '_knn10') for v in KNN.values() for k in v if k)})
    ra = json.load(open(os.path.join(ROOT, 'results', 'robust_assistant.json'))) if os.path.exists(os.path.join(ROOT, 'results', 'robust_assistant.json')) else {}
    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); ylab = D.y_joint if ds == 'genai' else D.y_app; nact = 2 if ds == 'genai' else 1
        NC = D.n_joint if ds == 'genai' else D.n_app; NA = NC // nact
        OUT[ds] = {}
        for key, label, g, c in CONDS:
            spec = g if ds == 'genai' else c
            kspec = KNN[key][0 if ds == 'genai' else 1]
            if spec is None or kspec is None: continue
            rid_ours, rid_lgbm, split = spec
            assert split == kspec[1], f'{key}: split differs from robust_assistant.py'
            rid = {'knn': kspec[0] + '_knn10', 'lgbm': rid_lgbm, 'ours': rid_ours}
            ref = dict(n_test_sessions=[], n_test_flows=[], sess_majority_acc=[], conn_uniform_f1=[])
            res = {a: {} for a in rid}
            for s in SEEDS:
                sp = os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json')
                idx, _ = D.split_indices(sp); it = idx['test']; y = ylab[it]; ya = y // nact
                us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
                ys = np.array([np.bincount(y[inv == j], minlength=NC).argmax() for j in range(nS)]); ysa = ys // nact
                ref['n_test_sessions'].append(int(nS)); ref['n_test_flows'].append(int(len(it)))
                ref['sess_majority_acc'].append(100 * float(np.bincount(ysa, minlength=NA).max() / nS)); ref['conn_uniform_f1'].append(uniform_f1(ya, NA))
                for att, r0 in rid.items():
                    run = f'{r0}_s{s}'; p = os.path.join(ROOT, 'results', 'runs', run, 'test_preds.npz')
                    if not os.path.exists(p): OUT['missing'].append(run); continue
                    z = np.load(p); pr = z['pred'].astype(int)
                    assert np.array_equal(z['idx'], it), f'{run}: idx differs from the test indices of {split}_s{s}'
                    assert np.array_equal(z['y'], y), f'{run}: stored labels differ from the dataset labels'
                    task_f1 = mf1(y, pr)
                    if run not in LOCKED: OUT['consistency']['no_record'].append(run)
                    else:
                        dlt = abs(task_f1 - 100 * LOCKED[run]['test']['macro_f1'])
                        OUT['consistency']['npz_vs_locked_max_abs_diff_pp'] = max(OUT['consistency']['npz_vs_locked_max_abs_diff_pp'], dlt)
                        if dlt > 1e-6: OUT['consistency']['mismatches'].append(dict(run=run, npz=task_f1, record=100 * LOCKED[run]['test']['macro_f1']))
                    pa = pr // nact
                    psa = np.array([np.bincount(pa[inv == j], minlength=NA).argmax() for j in range(nS)])
                    r = res[att]
                    r.setdefault('conn_task_f1', []).append(task_f1)
                    if ds == 'genai':
                        r.setdefault('conn_asst_f1', []).append(mf1(ya, pa))
                        r.setdefault('conn_mod_f1', []).append(mf1(y % nact, pr % nact))
                    r.setdefault('sess_asst_acc', []).append(100 * accuracy_score(ysa, psa))
                    r.setdefault('sess_asst_f1', []).append(mf1(ysa, psa))
                    r.setdefault('sess_n_correct', []).append(int((ysa == psa).sum()))
            entry = dict(label=label, split=split + '_s{0-4}', runs={a: r0 + '_s{0-4}' for a, r0 in rid.items()},
                         defence=kspec[2], defense_train=kspec[3],
                         n_test_sessions=ref['n_test_sessions'], n_test_flows=ref['n_test_flows'],
                         reference=dict(sess_uniform_acc=100 / NA, sess_majority_acc=summ(ref['sess_majority_acc']),
                                        conn_uniform_f1=summ(ref['conn_uniform_f1']) if ds == 'genai' else None))
            for att in rid:
                r = res[att]; ns = len(r.get('sess_asst_acc', []))
                entry[att] = dict(n_seeds=ns, per_seed={k: [float(x) for x in v] for k, v in r.items()})
                if ns == len(SEEDS): entry[att].update({k: summ(v) for k, v in r.items() if k != 'sess_n_correct'})
            for other in ('lgbm', 'ours'):
                if entry['knn']['n_seeds'] == len(SEEDS) and entry[other]['n_seeds'] == len(SEEDS):
                    dd = {}
                    for k in ('conn_task_f1', 'conn_asst_f1', 'sess_asst_acc'):
                        if k in res['knn']:
                            diff = np.array(res['knn'][k]) - np.array(res[other][k])
                            dd[k] = dict(per_seed=[float(x) for x in diff], mean=float(diff.mean()), min=float(diff.min()), max=float(diff.max()),
                                         knn_higher=int((diff > 0).sum()), equal=int((diff == 0).sum()), knn_lower=int((diff < 0).sum()))
                    entry[f'knn_minus_{other}'] = dd
            # the LightGBM / encoder numbers recomputed here must equal results/robust_assistant.json
            prev = ra.get(ds, {}).get(key, {})
            for att in ('lgbm', 'ours'):
                for k in ('conn_task_f1', 'conn_asst_f1', 'sess_asst_acc'):
                    if k in prev.get(att, {}) and k in entry[att]:
                        a0, a1 = round(prev[att][k][0], 2), round(entry[att][k]['mean'], 2)
                        if abs(a0 - a1) > 0.011: OUT['consistency']['vs_robust_assistant_json'].append(dict(ds=ds, key=key, att=att, metric=k, json=a0, here=a1))
            OUT[ds][key] = entry
            kn = entry['knn']
            line = f'{ds:5s} {key:17s} '
            if 'conn_task_f1' in kn: line += f'task F1 {kn["conn_task_f1"]["mean"]:5.1f}±{kn["conn_task_f1"]["std"]:4.1f} '
            if 'conn_asst_f1' in kn: line += f'| asst F1 {kn["conn_asst_f1"]["mean"]:5.1f} [{kn["conn_asst_f1"]["min"]:.1f},{kn["conn_asst_f1"]["max"]:.1f}] '
            if 'sess_asst_acc' in kn: line += f'| sess {kn["sess_asst_acc"]["mean"]:5.1f} [{kn["sess_asst_acc"]["min"]:.1f},{kn["sess_asst_acc"]["max"]:.1f}] '
            for other in ('lgbm', 'ours'):
                if 'sess_asst_acc' in entry[other]: line += f'| {other} sess {entry[other]["sess_asst_acc"]["mean"]:5.1f} '
            print(line, flush=True)

    # six-class / nine-app macro-F1 of the 1-NN laid out as the paper tables (value = mean, std, min, max over seeds)
    def cell(ds, key):
        e = OUT[ds].get(key)
        return None if e is None or 'conn_task_f1' not in e['knn'] else e['knn']['conn_task_f1']
    OUT['tables'] = {
        'tab:drift (temporal drift, test macro-F1)': {f'{ds}_K{K}': cell(ds, f'drift_k{K}') for ds in ('genai', 'ccma') for K in (2, 4, 8)},
        'tab:cd (unseen phone, K=4)': {'GenAI->Pixel(8e)': cell('genai', 'phone1'), 'GenAI->Xiaomi(a6)': cell('genai', 'phone2'),
                                       'CCMA->2c': cell('ccma', 'phone1'), 'CCMA->f4': cell('ccma', 'phone2'), 'CCMA->f8': cell('ccma', 'phone3')},
        'tab:named (K=4)': {f'{ds}_{c}': cell(ds, c) for ds in ('genai', 'ccma') for c in ('none', 'front_unaware', 'front_adaptive', 'tamaraw_unaware', 'tamaraw_adaptive')},
        'tab:mitig (K=4)': {f'{ds}_{c}': cell(ds, c) for ds in ('genai', 'ccma') for c in ('none', 'pj_unaware', 'pj_adaptive')},
        'ECH (K=4, not in a macro-F1 table)': {f'{ds}_{c}': cell(ds, c) for ds in ('genai', 'ccma') for c in ('ech', 'echfull')},
    }
    OUT['consistency']['npz_vs_locked_max_abs_diff_pp'] = float(OUT['consistency']['npz_vs_locked_max_abs_diff_pp'])
    json.dump(rnd(OUT), open(os.path.join(ROOT, 'results', 'robust_assistant_knn.json'), 'w'), indent=1)
    print('missing:', OUT['missing'] or 'none'); print('consistency:', json.dumps(rnd(OUT['consistency'])))
    print('written results/robust_assistant_knn.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='all', choices=['all', 'run', 'analyse'])
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--only', default='', help='optional comma list of conditions (KNN keys) to run (default: all)')
    ap.add_argument('--tag', default='', help='tag of the appended records (default: TAG)'); a = ap.parse_args()
    log = {}   # the run log (defence overheads, reproduced reference, appended records) goes into the analysis JSON
    if a.stage in ('all', 'run'):
        log = stage_run([int(x) for x in a.seeds.split(',')], [k for k in a.only.split(',') if k] or None, a.tag or None)
    if a.stage in ('all', 'analyse'):
        stage_analyse(log)
