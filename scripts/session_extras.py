"""Session-level extras from the saved test predictions (no model is trained; the 1-NN is training-free and recomputed).

Rules.
R1 Session vote (as scripts/robust_assistant.py and scripts/threat_stats.py): the class of a test session is the majority
   of the per-connection decision over the session's test connections (GenAI: assistant = 6-class prediction // 2;
   CCMA: the 9-app prediction), np.bincount(minlength=C).argmax(), so a tie goes to the lowest class index. The true
   class of a session is the majority of its connection labels (all its connections carry the session label).
R2 Ties: a vote is tied when two or more classes share the maximum count. 'Strict' accuracy counts every tied session as
   wrong, whichever class the tie rule picked.
R3 Clopper-Pearson (exact, two-sided 95%, beta quantiles): (i) per seed, x = correct sessions, n = test sessions;
   (ii) pooled over the five seeds, x = sum of correct, n = sum of test sessions (this treats the same test session in
   five seeds as five independent trials, so it is narrower than warranted whenever the test sessions repeat across
   seeds; reported for reference and flagged in the JSON); (iii) single-draw interval of the seed-mean: x = mean
   correct count over the seeds, n = the number of test sessions of one seed (only when all seeds have the same test
   sessions). (iii) with the per-seed range of (i) is the primary interval.
R4 Observers: the runs named below, each npz checked (idx equals the split's test indices, y equals the dataset label)
   and its task macro-F1 compared with the last non-smoke record in results/locked_test_metrics.jsonl. The 1-NN (L1 over
   the four channels of the first ten packets to every training connection, trivial_baselines.feats_knn, argmin = first
   index on distance ties) keeps no prediction file and is recomputed on each split; on the few-shot splits its macro-F1
   is checked against the fs{K}_knn10 records. Seeds 0-4; mean and seed range (min, max) for every number.
R5 Session modality (GenAI): score of a session = share of its test connections whose 6-class prediction is an image
   class (pred % 2 == 1). Vote: image iff share > 0.5 (share == 0.5 -> text, the lower index, = bincount argmax; counted
   as a tie). ROC-AUC of the score (sklearn, tied scores count 1/2) over the 48 test sessions (24 image vs 24 text) and
   within each assistant (8 vs 8). No threshold is selected.
R6 Host categories (GenAI, K=4, few-shot split): server name of each test connection from the raw payloads and category
   rules exactly as in scripts/construct_validity.py (server_name(), category(), imported unchanged; ground truth only,
   no observer reads it). Per assistant and category: share of the assistant's test connections, and recall of the
   assistant decision (pred // 2 == assistant) per observer per seed. Session vote restricted to 'assistant' + 'image'
   connections (R1 rule over those connections); a session without such a connection is an error; the accuracy over the
   sessions that have them is given as well.
R7 App versions: version of a session = BF_label_version_code (with BF_label_version_name) of the connections of the
   target package (build_dataset.TARGET_PKG) in the raw JSON of MIRAGE-GenAI-2025; every code seen is kept (a session
   with more than one code is flagged). Labelled = the train sessions of temporal_k{2,4,8}_s{0-4}; attacked = their test
   sessions. A code is 'unseen' if it occurs in no train set of the 15 temporal splits (val sessions are listed but, as in
   every run, never used for training or selection). The session vote on the attacked sessions with an unseen code comes
   from the drift runs of the assistant-level robustness table (encoder, LightGBM) and from the 1-NN recomputed on the temporal split.
R8 Overheads, over the labelled connections of each campaign, first 64 packets (the observer's window), with the
   defences applied to all connections exactly as in the runs (train_meta.apply_defense(meta[:, :64], 'pad256_jitter20',
   seed) for seeds 0-4; wf_defenses.ech / ech_full on the full array). Byte overhead (primary) = sum over connections of
   (IP bytes in the defended 64-packet window - IP bytes in the original window) / sum of IP bytes in the original window,
   the definition behind the paper's Tamaraw and ECH figures; also the mean and median of per-connection ratios. For ECH
   the server part = (ClientHello + server padding) - (ClientHello only); the untruncated flight accounting (IP bytes of
   the replaced flight minus those of the original flight) and payload-only padding are given as checks. Jitter delay:
   added prefix span (sum of the added gaps after the first packet) relative to the original span (first to last
   observed packet), aggregate and per-connection median, plus the mean cumulative added delay per observed packet.
Writes results/session_extras.json."""
import collections, glob, json, os, sys, time
import numpy as np
from scipy.stats import beta
from sklearn.metrics import f1_score, roc_auc_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from trivial_baselines import feats_knn
from train_meta import apply_defense
import wf_defenses as W
import construct_validity as CV
from build_dataset import TARGET_PKG

SEEDS = list(range(5)); KS = (1, 2, 4, 8); APPS = ['ChatGPT', 'Copilot', 'Gemini']
SPL = os.path.join(ROOT, 'configs', 'splits'); RUNS = os.path.join(ROOT, 'results', 'runs')
T0 = time.time()

LOCKED = {}
for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if not str(j.get('tag', '')).startswith('smoke'): LOCKED[j['run_id']] = j['test'].get('macro_f1')
CHECK = dict(max_abs_diff_vs_locked_pp=0.0, mismatches=[], no_record=[], missing=[])


def summ(v):
    v = [float(x) for x in v]
    return dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), per_seed=v)


def cp(x, n, a=0.05):
    lo = 0.0 if x <= 0 else float(beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x >= n else float(beta.ppf(1 - a / 2, x + 1, n - x))
    return [100 * lo, 100 * hi]


def rnd(o, d=3):
    if isinstance(o, (float, np.floating)): return round(float(o), d)
    if isinstance(o, dict): return {str(k): rnd(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v, d) for v in o]
    if isinstance(o, np.integer): return int(o)
    return o


DATA, FK = {}, {}


def data(ds):
    if ds not in DATA:
        DATA[ds] = GenAIData(ROOT, prefix=ds); FK[ds] = feats_knn(DATA[ds].meta, DATA[ds].meta_len)
    return DATA[ds]


def labels(ds):
    D = data(ds); return D.y_joint if ds == 'genai' else D.y_app


def split(ds, name, s):
    idx, _ = data(ds).split_indices(os.path.join(SPL, f'{name}_s{s}.json')); return idx


KNN_CACHE = {}


def knn1(ds, name, s):
    if (ds, name, s) in KNN_CACHE: return KNN_CACHE[(ds, name, s)]
    D = data(ds); y = labels(ds); idx = split(ds, name, s); tr, it = idx['train'], idx['test']; F = FK[ds]
    p = np.empty(len(it), int)
    for b in range(0, len(it), 512): p[b:b + 512] = y[tr][np.abs(F[it][b:b + 512, None, :] - F[tr][None]).sum(-1).argmin(1)]
    KNN_CACHE[(ds, name, s)] = p
    return p


def preds(ds, obs, run_fmt, name, s, locked_id=None):
    """Test predictions of one observer on split name_s{s}; None if the run is missing."""
    y = labels(ds); it = split(ds, name, s)['test']
    if obs == 'knn1':
        p = knn1(ds, name, s); rid = locked_id
    else:
        rid = run_fmt.format(s=s); f = os.path.join(RUNS, rid, 'test_preds.npz')
        if not os.path.exists(f): CHECK['missing'].append(rid); return None
        z = np.load(f); p = z['pred'].astype(int)
        assert np.array_equal(z['idx'], it), f'{rid}: idx differs from {name}_s{s}'
        assert np.array_equal(z['y'], y[it]), f'{rid}: labels differ'
    if rid is not None:
        f1 = 100 * f1_score(y[it], p, average='macro')
        if LOCKED.get(rid) is None: CHECK['no_record'].append(rid)
        else:
            d = abs(f1 - 100 * LOCKED[rid]); CHECK['max_abs_diff_vs_locked_pp'] = max(CHECK['max_abs_diff_vs_locked_pp'], d)
            if d > 1e-6: CHECK['mismatches'].append([rid, f1, 100 * LOCKED[rid]])
    return p


def vote_stats(ds, name, s, p):
    """R1/R2: per-session vote of the assistant (GenAI) or app (CCMA)."""
    D = data(ds); y = labels(ds); it = split(ds, name, s)['test']; nact = 2 if ds == 'genai' else 1
    NA = (D.n_joint if ds == 'genai' else D.n_app) // nact
    us, inv = np.unique(D.session_id[it], return_inverse=True)
    ya, pa = y[it] // nact, p // nact
    true = np.array([np.bincount(ya[inv == j], minlength=NA).argmax() for j in range(len(us))])
    cnt = np.array([np.bincount(pa[inv == j], minlength=NA) for j in range(len(us))])
    vote = cnt.argmax(1); tie = (cnt == cnt.max(1, keepdims=True)).sum(1) > 1
    ok = vote == true
    return dict(sessions=us, n=len(us), correct=int(ok.sum()), ties=int(tie.sum()), ties_resolved_correct=int((tie & ok).sum()),
                correct_strict=int((ok & ~tie).sum()), ok=ok, tie=tie, true=true)


def row_summary(per):  # per: list over seeds of vote_stats dicts
    n = [r['n'] for r in per]; c = [r['correct'] for r in per]; cs = [r['correct_strict'] for r in per]
    same = all(np.array_equal(per[0]['sessions'], r['sessions']) for r in per)
    out = dict(n_test_sessions=n, test_sessions_identical_across_seeds=bool(same),
               acc=summ([100 * a / b for a, b in zip(c, n)]), n_correct=c,
               cp95_per_seed=[cp(a, b) for a, b in zip(c, n)],
               cp95_pooled_over_seeds=dict(x=int(sum(c)), n=int(sum(n)), ci=cp(sum(c), sum(n)),
                                           note='treats repeated test sessions as independent; narrower than warranted' if same else 'test sessions differ across seeds'),
               ties=[r['ties'] for r in per], ties_resolved_correct=[r['ties_resolved_correct'] for r in per],
               acc_ties_wrong=summ([100 * a / b for a, b in zip(cs, n)]), n_correct_ties_wrong=cs)
    if same and len(set(n)) == 1:
        out['cp95_seed_mean'] = dict(x=float(np.mean(c)), n=int(n[0]), ci=cp(float(np.mean(c)), n[0]))
        out['cp95_seed_mean_ties_wrong'] = dict(x=float(np.mean(cs)), n=int(n[0]), ci=cp(float(np.mean(cs)), n[0]))
    return out


OUT = dict(rules=__doc__.split('Rules.')[1].split('Writes results')[0].strip())

# ============================================================== (a)+(b) the assistant-identity table (few-shot splits) and CCMA analogue
THREAT_OBS = {'lgbm': 'fs{K}_lgbm_meta64', 'knn1': 'fs{K}_knn10', 'netclr': 'fs{K}_netclr', 'ours': 'fs{K}_sslXL', 'yatc': 'fs{K}_yatc_pre'}
OUT['a_threat'] = {}
for ds in ('genai', 'ccma'):
    pre = '' if ds == 'genai' else 'ccma_'
    for K in KS:
        name = f'{pre}fewshot_k{K}'; row = {}
        for o, fmt in THREAT_OBS.items():
            rid = pre + fmt.format(K=K) + '_s{s}'; per = []
            for s in SEEDS:
                p = preds(ds, o, rid, name, s, locked_id=rid.format(s=s) if o == 'knn1' else None)
                if p is None: break
                per.append(vote_stats(ds, name, s, p))
            if len(per) == len(SEEDS): row[o] = row_summary(per)
        OUT['a_threat'][f'{ds}_K{K}'] = row
        print(f'[a] {ds} K={K} ' + '  '.join(f'{o} {r["acc"]["mean"]:.1f} [{r["acc"]["min"]:.1f},{r["acc"]["max"]:.1f}] '
                                             f'CPm {r["cp95_seed_mean"]["ci"][0]:.1f}-{r["cp95_seed_mean"]["ci"][1]:.1f} ties {sum(r["ties"])} strict {r["acc_ties_wrong"]["mean"]:.1f}'
                                             for o, r in row.items()), flush=True)

# ============================================================== (a)+(b) the assistant-level robustness table rows
CONDS = [
    ('none', ('fs4_sslXL', 'fs4_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL', 'ccma_fs4_lgbm_meta64', 'ccma_fewshot_k4')),
    ('drift_k2', ('fs2_sslXL_drift', 'fs2_drift_lgbm_meta64', 'temporal_k2'), ('ccma_fs2_sslXL_drift', 'ccma_fs2_drift_lgbm_meta64', 'ccma_temporal_k2')),
    ('drift_k4', ('fs4_sslXL_drift', 'fs4_drift_lgbm_meta64', 'temporal_k4'), ('ccma_fs4_sslXL_drift', 'ccma_fs4_drift_lgbm_meta64', 'ccma_temporal_k4')),
    ('drift_k8', ('fs8_sslXL_drift', 'fs8_drift_lgbm_meta64', 'temporal_k8'), ('ccma_fs8_sslXL_drift', 'ccma_fs8_drift_lgbm_meta64', 'ccma_temporal_k8')),
    ('phone1', ('cd8e_k4_sslXL', 'cd8e_k4_lgbm_meta64', 'cd_8e_k4'), ('ccma_cd2c_k4_sslXL', 'ccma_cd2c_k4_lgbm_meta64', 'ccma_cd_2c_k4')),
    ('phone2', ('cda6_k4_sslXL', 'cda6_k4_lgbm_meta64', 'cd_a6_k4'), ('ccma_cdf4_k4_sslXL', 'ccma_cdf4_k4_lgbm_meta64', 'ccma_cd_f4_k4')),
    ('phone3', None, ('ccma_cdf8_k4_sslXL', 'ccma_cdf8_k4_lgbm_meta64', 'ccma_cd_f8_k4')),
    ('ech', ('fs4_sslXL_ech', 'fs4Lech_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_ech', 'ccma_fs4Lech_lgbm_meta64', 'ccma_fewshot_k4')),
    ('echfull', ('fs4_sslXL_echfull', 'fs4Lechfull_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_echfull', 'ccma_fs4Lechfull_lgbm_meta64', 'ccma_fewshot_k4')),
    ('pj_unaware', ('fs4_sslXL_pj_adv0', 'fs4_pj_adv0_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_pj_adv0', 'ccma_fs4_pj_adv0_lgbm_meta64', 'ccma_fewshot_k4')),
    ('pj_adaptive', ('fs4_sslXL_pj_adv1', 'fs4_pj_adv1_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_pj_adv1', 'ccma_fs4_pj_adv1_lgbm_meta64', 'ccma_fewshot_k4')),
    ('front_unaware', ('fs4_sslXL_front_adv0', 'fs4_front_adv0_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_front_adv0', 'ccma_fs4_front_adv0_lgbm_meta64', 'ccma_fewshot_k4')),
    ('front_adaptive', ('fs4_sslXL_front_adv1', 'fs4_front_adv1_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_front_adv1', 'ccma_fs4_front_adv1_lgbm_meta64', 'ccma_fewshot_k4')),
    ('tamaraw_unaware', ('fs4_sslXL_tamaraw_adv0', 'fs4_tamaraw_adv0_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_tamaraw_adv0', 'ccma_fs4_tamaraw_adv0_lgbm_meta64', 'ccma_fewshot_k4')),
    ('tamaraw_adaptive', ('fs4_sslXL_tamaraw_adv1', 'fs4_tamaraw_adv1_lgbm_meta64', 'fewshot_k4'), ('ccma_fs4_sslXL_tamaraw_adv1', 'ccma_fs4_tamaraw_adv1_lgbm_meta64', 'ccma_fewshot_k4')),
]
OUT['a_robust_assistant'] = {'genai': {}, 'ccma': {}}
DRIFT_VOTES = {}   # (K, obs, s) -> vote_stats on the GenAI temporal split, reused in (e)
for ds in ('genai', 'ccma'):
    for key, g, c in CONDS:
        spec = g if ds == 'genai' else c
        if spec is None: continue
        rid_ours, rid_lgbm, name = spec; row = dict(split=name)
        for o, rid in (('ours', rid_ours), ('lgbm', rid_lgbm)):
            per = []
            for s in SEEDS:
                p = preds(ds, o, rid + '_s{s}', name, s)
                if p is None: break
                vs = vote_stats(ds, name, s, p); per.append(vs)
                if ds == 'genai' and key.startswith('drift'): DRIFT_VOTES[(int(key[-1]), o, s)] = (vs, p)
            if len(per) == len(SEEDS): row[o] = row_summary(per)
        OUT['a_robust_assistant'][ds][key] = row
        print(f'[a] {ds} {key:17s} n={row["ours"]["n_test_sessions"]} ' + '  '.join(
            f'{o} {row[o]["acc"]["mean"]:.1f} [{row[o]["acc"]["min"]:.1f},{row[o]["acc"]["max"]:.1f}] pooled {row[o]["cp95_pooled_over_seeds"]["ci"][0]:.1f}-{row[o]["cp95_pooled_over_seeds"]["ci"][1]:.1f} '
            f'CPm {row[o].get("cp95_seed_mean", {}).get("ci", [np.nan, np.nan])[0]:.1f}-{row[o].get("cp95_seed_mean", {}).get("ci", [np.nan, np.nan])[1]:.1f} '
            f'ties {row[o]["ties"]} strict {row[o]["acc_ties_wrong"]["mean"]:.1f}' for o in ('lgbm', 'ours')), flush=True)
print(f'[a] done {time.time() - T0:.0f}s; consistency {CHECK["max_abs_diff_vs_locked_pp"]:.2e}, mismatches {len(CHECK["mismatches"])}', flush=True)

# ============================================================== (c) session-level modality (GenAI, few-shot splits)
D = data('genai'); y6 = D.y_joint
MOD_OBS = {'ours': 'fs{K}_sslXL', 'lgbm': 'fs{K}_lgbm_meta64', 'knn1': 'fs{K}_knn10'}
OUT['c_session_modality'] = {}
for K in KS:
    name = f'fewshot_k{K}'; rk = {}
    for o, fmt in MOD_OBS.items():
        rid = fmt.format(K=K) + '_s{s}'
        acc, strict, ties, auc, pa_auc, pa_acc, gap = [], [], [], [], {a: [] for a in APPS}, {a: [] for a in APPS}, []
        for s in SEEDS:
            p = preds('genai', o, rid, name, s, locked_id=rid.format(s=s) if o == 'knn1' else None)
            it = split('genai', name, s)['test']; us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
            yy = y6[it]; sm = np.array([np.bincount(yy[inv == j] % 2).argmax() for j in range(nS)]); sa = np.array([np.bincount(yy[inv == j] // 2).argmax() for j in range(nS)])
            sh = np.array([np.mean(p[inv == j] % 2 == 1) for j in range(nS)])
            v = (sh > 0.5).astype(int); t = sh == 0.5
            acc.append(100 * np.mean(v == sm)); strict.append(100 * np.mean((v == sm) & ~t)); ties.append(int(t.sum()))
            auc.append(100 * roc_auc_score(sm, sh)); gap.append(100 * (sh[sm == 1].mean() - sh[sm == 0].mean()))
            for a in range(3):
                m = sa == a; pa_acc[APPS[a]].append(100 * np.mean(v[m] == sm[m])); pa_auc[APPS[a]].append(100 * roc_auc_score(sm[m], sh[m]))
        rk[o] = dict(n_sessions=int(nS), vote_acc=summ(acc), vote_acc_ties_wrong=summ(strict), ties=ties, auc=summ(auc),
                     mean_share_gap_image_minus_text=summ(gap), auc_per_assistant={a: summ(v) for a, v in pa_auc.items()},
                     vote_acc_per_assistant={a: summ(v) for a, v in pa_acc.items()})
    OUT['c_session_modality'][f'K{K}'] = rk
    print(f'[c] K={K} ' + '  '.join(f'{o} vote {r["vote_acc"]["mean"]:.1f} [{r["vote_acc"]["min"]:.1f},{r["vote_acc"]["max"]:.1f}] AUC {r["auc"]["mean"]:.1f} [{r["auc"]["min"]:.1f},{r["auc"]["max"]:.1f}] '
                                    'per-asst AUC ' + '/'.join(f'{v["mean"]:.0f}' for v in r['auc_per_assistant'].values()) for o, r in rk.items()), flush=True)
ts_p = os.path.join(ROOT, 'results', 'threat_stats.json')
if os.path.exists(ts_p):
    ts = json.load(open(ts_p)); chk = {}
    for K in KS:
        for o, a in (('ours', 'ours'), ('lgbm', 'lgbm'), ('knn1', 'knn10')):
            r = ts.get(f'genai_K{K}', {}).get('attackers', {}).get(a, {})
            if 'sess_act_acc' in r: chk[f'K{K}_{o}'] = [round(r['sess_act_acc'][0], 3), round(OUT['c_session_modality'][f'K{K}'][o]['vote_acc']['mean'], 3)]
    OUT['c_session_modality']['consistency_threat_stats_[threat_stats, here]'] = chk

# ============================================================== raw pass: server names of the K=4 test connections, app versions
ses = D.sessions; files = dict(zip(ses.session_id, ses.file)); RAW = os.path.join(ROOT, 'data', 'mirage2025genai')
row_of = {(files[s], k): i for i, (s, k) in enumerate(zip(D.index.session_id, D.index.biflow_key))}
it4 = split('genai', 'fewshot_k4', 0)['test']; it4_set = set(it4.tolist()); test_files = set(files[s] for s in np.unique(D.session_id[it4]))
CV.selftest()
H = {}; VER = {}
for sid, f, cls in zip(ses.session_id, ses.file, ses.class_dir):
    app = 'Chatgpt' if cls.lower().startswith('chatgpt') else 'Copilot' if cls.lower().startswith('copilot') else 'Gemini'
    d = json.load(open(os.path.join(RAW, f))); vv = collections.Counter()
    for key, b in d.items():
        md = b['flow_metadata']
        if md['BF_label'] in TARGET_PKG[app]: vv[str(md.get('BF_label_version_code'))] += 1
        if f in test_files and (f, key) in row_of and row_of[(f, key)] in it4_set:
            H[row_of[(f, key)]] = CV.server_name(b['packet_data'], int(key.split(',')[-1]))[0]
    VER[int(sid)] = dict(app=app, file=f, ts=int(ses.session_ts[ses.session_id == sid].iloc[0]), device=str(ses.device[ses.session_id == sid].iloc[0]),
                         part=str(ses.part[ses.session_id == sid].iloc[0]), codes=dict(vv))
print(f'[raw] {len(H)} test connections named/unnamed, {len(VER)} sessions, {time.time() - T0:.0f}s', flush=True)

# ============================================================== (d) host categories, K=4
assert len(H) == len(it4), 'every K=4 test connection must be matched to its raw record'
ya4 = D.y_app[it4]; cat = np.array([CV.category(H[i], int(D.y_app[i])) for i in it4], dtype=object)
us4, inv4 = np.unique(D.session_id[it4], return_inverse=True); s_app = np.array([ya4[inv4 == j][0] for j in range(len(us4))])
P4 = {}
for o, fmt in MOD_OBS.items():
    rid = fmt.format(K=4) + '_s{s}'
    for s in SEEDS: P4[(o, s)] = preds('genai', o, rid, 'fewshot_k4', s, locked_id=rid.format(s=s) if o == 'knn1' else None)
cvj = json.load(open(os.path.join(ROOT, 'results', 'construct_validity.json')))['q1']['K4']['per_app']
dres = dict(n_test_connections=int(len(it4)), per_assistant={})
for a in range(3):
    ma = ya4 == a; ra = dict(n_connections=int(ma.sum()), n_sessions=int((s_app == a).sum()), categories={}, session_vote={})
    for c in CV.CATS:
        m = ma & (cat == c)
        e = dict(n=int(m.sum()), share=100 * float(m.sum() / ma.sum()), n_sessions=int(len(np.unique(D.session_id[it4][m]))),
                 matches_construct_validity=bool(cvj[APPS[a]]['categories'][c]['n'] == int(m.sum())), recall={})
        if m.any():
            for o in MOD_OBS: e['recall'][o] = summ([100 * np.mean(P4[(o, s)][m] // 2 == a) for s in SEEDS])
        ra['categories'][c] = e
    sa_idx = np.where(s_app == a)[0]
    for o in MOD_OBS:
        rules = {'all': np.ones(len(it4), bool), 'assistant_and_image_only': np.isin(cat, ['assistant', 'image'])}
        sv = {}
        for rn, keep in rules.items():
            corr, nonempty, empty = [], [], None
            for s in SEEDS:
                pa = P4[(o, s)] // 2; c_ = 0; ne = 0; em = 0; cne = 0
                for j in sa_idx:
                    m = (inv4 == j) & keep
                    if not m.any(): em += 1; continue
                    v = np.bincount(pa[m], minlength=3).argmax(); ne += 1; cne += int(v == a)
                corr.append(cne); empty = em; nonempty.append(ne)
            nS = len(sa_idx)
            sv[rn] = dict(n_sessions=nS, sessions_without_connections=empty, n_correct=corr, acc_empty_as_error=summ([100 * x / nS for x in corr]),
                          acc_on_sessions_with_connections=summ([100 * x / max(nS - empty, 1) for x in corr]) if nS > empty else None)
        ra['session_vote'][o] = sv
    dres['per_assistant'][APPS[a]] = ra
    print(f'[d] {APPS[a]} ' + '  '.join(f'{c}: {e["share"]:.1f}% ' + '/'.join(f'{e["recall"][o]["mean"]:.1f}' for o in e['recall']) for c, e in ra['categories'].items() if e['n']) +
          ' | A+I vote ' + ' '.join(f'{o} {v["assistant_and_image_only"]["acc_empty_as_error"]["mean"]:.1f} (non-empty {v["assistant_and_image_only"]["acc_on_sessions_with_connections"]["mean"] if v["assistant_and_image_only"]["acc_on_sessions_with_connections"] else float("nan"):.1f}, empty {v["assistant_and_image_only"]["sessions_without_connections"]})' for o, v in ra['session_vote'].items()), flush=True)
dres['overall_share'] = {c: 100 * float(np.mean(cat == c)) for c in CV.CATS}
OUT['d_host_categories_K4'] = dres

# ============================================================== (e) app versions in the temporal split
file2sid = D.file2sid; sid_ver = {}
for sid, v in VER.items():
    codes = v['codes']; sid_ver[sid] = max(codes, key=codes.get) if codes else None
multi = {sid: v['codes'] for sid, v in VER.items() if len(v['codes']) > 1}
eres = dict(sessions_with_more_than_one_code=multi, per_assistant={}, splits={})
train_codes = {a: collections.Counter() for a in APPS}; val_codes = {a: collections.Counter() for a in APPS}
test_sets = {}
AMAP = {'Chatgpt': 'ChatGPT', 'Copilot': 'Copilot', 'Gemini': 'Gemini'}
for K in (2, 4, 8):
    for s in SEEDS:
        sp = json.load(open(os.path.join(SPL, f'temporal_k{K}_s{s}.json'))); r = {}
        for part in ('train', 'val', 'test'):
            cc = collections.Counter((AMAP[VER[file2sid[f]]['app']], sid_ver[file2sid[f]]) for f in sp[part])
            r[part] = {f'{a}|{c}': n for (a, c), n in sorted(cc.items())}
            for (a, c), n in cc.items():
                if part == 'train': train_codes[a][c] += n
                if part == 'val': val_codes[a][c] += n
            if part == 'test': test_sets[(K, s)] = sorted(file2sid[f] for f in sp['test'])
        eres['splits'][f'temporal_k{K}_s{s}'] = r
same_test = len(set(tuple(v) for v in test_sets.values())) == 1
test_sids = test_sets[(4, 0)]
for a in APPS:
    sids_a = [sid for sid in test_sids if AMAP[VER[sid]['app']] == a]
    tc = collections.Counter(sid_ver[sid] for sid in sids_a)
    unseen = sorted(c for c in tc if c not in train_codes[a])
    all_lab = sorted([(VER[sid]['ts'], sid_ver[sid]) for sid in VER if AMAP[VER[sid]['app']] == a and VER[sid]['part'] == 'generic'])
    first_seen = {}
    for ts_, c in all_lab: first_seen.setdefault(c, [ts_, ts_]); first_seen[c][1] = ts_
    eres['per_assistant'][a] = dict(train_codes_all_15_splits=dict(train_codes[a]), val_codes_all_15_splits=dict(val_codes[a]), test_codes=dict(tc),
                                    n_test_sessions=len(sids_a), unseen_codes=unseen, n_test_sessions_unseen=int(sum(tc[c] for c in unseen)),
                                    unseen_codes_in_any_val=[c for c in unseen if c in val_codes[a]],
                                    generic_sessions_code_span_utc={c: [time.strftime('%Y-%m-%d', time.gmtime(v[0])), time.strftime('%Y-%m-%d', time.gmtime(v[1]))] for c, v in first_seen.items()})
eres['test_sessions_identical_across_temporal_splits'] = bool(same_test)
# vote on the attacked sessions whose code is unseen, per observer and K
unseen_sids = {sid for a in APPS for sid in test_sids if AMAP[VER[sid]['app']] == a and sid_ver[sid] in eres['per_assistant'][a]['unseen_codes']}
eres['unseen_session_ids'] = sorted(int(x) for x in unseen_sids)
votes = {}
for K in (2, 4, 8):
    for o in ('ours', 'lgbm', 'knn1'):
        corr_u, corr_seen_cop, rec_u, n_u = [], [], [], None
        for s in SEEDS:
            if o == 'knn1':
                p = knn1('genai', f'temporal_k{K}', s); vs = vote_stats('genai', f'temporal_k{K}', s, p)
            else:
                vs, p = DRIFT_VOTES[(K, o, s)]
            it = split('genai', f'temporal_k{K}', s)['test']; sess = vs['sessions']
            mu = np.isin(sess, list(unseen_sids)); n_u = int(mu.sum())
            corr_u.append(int(vs['ok'][mu].sum()))
            mcop = np.array([AMAP[VER[int(x)]['app']] == 'Copilot' for x in sess]) & ~mu
            corr_seen_cop.append([int(vs['ok'][mcop].sum()), int(mcop.sum())])
            mc = np.isin(D.session_id[it], list(unseen_sids)); rec_u.append(100 * np.mean(p[mc] // 2 == D.y_app[it][mc]))
        votes[f'K{K}_{o}'] = dict(
            n_unseen_sessions=n_u, n_correct=corr_u, acc=summ([100 * c / n_u for c in corr_u]), conn_assistant_recall_on_unseen=summ(rec_u),
            other_copilot_test_sessions_correct_of_n=corr_seen_cop, run='1-NN recomputed on the temporal split' if o == 'knn1' else 'drift run of the assistant-level robustness table')
        print(f'[e] K={K} {o:5s} unseen-version sessions {n_u}: correct {corr_u}  conn recall {np.mean(rec_u):.1f}', flush=True)
eres['vote_on_unseen_version_sessions'] = votes
OUT['e_versions'] = eres
for a, v in eres['per_assistant'].items(): print('[e]', a, json.dumps({k: v[k] for k in ('train_codes_all_15_splits', 'test_codes', 'unseen_codes', 'n_test_sessions_unseen', 'n_test_sessions')}))

# ============================================================== (f) overheads
T = 64


def ch_flight(m, n):  # wf_defenses.ech first_flight (replicated, unchanged)
    j0 = next((j for j in range(n) if m[j, 2] > 0), None)
    if j0 is None or j0 == 0 or m[j0, 0] != 1.0 or (m[j0, 1] - m[j0, 2]) in (28.0, 48.0): return None
    j1 = j0
    while j1 + 1 < n and not (m[j1 + 1, 0] == -1.0 and m[j1 + 1, 2] > 0): j1 += 1
    return j0, j1, [j for j in range(j0, j1 + 1) if m[j, 0] == 1.0 and m[j, 2] > 0]


def sv_flight(m, n):  # wf_defenses.ech_full flights (replicated, unchanged)
    j0 = next((j for j in range(n) if m[j, 2] > 0), None)
    if j0 is None or j0 == 0 or m[j0, 0] != 1.0 or (m[j0, 1] - m[j0, 2]) in (28.0, 48.0): return None
    s0 = next((j for j in range(j0 + 1, n) if m[j, 0] == -1.0 and m[j, 2] > 0), None)
    if s0 is None: return None
    s1 = s0
    while s1 + 1 < n and not (m[s1 + 1, 0] == 1.0 and m[s1 + 1, 2] > 0): s1 += 1
    return s0, s1, [j for j in range(s0, s1 + 1) if m[j, 0] == -1.0 and m[j, 2] > 0]


def segs_of(total, C, mss=1388.0):
    new = C if total <= C else float(np.ceil(total / 256.0) * 256.0)
    return new, [mss] * int(new // mss) + ([new % mss] if new % mss else [])


def ratio_stats(add, real):
    per = add / np.maximum(real, 1.0)
    return dict(aggregate=float(add.sum() / real.sum()), mean_of_ratios=float(per.mean()), median_of_ratios=float(np.median(per)))


fres = {}
for ds in ('genai', 'ccma'):
    Dd = data(ds); lab = labels(ds) >= 0
    mA = Dd.meta[:, :T].copy(); nA = np.minimum(Dd.meta_len, T).astype(int)       # exactly what the runs pass to the defences
    valid = np.arange(T)[None, :] < nA[:, None]
    m64 = mA.astype(np.float64); real = (m64[..., 1] * valid).sum(1)
    r = dict(n_connections=int(lab.sum()), observed_ip_bytes=float(real[lab].sum()))
    # padding to 256 B (deterministic) and jitter (seeds 0-4), as train_meta.apply_defense
    pad = apply_defense(m64, 'pad256', 0); addp = ((pad[..., 1] - m64[..., 1]) * valid).sum(1)
    r['pad256_bytes'] = ratio_stats(addp[lab], real[lab])
    r['pad256_share_of_payload_packets_changed'] = float(((pad[..., 2] != m64[..., 2]) & valid)[lab].sum() / ((m64[..., 2] > 0) & valid)[lab].sum())
    jit = {k: [] for k in ('aggregate_added_span_over_original_span', 'median_per_conn_ratio', 'mean_cumulative_delay_per_packet_ms',
                           'mean_added_delay_last_observed_packet_ms', 'median_added_span_ms', 'share_conns_span_at_least_doubled')}
    span0 = (m64[:, 1:, 3] * valid[:, 1:]).sum(1); two = lab & (nA >= 2)
    for s in SEEDS:
        sh = apply_defense(m64, 'pad256_jitter20', s)
        assert np.allclose(sh[..., 1], pad[..., 1]), 'padding must not depend on the seed'
        dj = (sh[..., 3] - m64[..., 3]) * valid
        add_span = dj[:, 1:].sum(1); cum = np.cumsum(dj, 1)
        jit['aggregate_added_span_over_original_span'].append(float(add_span[two].sum() / span0[two].sum()))
        pos = two & (span0 > 0)
        jit['median_per_conn_ratio'].append(float(np.median(add_span[pos] / span0[pos])))
        jit['mean_cumulative_delay_per_packet_ms'].append(1000 * float(cum[valid & lab[:, None]].mean()))
        jit['mean_added_delay_last_observed_packet_ms'].append(1000 * float(cum[np.where(lab)[0], nA[lab] - 1].mean()))
        jit['median_added_span_ms'].append(1000 * float(np.median(add_span[two])))
        jit['share_conns_span_at_least_doubled'].append(float(np.mean(add_span[two] >= span0[two])))
    r['jitter'] = {k: summ(v) for k, v in jit.items()}
    r['original_prefix_span_s'] = dict(median=float(np.median(span0[two])), mean=float(span0[two].mean()), n_conns_2plus_packets=int(two.sum()))
    # ECH: ClientHello only, and with server padding, exactly as the runs call them (whole array, float32)
    o1, l1, ov1 = W.ech(mA, nA.copy()); oF, lF, ovF = W.ech_full(mA, nA.copy())
    w1 = np.array([o1[i, :l1[i], 1].astype(np.float64).sum() for i in range(len(mA))]); wF = np.array([oF[i, :lF[i], 1].astype(np.float64).sum() for i in range(len(mA))])
    ch_w = ratio_stats((w1 - real)[lab], real[lab]); full_w = ratio_stats((wF - real)[lab], real[lab])
    # untruncated flight accounting and payload-only padding (checks)
    Cc, Cs = float(ov1['clienthello_bytes']), float(ovF['server_flight_bytes'])
    ch_ip = np.zeros(len(mA)); ch_pay = np.zeros(len(mA)); sv_ip = np.zeros(len(mA)); sv_pay = np.zeros(len(mA)); ch_on = np.zeros(len(mA), bool); sv_on = np.zeros(len(mA), bool)
    for i in range(len(mA)):
        n = int(nA[i]); m = mA[i, :n].astype(np.float64); ff = ch_flight(m, n) if n else None
        if ff:
            j0, j1, idx = ff; hdr = m[j0, 1] - m[j0, 2]; tot = m[idx, 2].sum(); new, sg = segs_of(tot, Cc)
            ch_ip[i] = sum(x + hdr for x in sg) - m[idx, 1].sum(); ch_pay[i] = new - tot; ch_on[i] = True
        n1 = int(min(l1[i], T)); m1 = o1[i, :n1].astype(np.float64); f2 = sv_flight(m1, n1) if n1 else None
        if f2:
            s0, s1, idx = f2; hdr = m1[s0, 1] - m1[s0, 2]; tot = m1[idx, 2].sum(); new, sg = segs_of(tot, Cs)
            up_ack = [float(m1[j, 1]) for j in range(1, n1) if m1[j, 0] == 1.0 and m1[j, 2] == 0]
            ack_ip = max(set(up_ack), key=up_ack.count) if up_ack else hdr
            fl = 0.0
            for k, x in enumerate(sg):
                fl += x + hdr
                if k % 2 == 1 or k == len(sg) - 1: fl += ack_ip
            sv_ip[i] = fl - m1[s0:s1 + 1, 1].sum(); sv_pay[i] = new - tot; sv_on[i] = True
    r['ech'] = dict(clienthello_constant_bytes=Cc, server_flight_constant_bytes=Cs,
                    share_conns_clienthello_padded=float(ch_on[lab].mean()), share_conns_server_flight_padded=float(sv_on[lab].mean()),
                    window_clienthello_only=ch_w, window_clienthello_plus_server=full_w,
                    window_server_part=dict(aggregate=full_w['aggregate'] - ch_w['aggregate'], mean_of_ratios=full_w['mean_of_ratios'] - ch_w['mean_of_ratios']),
                    flight_accounting_ip=dict(clienthello=ratio_stats(ch_ip[lab], real[lab]), server=ratio_stats(sv_ip[lab], real[lab]),
                                              both=ratio_stats((ch_ip + sv_ip)[lab], real[lab])),
                    payload_only=dict(clienthello=ratio_stats(ch_pay[lab], real[lab]), server=ratio_stats(sv_pay[lab], real[lab]),
                                      both=ratio_stats((ch_pay + sv_pay)[lab], real[lab])),
                    logged_bandwidth_overhead=dict(ech=float(ov1['bandwidth_overhead']), ech_full=float(ovF['bandwidth_overhead'])))
    fres[ds] = r
    e = r['ech']
    print(f'[f] {ds} pad256 agg {r["pad256_bytes"]["aggregate"]:.3f} mean {r["pad256_bytes"]["mean_of_ratios"]:.3f} | jitter span agg {r["jitter"]["aggregate_added_span_over_original_span"]["mean"]:.2f} '
          f'median ratio {r["jitter"]["median_per_conn_ratio"]["mean"]:.2f} cum/pkt {r["jitter"]["mean_cumulative_delay_per_packet_ms"]["mean"]:.0f}ms orig median span {r["original_prefix_span_s"]["median"]:.3f}s | '
          f'ECH C_c {e["clienthello_constant_bytes"]:.0f} C_s {e["server_flight_constant_bytes"]:.0f} window CH {e["window_clienthello_only"]["aggregate"]:.3f} both {e["window_clienthello_plus_server"]["aggregate"]:.3f} '
          f'server {e["window_server_part"]["aggregate"]:.3f} | flight CH {e["flight_accounting_ip"]["clienthello"]["aggregate"]:.3f} server {e["flight_accounting_ip"]["server"]["aggregate"]:.3f} | '
          f'payload CH {e["payload_only"]["clienthello"]["aggregate"]:.3f} server {e["payload_only"]["server"]["aggregate"]:.3f}', flush=True)
OUT['f_overheads'] = fres

CHECK['missing'] = sorted(set(CHECK['missing'])); CHECK['no_record'] = sorted(set(CHECK['no_record']))
OUT['consistency'] = CHECK
json.dump(rnd(OUT), open(os.path.join(ROOT, 'results', 'session_extras.json'), 'w'), indent=1)
print(f'consistency: max |F1 npz - locked| = {CHECK["max_abs_diff_vs_locked_pp"]:.2e} pp, mismatches {CHECK["mismatches"][:5]}, no record {CHECK["no_record"][:10]}, missing {CHECK["missing"][:10]}')
print(f'written results/session_extras.json ({time.time() - T0:.0f}s)')
