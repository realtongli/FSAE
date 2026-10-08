"""What each session-scarce attacker learns, from the saved test predictions (no training):
  - flow-level macro-F1 of the task (6 app x modality classes on GenAI, 9 apps on CCMA),
  - on GenAI also the assistant (3-class) and modality (2-class) macro-F1 derived from the 6-class prediction,
  - the session-level decision by majority vote over a session's flows (accuracy and macro-F1),
  - for every reference attacker, the 95% interval of 'pre-trained encoder minus attacker' from the two-level paired
    bootstrap of scripts/boot2.py (rules in its docstring): the five K-session draws are resampled with replacement, then
    the test sessions (4000 resamples, RandomState(20260926), the same resamples for every attacker of a cell), so the
    interval reflects both the choice of the labelled sessions and test-set sampling; 'sig' = the interval excludes zero.
    Also stored: the per-draw differences and, for reference only, the session-only interval that conditions on the five
    draws (test sessions only, 2000 resamples; 'ci_session_only').
The input-space 1-NN keeps no prediction file; it is deterministic and recomputed here (it matches its records).
Writes results/threat_stats.json."""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from trivial_baselines import feats_knn
import boot2

ATT = {'lgbm': 'fs{K}_lgbm_meta64', 'df': 'fs{K}_dfmeta64', 'tiktok': 'fs{K}_tiktok', 'tf': 'fs{K}_tf', 'cf': 'fs{K}_cfpub',
       'netclr': 'fs{K}_netclr', 'netclrtr': 'fs{K}_netclrtr', 'patch': 'meta_fewshot{K}_base', 'knn10': 'fs{K}_knn10',
       'cnn1d': 'fs{K}_paper1dcnn_512B', 'yatc': 'fs{K}_yatc_pre', 'etbert': 'fs{K}_etbert', 'scratch': 'fs{K}_sslscratchXL', 'ours': 'fs{K}_sslXL'}


def rid(ds, fmt, K, s):
    f = fmt.format(K=K)
    if ds == 'ccma': f = f'ccma_fewshot{K}_base' if f.startswith('meta_fewshot') else 'ccma_' + f
    return f + f'_s{s}'


def macro_f1_cm(cm):  # cm [..., C, C] (true, pred); classes absent from truth and prediction are skipped
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float); denom = cm.sum(-1) + cm.sum(-2)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0); present = denom > 0
    return (f1 * present).sum(-1) / present.sum(-1)


out = {}
for ds in ('genai', 'ccma'):
    D = GenAIData(ROOT, prefix=ds); ylab = D.y_joint if ds == 'genai' else D.y_app
    NC = D.n_joint if ds == 'genai' else D.n_app; nact = 2 if ds == 'genai' else 1
    Fk = feats_knn(D.meta, D.meta_len)
    for K in (1, 2, 4, 8):
        preds, test_idx = {}, None
        for s in range(5):
            sp = os.path.join(ROOT, 'configs', 'splits', f'{"" if ds == "genai" else "ccma_"}fewshot_k{K}_s{s}.json')
            idx, _ = D.split_indices(sp); tr, it = idx['train'], idx['test']
            if test_idx is None: test_idx = it
            assert np.array_equal(test_idx, it), 'the test sessions must be identical across seeds'
            p1 = np.empty(len(it), int)
            for b in range(0, len(it), 512): p1[b:b + 512] = ylab[tr][np.abs(Fk[it][b:b + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)]
            preds[('knn10', s)] = p1
            for a, fmt in ATT.items():
                if a == 'knn10': continue
                p = os.path.join(ROOT, 'results', 'runs', rid(ds, fmt, K, s), 'test_preds.npz')
                if not os.path.exists(p): continue
                z = np.load(p); assert np.array_equal(z['idx'], it) and np.array_equal(z['y'], ylab[it]), (a, K, s)
                preds[(a, s)] = z['pred']
        y = ylab[test_idx]; sid = D.session_id[test_idx]; us, inv = np.unique(sid, return_inverse=True); nS = len(us)
        exact_t = D.index['labeling_type'].values[test_idx] == 'exact' if 'labeling_type' in D.index else np.ones(len(test_idx), bool)
        ys = np.array([np.bincount(y[inv == j]).argmax() for j in range(nS)])
        cell = boot2.Cell(y, sid, NC)  # two-level resamples of this cell (scripts/boot2.py)
        res = {}
        for a in ATT:
            if not all((a, s) in preds for s in range(5)): continue
            st = {k: [] for k in ('flow', 'flow_exact', 'app', 'act', 'sess_acc', 'sess_f1', 'sess_app_acc', 'sess_act_acc')}
            cell.add(a, [preds[(a, s)] for s in range(5)])
            for s in range(5):
                pr = preds[(a, s)]
                st['flow'].append(100 * f1_score(y, pr, average='macro'))
                st['flow_exact'].append(100 * f1_score(y[exact_t], pr[exact_t], average='macro'))  # exact socket attribution only
                if nact > 1:
                    st['app'].append(100 * f1_score(y // nact, pr // nact, average='macro')); st['act'].append(100 * f1_score(y % nact, pr % nact, average='macro'))
                ps = np.array([np.bincount(pr[inv == j], minlength=NC).argmax() for j in range(nS)])
                st['sess_acc'].append(100 * accuracy_score(ys, ps)); st['sess_f1'].append(100 * f1_score(ys, ps, average='macro'))
                if nact > 1:  # GenAI: majority vote of the assistant (and of the modality) over the session's flows
                    pa = np.array([np.bincount(pr[inv == j] // nact, minlength=NC // nact).argmax() for j in range(nS)])
                    pm = np.array([np.bincount(pr[inv == j] % nact, minlength=nact).argmax() for j in range(nS)])
                    st['sess_app_acc'].append(100 * accuracy_score(ys // nact, pa)); st['sess_act_acc'].append(100 * accuracy_score(ys % nact, pm))
            res[a] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in st.items() if v}
        for a in res:
            if a == 'ours': continue
            c = cell.compare('ours', a)
            res[a]['diff'] = float(res['ours']['flow'][0] - res[a]['flow'][0]); res[a]['ci'] = tuple(c['ci']); res[a]['sig'] = c['sig']
            res[a]['per_draw'] = c['per_draw']; res[a]['ci_session_only'] = tuple(c['session_only']['ci']); res[a]['sig_session_only'] = c['session_only']['sig']
        out[f'{ds}_K{K}'] = dict(n_test_sessions=int(nS), attackers=res)
        print(f'\n=== {ds} K={K} (test sessions {nS})')
        for a, r in res.items():
            line = f'{a:9s} flow {r["flow"][0]:5.1f}±{r["flow"][1]:3.1f}'
            if 'app' in r: line += f'  app {r["app"][0]:5.1f}  act {r["act"][0]:5.1f}'
            line += f'  sessAcc {r["sess_acc"][0]:5.1f} sessF1 {r["sess_f1"][0]:5.1f}'
            if 'sess_app_acc' in r: line += f'  sessApp {r["sess_app_acc"][0]:5.1f} sessMod {r["sess_act_acc"][0]:5.1f}'
            if 'ci' in r: line += f'  ours-{a} {r["diff"]:+5.1f} [{r["ci"][0]:+5.1f},{r["ci"][1]:+5.1f}]{" *" if r["sig"] else ""}'
            print(line)
json.dump(out, open(os.path.join(ROOT, 'results', 'threat_stats.json'), 'w'), indent=1)
print('\nwritten results/threat_stats.json')
