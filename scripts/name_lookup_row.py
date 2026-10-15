"""Server-name lookup, the reference row of the assistant-identification table (training-free; no model is trained,
nothing is selected).

Lookup. Exactly the 'server-name lookup' of scripts/leak_source.py: the server name of a connection is the one its
  plaintext ClientHello carries ('<none>' if it carries none); over the connections of the K labelled sessions the
  lookup stores the majority class of every name; a test connection gets the class of its name, or, if the name does
  not occur in the labelled sessions, the majority class of the labelled connections.
Splits. configs/splits/[ccma_]fewshot_k{K}_s{s}.json, K = 1, 2, 4, 8, draws s = 0-4 (the draws of that table).
Metrics. Exactly scripts/threat_stats.py: macro-F1 of the task classes ('flow'); on GenAI the assistant macro-F1
  (pred//2 vs y//2, 'app'), the modality macro-F1 (pred%2 vs y%2, 'act') and the share of test sessions whose
  assistant wins the vote over the session's test connections ('sess_app_acc', ties to the lowest index); on CCMA the
  share of test sessions whose app wins the vote ('sess_acc'). Mean and population s.d. over the five draws.
Diagnostics. Share of test connections without a server name, and share whose name occurs in the labelled sessions
  ('coverage'), with the task accuracy of the lookup on the covered and on the other test connections.
Cross-check. 'flow' and 'app' at K = 1, 4, 8 must equal results/leak_source.json (lookup_f1, lookup_app_f1).
Writes results/name_lookup_row.json."""
import collections, json, os, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData

NC = {'genai': 6, 'ccma': 9}


def host_from(buf, span):  # as scripts/leak_source.py
    s = int(span[0])
    if s < 0: return None
    b = bytes(buf)
    try:
        nl = int.from_bytes(b[s + 7:s + 9], 'big'); h = b[s + 9:s + 9 + nl].decode('ascii', 'replace').lower()
        return h or None
    except Exception:
        return None


def hosts(d):  # as scripts/leak_source.py
    out = []
    for i in range(len(d.meta)):
        h = host_from(d.flat_bytes[i], d.flat_sni[i])
        if h is None:
            for p in range(d.pkt_sni.shape[1]):
                h = host_from(d.pkt_bytes[i, p], d.pkt_sni[i, p])
                if h: break
        out.append(h or '<none>')
    return np.array(out, dtype=object)


mf1 = lambda y, p, nc: 100 * f1_score(y, p, labels=list(range(nc)), average='macro', zero_division=0)
REF = json.load(open(os.path.join(ROOT, 'results', 'leak_source.json'), encoding='utf-8'))
OUT = {}
for ds, pre in (('genai', ''), ('ccma', 'ccma_')):
    d = GenAIData(ROOT, prefix=ds); y_all = d.y_joint if ds == 'genai' else d.y_app; nact = 2 if ds == 'genai' else 1
    H = hosts(d); nc = NC[ds]; OUT[ds] = {}
    for K in (1, 2, 4, 8):
        st = collections.defaultdict(list)
        for s in range(5):
            idx, _ = d.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json')); tr, te = idx['train'], idx['test']
            cnt = collections.defaultdict(collections.Counter)
            for i in tr: cnt[H[i]][y_all[i]] += 1
            maj = np.bincount(y_all[tr], minlength=nc).argmax()
            pr = np.array([cnt[H[i]].most_common(1)[0][0] if H[i] in cnt else maj for i in te])
            y = y_all[te]; us, inv = np.unique(d.session_id[te], return_inverse=True); nS = len(us)
            ys = np.array([np.bincount(y[inv == j]).argmax() for j in range(nS)])
            st['flow'].append(mf1(y, pr, nc))
            ps = np.array([np.bincount(pr[inv == j], minlength=nc).argmax() for j in range(nS)])
            st['sess_acc'].append(100 * accuracy_score(ys, ps))
            if nact > 1:
                st['app'].append(100 * f1_score(y // nact, pr // nact, average='macro'))
                st['act'].append(100 * f1_score(y % nact, pr % nact, average='macro'))
                pa = np.array([np.bincount(pr[inv == j] // nact, minlength=nc // nact).argmax() for j in range(nS)])
                st['sess_app_acc'].append(100 * accuracy_score(ys // nact, pa))
            named = set(H[tr]) - {'<none>'}; cov = np.array([H[i] in named for i in te]); none = H[te] == '<none>'
            hit = (y // nact) == (pr // nact)   # GenAI: assistant; CCMA: app
            st['no_name'].append(100 * float(none.mean())); st['coverage'].append(100 * float(cov.mean()))
            st['acc_covered'].append(100 * float(hit[cov].mean())); st['acc_other'].append(100 * float(hit[~cov].mean()) if (~cov).any() else float('nan'))
            st['n_test_sessions'].append(nS)
        r = {k: (float(np.mean(v)), float(np.std(v))) for k, v in st.items()}
        if str(K) in REF[ds]['per_K']:
            ref = REF[ds]['per_K'][str(K)]
            assert abs(ref['lookup_f1'][0] - r['flow'][0]) < 1e-9, (ds, K, ref['lookup_f1'], r['flow'])
            if ds == 'genai': assert abs(ref['lookup_app_f1'][0] - r['app'][0]) < 1e-9, (ds, K)
        OUT[ds][f'K{K}'] = r
        print(f'{ds} K={K}  ' + '  '.join(f'{k} {v[0]:.2f}' for k, v in r.items()))
json.dump(OUT, open(os.path.join(ROOT, 'results', 'name_lookup_row.json'), 'w'), indent=1)
print('cross-check against results/leak_source.json passed; written results/name_lookup_row.json')
