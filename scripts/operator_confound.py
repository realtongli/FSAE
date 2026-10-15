"""Do the observers recognise the app or only the front-end operator its connections end at? Training-free: reads the
saved test predictions and writes results/operator_confound.json.

Attribution. The operator and the service of a server address exactly as scripts/cdn_attribution_archived.py: the
  archived published lists of data/ip_ranges/2026-09-27/ and the replayed RDAP registrations of
  results/cdn_attribution.json. No network query is made; an address that no list matches and no stored registration
  contains is 'unlisted'.

Part A, monitored classes in the closed world (GenAI: the assistant, joint class // 2; CCMA: the app).
  Population: the test connections of configs/splits/[ccma_]fewshot_k{K}_s{s}.json, K = 1, 2, 4, 8, draws 0-4.
  Operator-only bounds, at two granularities (operator, service):
    'oracle'  every test connection gets the most frequent TRUE class of its operator among the test connections. No
              observer that knows only the operator can be more accurate; it is fitted on the test labels, so it favours
              the hypothesis that the operator explains the result.
    'lookup'  the same mapping fitted on the connections of the K labelled sessions (an operator absent from them gets
              their majority class).
  Observers: the pre-trained encoder ([ccma_]fs{K}_sslXL), LightGBM ([ccma_]fs{K}_lgbm_meta64), both from their saved
    test predictions; the nearest neighbour on the first ten packets (scripts/trivial_baselines.feats_knn, L1); the
    address back-off lookup (scripts/ip_baseline.predict).
  Reported per K (mean over the draws): accuracy and macro-F1 over all test connections; the same restricted to the
    test connections of SHARED operators, i.e. operators on which at least two classes each hold at least 5% of the
    operator's test connections; per shared operator at K = 4 its number of test connections, the classes that hold at
    least 5% and the share of the largest.

Helpers used by scripts/build_bg_ownservice_mask.py: Attr (the attribution above), background_addresses() (the server
  address, package and session of every row of data/derived/genai_background.npz, read again from the raw JSON in the
  order of scripts/build_background.py, with package and session asserted equal to the npz) and host_of() (the
  plaintext server name of a connection).
"""
import collections, glob, ipaddress, json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cdn_attribution as CA  # noqa: E402
import cdn_attribution_archived as CAA  # noqa: E402
from data_mfr import GenAIData  # noqa: E402
from ip_baseline import server_addr, keys_of, predict  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

SCR = os.environ.get('OC_CACHE', os.path.join(ROOT, 'results', '_operator_confound_cache'))
ARCH_REF = json.load(open(os.path.join(ROOT, 'results', 'cdn_attribution_archived.json'), encoding='utf-8'))
ASSIST = ['ChatGPT', 'Copilot', 'Gemini']


class Attr:
    def __init__(self):
        self.L, _ = CAA.load_archived(); self.rd = CAA.ReplayRDAP(CAA.REF['rdap_registrations']); self.memo = {}

    def __call__(self, a):
        k = str(a)
        if k not in self.memo:
            n0 = len(self.rd.missing); r = CA.attribute(self.L, a, self.rd, False)
            if len(self.rd.missing) > n0 and r['operator'] in ('other', 'unknown'): r = dict(r, operator='unlisted', service='unlisted')
            self.memo[k] = r
        return self.memo[k]


def part_a(AT):
    out = {}
    from trivial_baselines import feats_knn
    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); chk = collections.Counter(); A = [server_addr(k, chk) for k in D.index.biflow_key.values]
        ylab = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app; nact = 2 if ds == 'genai' else 1
        pre = '' if ds == 'genai' else 'ccma_'; lab = ylab >= 0; KK = keys_of(A); Fk = feats_knn(D.meta, D.meta_len)
        names = ASSIST if ds == 'genai' else [str(D.index['app'].values[np.where(lab & (D.y_app == c))[0][0]]) for c in range(D.n_app)]
        G = {g: np.array([AT(A[i])[g] if lab[i] else '' for i in range(len(A))], dtype=object) for g in ('operator', 'service')}
        res = {}
        for K in (1, 2, 4, 8):
            st = collections.defaultdict(list); shared_info = None
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json')); tr, te = idx['train'], idx['test']
                y = ylab[te] // nact; ytr = ylab[tr] // nact; na = NC // nact
                P = {}
                for a, rid in (('ours', f'{pre}fs{K}_sslXL_s{s}'), ('lgbm', f'{pre}fs{K}_lgbm_meta64_s{s}')):
                    z = np.load(os.path.join(ROOT, 'results', 'runs', rid, 'test_preds.npz')); assert np.array_equal(z['idx'], te); P[a] = z['pred'] // nact
                nb = np.empty(len(te), int)
                for b in range(0, len(te), 512): nb[b:b + 512] = np.abs(Fk[te][b:b + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)
                P['knn10'] = ylab[tr][nb] // nact
                P['address'] = predict(KK, tr, te, ylab, NC, 'backoff')[0] // nact
                maj = np.bincount(ytr, minlength=na).argmax()
                for g in ('operator', 'service'):
                    v = G[g]; orc = np.empty(len(te), int); lk = np.empty(len(te), int)
                    for o in np.unique(v[te]):
                        m = v[te] == o; orc[m] = np.bincount(y[m], minlength=na).argmax()
                        mt = v[tr] == o; lk[m] = np.bincount(ytr[mt], minlength=na).argmax() if mt.any() else maj
                    P[f'{g}_oracle'] = orc; P[f'{g}_lookup'] = lk
                # shared operators: at least two classes with >= 5% of the operator's test connections
                v = G['operator'][te]; shared = np.zeros(len(te), bool); info = {}
                for o in np.unique(v):
                    m = v == o; sh = np.bincount(y[m], minlength=na) / m.sum(); big = np.where(sh >= 0.05)[0]
                    if len(big) >= 2:
                        shared |= m; info[o] = dict(n=int(m.sum()), classes={(names[c] if names else str(c)): round(100 * float(sh[c]), 1) for c in big}, largest=round(100 * float(sh.max()), 1))
                shared_info = info; st['n_test'].append(len(te)); st['n_shared'].append(int(shared.sum()))
                if K == 4 and s == 0:   # the attribution must reproduce results/cdn_attribution_archived.json
                    refn = ARCH_REF['accuracy_by_operator'][ds]['K4']
                    for o in np.unique(v): assert refn[f'operator:{o}']['n'] == int((v == o).sum()), (ds, o)
                for a, p in P.items():
                    st[f'{a}/acc'].append(100 * float((p == y).mean())); st[f'{a}/f1'].append(100 * f1_score(y, p, labels=list(range(na)), average='macro', zero_division=0))
                    if shared.any(): st[f'{a}/acc_shared'].append(100 * float((p[shared] == y[shared]).mean()))
                    if K == 4:
                        for o in info: st[f'{a}/op:{o}'].append(100 * float((p[v == o] == y[v == o]).mean()))
            res[f'K{K}'] = {k: round(float(np.mean(x)), 2) for k, x in st.items()}
            if K == 4: res['shared_operators_K4'] = shared_info
            r = res[f'K{K}']
            print(f'{ds} K={K} n={r["n_test"]:.0f} shared={r["n_shared"]:.0f}  ' + '  '.join(f'{a} {r[a + "/acc"]:.1f}/{r[a + "/f1"]:.1f}/{r.get(a + "/acc_shared", float("nan")):.1f}'
                  for a in ('operator_oracle', 'operator_lookup', 'service_oracle', 'address', 'lgbm', 'knn10', 'ours')), flush=True)
        if 'shared_operators_K4' in res:
            for o, i in res['shared_operators_K4'].items():
                print(f'   {ds} shared operator {o:12s} n={i["n"]:5d} largest {i["largest"]:5.1f}%  ' + '  '.join(f'{a} {res["K4"][a + "/op:" + o]:.1f}' for a in ('operator_oracle', 'service_oracle', 'address', 'lgbm', 'knn10', 'ours')) + f'  {i["classes"]}', flush=True)
        out[ds] = res
    return out


def background_addresses():
    """Server address, package, session and server name of every row of genai_background.npz (walk of build_background.py)."""
    os.makedirs(SCR, exist_ok=True); cf = os.path.join(SCR, 'genai_background_addr.json')
    if os.path.exists(cf): return json.load(open(cf, encoding='utf-8'))
    import pandas as pd
    from build_dataset import TARGET_PKG
    from build_background import MIN_PAY_PKTS
    RAW = os.path.join(ROOT, 'data', 'mirage2025genai'); ANY = set().union(*TARGET_PKG.values())
    ses = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'genai_sessions.csv')); file2sid = dict(zip(ses.file, ses.session_id))
    keys, pkgs, sids = [], [], []; t0 = time.time()
    for f in sorted(glob.glob(os.path.join(RAW, '*', '*', '*.json'))):
        rel = os.path.relpath(f, RAW).replace('\\', '/')
        if rel not in file2sid: continue
        d = json.load(open(f))
        for key, b in d.items():
            pkg = b['flow_metadata']['BF_label']
            if pkg in ANY: continue
            pdt = b['packet_data']; lp = pdt['L4_payload_bytes']; raw = pdt['L4_raw_payload']
            if sum(1 for i, l in enumerate(lp) if l > 0 and i < len(raw) and len(raw[i]) > 0) < MIN_PAY_PKTS: continue
            keys.append(key); pkgs.append(pkg); sids.append(int(file2sid[rel]))
    z = np.load(os.path.join(ROOT, 'data', 'derived', 'genai_background.npz'), allow_pickle=True)
    assert len(keys) == len(z['pkg']) and np.array_equal(np.array(pkgs), z['pkg']) and np.array_equal(np.array(sids), z['session_id']), 'walk does not reproduce genai_background.npz'
    r = dict(key=keys, pkg=pkgs, sid=sids); json.dump(r, open(cf, 'w', encoding='utf-8'))
    print(f'background walk {len(keys)} rows, {time.time() - t0:.0f}s', flush=True)
    return r


def host_of(pb, ps):
    for p in range(ps.shape[0]):
        s = int(ps[p, 0])
        if s < 0: continue
        b = bytes(pb[p])
        try:
            nl = int.from_bytes(b[s + 7:s + 9], 'big'); h = b[s + 9:s + 9 + nl].decode('ascii', 'replace').lower()
            if h: return h
        except Exception:
            pass
    return '<none>'


if __name__ == '__main__':
    AT = Attr(); OUT = dict(rules=__doc__.strip())
    OUT['closed_world'] = part_a(AT)
    # the shared operators that have a name (not 'other', i.e. no published list, nor private addresses) at K = 4:
    # accuracy weighted by their test connections (the paper reports it on CCMA)
    OUT['named_shared_operators_K4'] = {}
    for ds, res in OUT['closed_world'].items():
        sh = res.get('shared_operators_K4') or {}; ops = [o for o in sh if o not in ('other', 'private', 'unlisted', 'unknown')]
        n = sum(sh[o]['n'] for o in ops)
        if not n: continue
        r = dict(operators=ops, n=n, **{a: round(sum(sh[o]['n'] * res['K4'][f'{a}/op:{o}'] for o in ops) / n, 2)
                                        for a in ('operator_oracle', 'service_oracle', 'address', 'lgbm', 'knn10', 'ours')})
        OUT['named_shared_operators_K4'][ds] = r
        print(f'{ds} named shared operators {ops}: {n} test connections; ' + '  '.join(f'{a} {r[a]:.1f}' for a in ('operator_oracle', 'ours', 'lgbm', 'knn10', 'address')))
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'operator_confound.json'), 'w', encoding='utf-8'), indent=1)
    print('written results/operator_confound.json')
