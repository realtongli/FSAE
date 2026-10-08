"""Address-level reference observer and the address anonymity set of the assistants (training-free, descriptive).

Rules:

Server address. For every connection the server address is taken from the MIRAGE biflow key
'cliIP,cliPort,srvIP,srvPort,proto' (field 2). A key is flipped only if field 2 is a private/non-global address and
field 0 is a global one; IPv6 keys would be kept as full addresses with /48 and /32 standing in for /24 and /16. Every
such case (flips, IPv6, private server addresses) is counted and reported under data_checks.

(a) Closed world, both campaigns (GenAI 6 app x modality classes, CCMA 9 apps), K = 1, 2, 4, 8, seeds 0-4, the
    few-shot splits configs/splits/[ccma_]fewshot_k{K}_s{s}.json. Three lookup observers are fitted on the flows of
    the K labelled training sessions per class only (no validation sessions, nothing to select):
      ip       key = exact server address,
      p24      key = the server's /24 prefix,
      backoff  exact address if seen in training, else its /24 if seen, else its /16 if seen.
    Each key predicts the majority training class among the training flows with that key; ties go to the class with
    more training flows overall, then to the lower class index. A key never seen in training predicts the majority
    training class (same tie rule). Metrics are those of scripts/threat_stats.py: flow-level macro-F1 of the task, the
    same on 'exact' socket-attributed flows, on GenAI the assistant (3-class) and modality (2-class) macro-F1 derived
    from the 6-class prediction, and the session-level majority vote. Also reported: key coverage (share of test flows
    whose key occurs in training) and accuracy on covered flows. The pre-trained encoder ('ours', fs{K}_sslXL_s{s}) and
    LightGBM (fs{K}_lgbm_meta64_s{s}) are read from results/threat_stats.json and recomputed from their saved
    test_preds.npz (both must agree). For each observer the paired session-bootstrap 95% CI of 'ours minus observer'
    uses the resampling of threat_stats.py exactly (RandomState(12345), 2000 resamples of the test sessions, seed-
    averaged macro-F1); it is reported for the task macro-F1 and, on GenAI, for the assistant macro-F1.
(a2) The same three observers on the temporal splits configs/splits/[ccma_]temporal_k{K}_s{s}.json (K = 2, 4, 8; the
    test sessions are the latest 20% by capture time), next to the saved drift runs of ours (fs{K}_sslXL_drift_s{s})
    and LightGBM (fs{K}_drift_lgbm_meta64_s{s}). Same metrics and the same bootstrap.
(b) Shared infrastructure, GenAI campaign. Assistant connections = the evaluated target flows (genai_index rows with
    y_joint >= 0). The comparison population = every biflow of every raw JSON file under data/mirage2025genai (all
    242 captures, including the Telegram/WhatsApp controlled captures, and zero-payload biflows, since those also
    expose the address). For each assistant and each address level (exact address, /24, /16) and two scopes
    ('same_session' = another biflow of the same capture file; 'campaign' = any capture file) we report the share of
    the assistant's flows whose address/prefix is also contacted by a biflow of another package, split by package
    group (groups are defined by package name only):
      other_assistant  the other two assistants' packages,
      non_assistant    every package that is not one of the three assistant packages, subdivided into
        google_services  com.google.* (except com.google.android.googlequicksearchbox) and com.android.vending,
        browser          com.android.chrome, com.android.webview, org.lineageos.jelly,
        system           /system/bin/netd, system_server, com.android.*, android.*, org.lineageos.* (other than jelly),
        messaging        org.telegram.messenger, com.whatsapp.
    Anonymity set: for each assistant flow, the number of distinct other packages that contact the same address /
    prefix anywhere in the campaign, and the flow-weighted share of the biflows on that address/prefix that are the
    assistant's own (own package in its own captures). Also the number of distinct addresses and prefixes per
    assistant and the largest /16 prefixes (descriptive only).
(c) Gemini vs other Google traffic. Gemini runs inside the Google app (com.google.android.googlequicksearchbox), so the
    Google-app biflows captured in non-Gemini captures (ChatGPT, Copilot, Telegram, WhatsApp) form a separate group
    'google_app_other_captures'; with google_services and the Google browsers (com.android.chrome, com.android.webview)
    they make up 'google_all'. Same shares, levels and scopes as (b) for the Gemini flows.

Label-robust views of (b)/(c). MIRAGE attributes biflows to packages from netstat socket tables; on the assistants' main
back-end addresses, biflows labelled with Google Play services or the Google app carry the assistant's own ClientHello
server name (e.g. the ChatGPT API host), i.e. they are assistant connections attributed to another package, mostly with
BF_labeling_type 'most-common'. Such biflows make the package view overstate sharing. Therefore every (b)/(c) quantity is
reported in four views:
  package_all    the package view above, all biflows,
  package_exact  the same with the comparison population restricted to 'exact'-attributed biflows,
  server_name    label-free: an assistant flow's address/prefix counts as shared when some biflow of the campaign (any
                 package label; same capture file for 'same_session') on that address/prefix carries a parsed
                 ClientHello server name that never occurs among that assistant's own target flows. The anonymity set is
                 then the number of such distinct server names. This is a lower bound: only TLS-over-TCP ClientHellos in
                 the first five payload packets are parsed (QUIC Initials are encrypted), and the campaign contains only
                 the services the two phones contacted.
  server_name_own_captures  the same with the assistant's own names taken from every biflow attributed to its package
                 in its own captures (generic and controlled), not only from the evaluated target flows. In the
                 server_name view a single OpenAI image-host name, seen once in a ChatGPT-attributed biflow outside the
                 target set, marks a whole Cloudflare /16 as shared; this view counts such a name as the assistant's own.
                 Both variants are reported.
The number of other-package biflows that carry the assistant's most frequent own server name is reported as the size
of the attribution error (misattributed).
Writes results/ip_baseline.json."""
import collections, glob, ipaddress, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from build_dataset import find_sni_span, TARGET_PKG

B = 2000
ASSIST = ['Chatgpt', 'Copilot', 'Gemini']
APKG = {a: next(iter(TARGET_PKG[a])) for a in ASSIST}
GOOGLE_APP = APKG['Gemini']
OBS = ('ip', 'p24', 'backoff')
MISSING = []


# ---------------------------------------------------------------- addresses
def server_addr(key, chk):
    f = key.split(',')
    a0, a2 = ipaddress.ip_address(f[0]), ipaddress.ip_address(f[2])
    srv = a2
    if (not a2.is_global) and a0.is_global: srv = a0; chk['flipped'] += 1
    chk[f'v{srv.version}'] += 1
    if not srv.is_global: chk['server_not_global'] += 1
    if not ipaddress.ip_address(f[0]).is_private: chk['client_not_private'] += 1
    return srv


def prefix(addr, level):  # level 'ip' | 'p24' | 'p16'
    if level == 'ip': return str(addr)
    bits = {('p24', 4): 24, ('p16', 4): 16, ('p24', 6): 48, ('p16', 6): 32}[(level, addr.version)]
    return str(ipaddress.ip_network(f'{addr}/{bits}', strict=False))


def keys_of(addrs):
    return {lv: np.array([prefix(a, lv) for a in addrs], dtype=object) for lv in ('ip', 'p24', 'p16')}


# ---------------------------------------------------------------- lookup observers
def fit_lookup(k_tr, y_tr, NC):
    prior = np.bincount(y_tr, minlength=NC)
    maj = int(max(range(NC), key=lambda c: (prior[c], -c)))
    cnt = collections.defaultdict(lambda: np.zeros(NC, np.int64))
    for k, y in zip(k_tr, y_tr): cnt[k][y] += 1
    table = {k: int(max(range(NC), key=lambda c: (v[c], prior[c], -c))) for k, v in cnt.items()}
    return table, maj


def predict(K, tr, te, y, NC, obs):
    """-> (pred, covered) for observer obs; K = dict of key arrays over the dataset."""
    if obs in ('ip', 'p24'):
        t, maj = fit_lookup(K[obs][tr], y[tr], NC)
        cov = np.array([k in t for k in K[obs][te]])
        return np.array([t.get(k, maj) for k in K[obs][te]]), cov
    tabs = {lv: fit_lookup(K[lv][tr], y[tr], NC)[0] for lv in ('ip', 'p24', 'p16')}
    maj = fit_lookup(K['ip'][tr], y[tr], NC)[1]
    pred = np.empty(len(te), int); cov = np.zeros(len(te), bool)
    for j, i in enumerate(te):
        for lv in ('ip', 'p24', 'p16'):
            if K[lv][i] in tabs[lv]: pred[j] = tabs[lv][K[lv][i]]; cov[j] = True; break
        else: pred[j] = maj
    return pred, cov


def macro_f1_cm(cm):  # identical to scripts/threat_stats.py
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float); denom = cm.sum(-1) + cm.sum(-2)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0); present = denom > 0
    return (f1 * present).sum(-1) / present.sum(-1)


def run_npz(ds, name, K, s, drift):
    pre = '' if ds == 'genai' else 'ccma_'
    if name == 'ours': r = f'{pre}fs{K}_sslXL_drift_s{s}' if drift else f'{pre}fs{K}_sslXL_s{s}'
    else: r = f'{pre}fs{K}_drift_lgbm_meta64_s{s}' if drift else f'{pre}fs{K}_lgbm_meta64_s{s}'
    p = os.path.join(ROOT, 'results', 'runs', r, 'test_preds.npz')
    if not os.path.exists(p): MISSING.append(r); return None
    return np.load(p)


def evaluate(D, ds, K_keys, split_fmt, Ks, drift):
    ylab = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app; nact = 2 if ds == 'genai' else 1
    exact = D.index['labeling_type'].values == 'exact'
    out = {}
    for K in Ks:
        preds, covs, test_idx, same_test = {}, {}, None, True
        for s in range(5):
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', split_fmt.format(K=K, s=s))); tr, te = idx['train'], idx['test']
            if test_idx is None: test_idx = te
            same_test &= np.array_equal(test_idx, te)
            for o in OBS: preds[(o, s)], covs[(o, s)] = predict(K_keys, tr, te, ylab, NC, o)
            for a in ('ours', 'lgbm'):
                z = run_npz(ds, a, K, s, drift)
                if z is None: continue
                assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], ylab[te]), (ds, a, K, s)
                preds[(a, s)] = z['pred']
        assert same_test, 'test sessions differ across seeds'
        te = test_idx; y = ylab[te]; sid = D.session_id[te]; us, inv = np.unique(sid, return_inverse=True); nS = len(us)
        ys = np.array([np.bincount(y[inv == j]).argmax() for j in range(nS)]); ex = exact[te]
        W = np.random.RandomState(12345).multinomial(nS, np.ones(nS) / nS, size=B)
        res, boot, boot_app = {}, {}, {}
        for a in OBS + ('ours', 'lgbm'):
            if not all((a, s) in preds for s in range(5)): continue
            st = collections.defaultdict(list); bs, bsa = [], []
            for s in range(5):
                pr = preds[(a, s)]
                cm = np.zeros((nS, NC, NC), np.int64); np.add.at(cm, (inv, y, pr), 1); bs.append(macro_f1_cm(np.tensordot(W, cm, axes=(1, 0))))
                st['flow'].append(100 * f1_score(y, pr, average='macro')); st['flow_exact'].append(100 * f1_score(y[ex], pr[ex], average='macro'))
                ps = np.array([np.bincount(pr[inv == j], minlength=NC).argmax() for j in range(nS)])
                st['sess_acc'].append(100 * accuracy_score(ys, ps))
                if nact > 1:
                    na = NC // nact
                    cma = np.zeros((nS, na, na), np.int64); np.add.at(cma, (inv, y // nact, pr // nact), 1); bsa.append(macro_f1_cm(np.tensordot(W, cma, axes=(1, 0))))
                    st['app'].append(100 * f1_score(y // nact, pr // nact, average='macro')); st['act'].append(100 * f1_score(y % nact, pr % nact, average='macro'))
                    for c, nm in enumerate(ASSIST): st[f'app_f1_{nm}'].append(100 * f1_score((y // nact == c).astype(int), (pr // nact == c).astype(int)))
                    pa = np.array([np.bincount(pr[inv == j] // nact, minlength=na).argmax() for j in range(nS)])
                    st['sess_app_acc'].append(100 * accuracy_score(ys // nact, pa))
                if a in OBS:
                    cv = covs[(a, s)]; st['coverage'].append(100 * cv.mean())
                    st['acc_covered'].append(100 * float((pr[cv] == y[cv]).mean()) if cv.any() else float('nan'))
                    if nact > 1: st['app_acc_covered'].append(100 * float((pr[cv] // nact == y[cv] // nact).mean()) if cv.any() else float('nan'))
            boot[a] = 100 * np.mean(bs, 0)
            if bsa: boot_app[a] = 100 * np.mean(bsa, 0)
            res[a] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in st.items()}
        if 'ours' in res:
            for a in OBS + ('lgbm',):
                if a not in res: continue
                lo, hi = np.percentile(boot['ours'] - boot[a], [2.5, 97.5])
                res[a]['ours_minus'] = dict(flow=float(res['ours']['flow'][0] - res[a]['flow'][0]), ci=[float(lo), float(hi)], sig=bool(lo > 0 or hi < 0))
                if a in boot_app:
                    lo, hi = np.percentile(boot_app['ours'] - boot_app[a], [2.5, 97.5])
                    res[a]['ours_minus']['app'] = float(res['ours']['app'][0] - res[a]['app'][0]); res[a]['ours_minus']['app_ci'] = [float(lo), float(hi)]
                    res[a]['ours_minus']['app_sig'] = bool(lo > 0 or hi < 0)
        out[f'K{K}'] = dict(n_test_sessions=int(nS), n_test_flows=int(len(te)), observers=res)
        print(f'--- {ds} {"drift" if drift else "fewshot"} K={K} (test sessions {nS}, flows {len(te)})')
        for a, r in res.items():
            line = f'  {a:8s} F1 {r["flow"][0]:5.1f}±{r["flow"][1]:3.1f}'
            if 'app' in r: line += f'  app {r["app"][0]:5.1f} act {r["act"][0]:5.1f} sessApp {r["sess_app_acc"][0]:5.1f}'
            line += f'  sessAcc {r["sess_acc"][0]:5.1f}'
            if 'coverage' in r: line += f'  cov {r["coverage"][0]:5.1f} accCov {r["acc_covered"][0]:5.1f}'
            if 'ours_minus' in r:
                m = r['ours_minus']; line += f'  ours-{a} {m["flow"]:+5.1f} [{m["ci"][0]:+5.1f},{m["ci"][1]:+5.1f}]'
                if 'app' in m: line += f' app {m["app"]:+5.1f} [{m["app_ci"][0]:+5.1f},{m["app_ci"][1]:+5.1f}]'
            print(line)
    return out


# ---------------------------------------------------------------- raw GenAI scan for (b), (c)
def sni_host(raw_list, lp):
    n = 0
    for i, l in enumerate(lp):
        if l <= 0 or i >= len(raw_list) or not raw_list[i]: continue
        b = bytes(raw_list[i]); sp = find_sni_span(b)
        if sp is not None:
            s = sp[0]
            try:
                nl = int.from_bytes(b[s + 7:s + 9], 'big'); h = b[s + 9:s + 9 + nl].decode('ascii', 'replace').lower()
                return h or '<none>'
            except Exception: return '<none>'
        n += 1
        if n >= 5: break
    return '<none>'


def pkg_group(p):
    if p in APKG.values(): return 'assistant'
    if p == 'com.android.vending' or (p.startswith('com.google.') and p != GOOGLE_APP): return 'google_services'
    if p in ('com.android.chrome', 'com.android.webview', 'org.lineageos.jelly'): return 'browser'
    if p in ('org.telegram.messenger', 'com.whatsapp'): return 'messaging'
    if p in ('/system/bin/netd', 'system_server') or p.startswith('com.android.') or p.startswith('android.') or p.startswith('org.lineageos.'): return 'system'
    return 'other'


def scan_genai(D, chk):
    raw = os.path.join(ROOT, 'data', 'mirage2025genai'); rows = []
    for f in sorted(glob.glob(os.path.join(raw, '*', '*', '*.json'))):
        rel = os.path.relpath(f, raw).replace('\\', '/'); part, cls_dir, _ = rel.split('/')
        app_dir = cls_dir.split('_')[0] if part == 'generic' else cls_dir
        for key, b in json.load(open(f)).items():
            pdt = b['packet_data']; md = b['flow_metadata']; srv = server_addr(key, chk)
            rows.append(dict(file=rel, capture_app=app_dir, key=key, pkg=md['BF_label'], ltype=md['BF_labeling_type'], srv=srv,
                             host=sni_host(pdt['L4_raw_payload'], pdt['L4_payload_bytes'])))
    R = pd.DataFrame(rows)
    for lv in ('ip', 'p24', 'p16'): R[lv] = [prefix(a, lv) for a in R.srv]
    R['group'] = R.pkg.map(pkg_group)
    return R


LEVELS = ('ip', 'p24', 'p16')


def key_sets(R, lv, mask):
    """-> (campaign key set, {file: key set}) of the biflows selected by mask."""
    camp = set(R[lv].values[mask]); bys = collections.defaultdict(set)
    for f, k in zip(R.file.values[mask], R[lv].values[mask]): bys[f].add(k)
    return camp, bys


def shared_share(Ta, lv, camp, bys):
    m_c = np.array([k in camp for k in Ta[lv]]); m_s = np.array([k in bys.get(f, ()) for f, k in zip(Ta.file, Ta[lv])])
    return dict(campaign=float(m_c.mean()), same_session=float(m_s.mean()), campaign_key_share=float(np.mean([k in camp for k in Ta[lv].unique()])))


def dist(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), share_ge1=float((x >= 1).mean()), share_ge2=float((x >= 2).mean()),
                share_ge5=float((x >= 5).mean()))


def shared_infra(D, R):
    ses = D.sessions.set_index('session_id').file
    T = D.index[D.index.y_joint >= 0].copy(); T['file'] = T.session_id.map(ses)
    look = {(f, k): i for i, (f, k) in enumerate(zip(R.file, R.key))}
    T['r'] = [look[(f, k)] for f, k in zip(T.file, T.biflow_key)]
    assert (R.pkg.values[T.r.values] == T.bf_label.values).all()
    for lv in LEVELS: T[lv] = R[lv].values[T.r.values]
    exact = R.ltype.values == 'exact'; everything = np.ones(len(R), bool)

    def groups_for(a):  # package groups relative to assistant a (package names only)
        own = APKG[a]; g = {}; is_own = R.pkg.values == own
        g['any_other_pkg'] = ~is_own
        g['other_assistant'] = (R.group.values == 'assistant') & ~is_own
        g['non_assistant'] = R.group.values != 'assistant'
        for gg in ('google_services', 'browser', 'system', 'messaging', 'other'): g[gg] = R.group.values == gg
        if a == 'Gemini':
            g['google_app_other_captures'] = is_own & (R.capture_app.values != 'Gemini')
            g['google_all'] = g['google_app_other_captures'] | g['google_services'] | np.isin(R.pkg.values, ['com.android.chrome', 'com.android.webview'])
        return g

    out = {}
    for ai, a in enumerate(ASSIST):
        Ta = T[T.y_app == ai]; G = groups_for(a); own = APKG[a]
        own_names = collections.Counter(h for h in R.host.values[Ta.r.values] if h != '<none>'); S_a = set(own_names)
        main_name = own_names.most_common(1)[0][0]
        mis = (R.pkg.values != own) & (R.host.values == main_name)
        ra = dict(n_flows=int(len(Ta)), package=own, distinct={lv: int(Ta[lv].nunique()) for lv in LEVELS},
                  top_p16=[[k, int(v), float(v / len(Ta))] for k, v in Ta.p16.value_counts().head(6).items()],
                  own_flows_with_server_name=float(np.mean(R.host.values[Ta.r.values] != '<none>')), n_own_server_names=len(S_a),
                  most_frequent_own_server_name=main_name,
                  misattributed=dict(n_other_pkg_biflows_with_main_name=int(mis.sum()), by_labeling_type=dict(collections.Counter(R.ltype.values[mis])),
                                     by_package=collections.Counter(R.pkg.values[mis]).most_common(6), by_capture=dict(collections.Counter(R.capture_app.values[mis])),
                                     n_own_flows_with_main_name=int(own_names[main_name])))
        views = {}
        for vname, pop in (('package_all', everything), ('package_exact', exact)):
            v = {}
            for lv in LEVELS:
                v[lv] = {'shared': {g: shared_share(Ta, lv, *key_sets(R, lv, gm & pop)) for g, gm in G.items()}}
                pk = collections.defaultdict(set)  # anonymity set: distinct other packages on the same key, anywhere
                for k, p, cap in zip(R[lv].values[pop], R.pkg.values[pop], R.capture_app.values[pop]):
                    if p != own: pk[k].add(p)
                    elif a == 'Gemini' and cap != 'Gemini': pk[k].add('google_app_other_captures')
                v[lv]['n_other_packages'] = dist([len(pk.get(k, ())) for k in Ta[lv]])
                v[lv]['n_non_assistant_packages'] = dist([len({p for p in pk.get(k, ()) if p not in APKG.values() and p != 'google_app_other_captures'}) for k in Ta[lv]])
                if vname == 'package_all':  # flow-weighted share of the biflows on the key that are the assistant's own
                    tot = collections.Counter(R[lv].values); ownc = collections.Counter(R[lv].values[(R.pkg.values == own) & (R.capture_app.values == a)])
                    v[lv]['own_share_flow_weighted'] = float(np.mean([ownc[k] / tot[k] for k in Ta[lv]]))
            views[vname] = v
        S_own = set(R.host.values[(R.pkg.values == own) & (R.capture_app.values == a)]) - {'<none>'}
        ra['n_own_server_names_own_captures'] = len(S_own)
        for vname, S in (('server_name', S_a), ('server_name_own_captures', S_own)):
            v = {}
            named_other = (R.host.values != '<none>') & ~np.isin(R.host.values, list(S))
            for lv in LEVELS:
                camp, bys = key_sets(R, lv, named_other)
                nm = collections.defaultdict(set)
                for k, h in zip(R[lv].values[named_other], R.host.values[named_other]): nm[k].add(h)
                keys = set(Ta[lv]); on_keys = np.isin(R[lv].values, list(keys))
                v[lv] = dict(shared=shared_share(Ta, lv, camp, bys), n_other_server_names=dist([len(nm.get(k, ())) for k in Ta[lv]]),
                             named_share_of_biflows_on_keys=float(np.mean(R.host.values[on_keys] != '<none>')),
                             top_other_server_names=collections.Counter(R.host.values[on_keys & named_other]).most_common(10),
                             shared_keys=sorted(([k, int((Ta[lv] == k).sum()), sorted(nm[k])[:6]] for k in keys if k in nm), key=lambda x: -x[1])[:8])
            views[vname] = v
        ra['views'] = views; out[a] = ra
        print(f'\n=== shared infrastructure: {a} ({own}), {len(Ta)} flows, distinct {ra["distinct"]}, main name {main_name}, '
              f'other-pkg biflows with it {int(mis.sum())} {ra["misattributed"]["by_labeling_type"]}')
        show = ('any_other_pkg', 'non_assistant', 'google_services', 'google_app_other_captures', 'google_all')
        for lv in LEVELS:
            for vn in ('package_all', 'package_exact'):
                sh = views[vn][lv]['shared']
                print(f'  {lv:4s} {vn:13s} ' + '  '.join(f'{g} {100*sh[g]["campaign"]:.1f}/{100*sh[g]["same_session"]:.1f}' for g in show if g in sh)
                      + f'  #pkgs mean {views[vn][lv]["n_other_packages"]["mean"]:.2f}' + (f'  ownShare {100*views[vn][lv]["own_share_flow_weighted"]:.1f}' if vn == 'package_all' else ''))
            for vn in ('server_name', 'server_name_own_captures'):
                sn = views[vn][lv]
                print(f'  {lv:4s} {vn[:16]:16s} campaign {100*sn["shared"]["campaign"]:.1f} same_session {100*sn["shared"]["same_session"]:.1f}  '
                      f'#names mean {sn["n_other_server_names"]["mean"]:.2f} median {sn["n_other_server_names"]["median"]:.0f}  named {100*sn["named_share_of_biflows_on_keys"]:.1f}%  top {sn["top_other_server_names"][:3]}')
    return out


def main():
    t0 = time.time(); OUT = dict(rules=__doc__.split('Writes')[0].strip(), data_checks={}, closed_world={}, drift={})
    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); chk = collections.Counter()
        addrs = [server_addr(k, chk) for k in D.index.biflow_key.values]
        cli = pd.Series([k.split(',')[0] for k in D.index.biflow_key.values]).value_counts()
        chk_d = dict(chk); chk_d['client_addresses'] = {k: int(v) for k, v in cli.items()}
        lab = (D.y_joint >= 0)
        chk_d['labelled_flows_server_not_global_by_proto'] = pd.Series(D.index.proto.values[lab & np.array([not a.is_global for a in addrs])]).value_counts().rename(index=str).to_dict()
        OUT['data_checks'][f'{ds}_index'] = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in chk_d.items()}
        KK = keys_of(addrs)
        pre = '' if ds == 'genai' else 'ccma_'
        OUT['closed_world'][ds] = evaluate(D, ds, KK, pre + 'fewshot_k{K}_s{s}.json', (1, 2, 4, 8), False)
        OUT['drift'][ds] = evaluate(D, ds, KK, pre + 'temporal_k{K}_s{s}.json', (2, 4, 8), True)
        # agreement with results/threat_stats.json
        ts = json.load(open(os.path.join(ROOT, 'results', 'threat_stats.json')))
        for K in (1, 2, 4, 8):
            r = OUT['closed_world'][ds][f'K{K}']['observers']
            for a in ('ours', 'lgbm'):
                if a in r:
                    t = ts[f'{ds}_K{K}']['attackers'][a]
                    assert abs(t['flow'][0] - r[a]['flow'][0]) < 1e-6, (ds, K, a)
                    r[a]['from_threat_stats'] = {k: t[k] for k in ('flow', 'flow_exact', 'app', 'act', 'sess_acc', 'sess_app_acc') if k in t}
        if ds == 'genai':
            chk = collections.Counter(); R = scan_genai(D, chk)
            OUT['data_checks']['genai_raw'] = dict(chk, n_biflows=int(len(R)), n_files=int(R.file.nunique()), n_packages=int(R.pkg.nunique()),
                                                   client_addresses={k: int(v) for k, v in pd.Series([k.split(',')[0] for k in R.key]).value_counts().items()},
                                                   package_groups={g: sorted(set(R.pkg[R.group == g])) for g in sorted(R.group.unique())},
                                                   biflows_per_group={g: int(v) for g, v in R.group.value_counts().items()},
                                                   biflows_per_labeling_type={g: int(v) for g, v in R.ltype.value_counts().items()},
                                                   biflows_with_server_name=float(np.mean(R.host.values != '<none>')))
            OUT['shared_infra_genai'] = shared_infra(D, R)
    OUT['missing_npz'] = MISSING
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'ip_baseline.json'), 'w'), indent=1, default=str)
    print(f'\nmissing npz: {MISSING}\nwritten results/ip_baseline.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
