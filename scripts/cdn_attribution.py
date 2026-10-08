"""Which operators' address space the assistants' (and the CCMA apps') connections go to, and the address observer's
rows for the assistant-identity table / the main comparison table (descriptive; nothing is fitted or selected here).

Rules:

Server address. As in scripts/ip_baseline.py (server_addr): field 2 of the MIRAGE biflow key, flipped only if it is
non-global and field 0 is global. Populations: GenAI 'evaluated' = every flow with y_joint >= 0 (the flows of the six
app x modality classes, 6,501); GenAI 'test' = the test flows of configs/splits/fewshot_k*_s*.json (identical for all
K and seeds, asserted); CCMA 'evaluated' = every flow with y_app >= 0; CCMA 'test' = test flows of ccma_fewshot_k*.

Published lists (read over HTTPS into memory at run time; URL, access time (UTC), SHA-256, size, the list's own
timestamp and prefix count are recorded; the raw files are NOT written to disk):
  Cloudflare  https://www.cloudflare.com/ips-v4
  Fastly      https://api.fastly.com/public-ip-list
  AWS         https://ip-ranges.amazonaws.com/ip-ranges.json            (service CLOUDFRONT vs the rest)
  Google      https://www.gstatic.com/ipranges/goog.json and cloud.json (goog minus cloud = Google's own services)
  Microsoft   13.107.0.0/16 (Microsoft-documented edge block, fixed) and, if reachable, the Azure service tags
              (ServiceTags_Public_*.json linked from https://www.microsoft.com/en-us/download/details.aspx?id=56519):
              tag AzureFrontDoor.Frontend, then AzureCloud.
  Registry    RDAP (RIR chosen by the IANA bootstrap https://data.iana.org/rdap/ipv4.json). GenAI: one query per
              distinct /24 (its lowest observed address), for every /24, as an independent check of the lists.
              CCMA: only addresses no published list matches, one query per /24, reusing a returned registration
              when it contains the address. Akamai publishes no list; it is identified only by RDAP.
Operator of an address, first rule that applies:
  1 non-global -> 'private'; 2 Cloudflare list -> Cloudflare; 3 Fastly list -> Fastly; 4 AWS CLOUDFRONT prefix ->
  Amazon/CloudFront, other AWS prefix -> Amazon/AWS; 5 Google cloud.json -> Google/Cloud, else goog.json ->
  Google/own; 6 Azure tag AzureFrontDoor.Frontend -> Microsoft/AzureFrontDoor, else 13.107.0.0/16 ->
  Microsoft/edge-13.107, else AzureCloud -> Microsoft/Azure; 7 RDAP registrant or network name, by keyword (akamai ->
  Akamai; microsoft|msft -> Microsoft; amazon|aws -> Amazon; google -> Google; cloudflare -> Cloudflare; fastly ->
  Fastly; edgecast|edgio|verizon -> Edgio; anything else -> 'other' with the RDAP name kept); 8 no RDAP answer ->
  'unknown'. RDAP/list disagreements on the GenAI /24s are reported, not resolved by hand.
Front-end class, defined from the operators' own documentation, not from these data:
  shared_cdn        addresses that front many unrelated organisations: Cloudflare (anycast reverse proxy), Fastly,
                    Amazon CloudFront, Akamai, Azure Front Door, Edgio;
  operator_edge     addresses shared by one operator's own services: Google/own (Google Front End), Microsoft
                    edge-13.107 when not tagged AzureFrontDoor.Frontend;
  cloud_tenant      addresses of a cloud customer's VM or load balancer: Amazon/AWS, Google/Cloud, Microsoft/Azure;
  operator_other    addresses that RDAP gives to Amazon, Google or Microsoft but that none of their lists contains;
  other / private / unknown.
Reported per assistant (per CCMA app): share of connections per operator, per operator/service, per front-end class,
and the matched published prefixes; the same on the test flows.

Per-operator accuracy (descriptive): on the GenAI test flows, for K = 1, 2, 4, 8, the assistant accuracy of the
pre-trained encoder (fs{K}_sslXL_s{s}), LightGBM (fs{K}_lgbm_meta64_s{s}) and the address back-off lookup
(scripts/ip_baseline.py, recomputed), grouped by the operator of the server address; mean and range over seeds 0-4.

(b) Address-observer rows: the back-off lookup ('backoff' in results/ip_baseline.json), recomputed per seed with the
ip_baseline.py functions so that seed ranges can be given; the recomputed means must equal ip_baseline.json (asserted).
Reported: GenAI assistant macro-F1, modality macro-F1, six-class macro-F1, session accuracy (task) and session
assistant accuracy for K = 1, 2, 4, 8; CCMA nine-class macro-F1 and session accuracy; next to the encoder and
LightGBM (recomputed from test_preds.npz; means must equal results/threat_stats.json), and the paired session-bootstrap
differences 'encoder minus back-off' already stored in ip_baseline.json.
Writes results/cdn_attribution.json."""
import collections, datetime, hashlib, ipaddress, json, os, re, sys, time, urllib.request
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from ip_baseline import server_addr, keys_of, predict

UA = {'User-Agent': 'Mozilla/5.0 (research; address-attribution)'}
ASSIST = ['Chatgpt', 'Copilot', 'Gemini']
SOURCES = {}
FRONTEND = {'Cloudflare': 'shared_cdn', 'Fastly': 'shared_cdn', 'Amazon/CloudFront': 'shared_cdn', 'Akamai': 'shared_cdn',
            'Microsoft/AzureFrontDoor': 'shared_cdn', 'Edgio': 'shared_cdn', 'Google/own': 'operator_edge',
            'Microsoft/edge-13.107': 'operator_edge', 'Amazon/AWS': 'cloud_tenant', 'Google/Cloud': 'cloud_tenant',
            'Microsoft/Azure': 'cloud_tenant', 'Microsoft/other': 'operator_other', 'Amazon/other': 'operator_other',
            'Google/other': 'operator_other', 'private': 'private', 'unknown': 'unknown'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch(url, tries=4):
    err = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read(), r.geturl()
        except Exception as e:
            err = e
            if getattr(e, 'code', None) == 404: break
            time.sleep(2 * (t + 1))
    raise err


def record(name, url, body, stamp, n):
    SOURCES[name] = dict(url=url, accessed_utc=now(), sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), list_timestamp=stamp, n_ipv4_prefixes=n, status='ok')


def nets(lst):
    return [ipaddress.ip_network(p) for p in lst]


def load_lists():
    L = {}
    try:
        b, _ = fetch('https://www.cloudflare.com/ips-v4'); p = [x.strip() for x in b.decode().split() if x.strip()]
        L['Cloudflare'] = nets(p); record('Cloudflare', 'https://www.cloudflare.com/ips-v4', b, None, len(p))
    except Exception as e: SOURCES['Cloudflare'] = dict(status=f'failed: {e}')
    try:
        b, _ = fetch('https://api.fastly.com/public-ip-list'); p = json.loads(b)['addresses']
        L['Fastly'] = nets(p); record('Fastly', 'https://api.fastly.com/public-ip-list', b, None, len(p))
    except Exception as e: SOURCES['Fastly'] = dict(status=f'failed: {e}')
    try:
        b, _ = fetch('https://ip-ranges.amazonaws.com/ip-ranges.json'); j = json.loads(b)
        cf = sorted({x['ip_prefix'] for x in j['prefixes'] if x['service'] == 'CLOUDFRONT'}); al = sorted({x['ip_prefix'] for x in j['prefixes']})
        L['Amazon/CloudFront'] = nets(cf); L['Amazon/AWS'] = nets(al)
        record('AWS', 'https://ip-ranges.amazonaws.com/ip-ranges.json', b, dict(createDate=j.get('createDate'), syncToken=j.get('syncToken')), len(al))
        SOURCES['AWS']['n_cloudfront_ipv4_prefixes'] = len(cf)
    except Exception as e: SOURCES['AWS'] = dict(status=f'failed: {e}')
    for nm, url in (('Google/Cloud', 'https://www.gstatic.com/ipranges/cloud.json'), ('Google/own', 'https://www.gstatic.com/ipranges/goog.json')):
        try:
            b, _ = fetch(url); j = json.loads(b); p = [x['ipv4Prefix'] for x in j['prefixes'] if 'ipv4Prefix' in x]
            L[nm] = nets(p); record(nm, url, b, dict(creationTime=j.get('creationTime'), syncToken=j.get('syncToken')), len(p))
        except Exception as e: SOURCES[nm] = dict(status=f'failed: {e}')
    L['Microsoft/edge-13.107'] = nets(['13.107.0.0/16']); SOURCES['Microsoft/edge-13.107'] = dict(url='fixed: Microsoft-documented edge block 13.107.0.0/16', status='ok')
    try:
        page, _ = fetch('https://www.microsoft.com/en-us/download/details.aspx?id=56519')
        m = re.search(rb'https://download\.microsoft\.com/download/[^"\'\s]+?ServiceTags_Public_\d+\.json', page)
        if not m: raise RuntimeError('ServiceTags link not found on the download page')
        url = m.group(0).decode(); b, _ = fetch(url); j = json.loads(b); tags = {v['name']: v['properties']['addressPrefixes'] for v in j['values']}
        afd = [p for p in tags.get('AzureFrontDoor.Frontend', []) if ':' not in p]; az = [p for p in tags.get('AzureCloud', []) if ':' not in p]
        L['Microsoft/AzureFrontDoor'] = nets(afd); L['Microsoft/Azure'] = nets(az)
        record('Azure service tags', url, b, dict(changeNumber=j.get('changeNumber'), cloud=j.get('cloud')), len(afd) + len(az))
        SOURCES['Azure service tags'].update(n_afd_frontend=len(afd), n_azurecloud=len(az))
    except Exception as e: SOURCES['Azure service tags'] = dict(status=f'failed: {e}')
    return L


def match(L, name, a):
    best = None
    for n in L.get(name, ()):
        if a in n and (best is None or n.prefixlen > best.prefixlen): best = n
    return best


# ---------------------------------------------------------------- RDAP
class RDAP:
    def __init__(self):
        b, _ = fetch('https://data.iana.org/rdap/ipv4.json'); j = json.loads(b)
        self.boot = [(ipaddress.ip_network(p), urls[0]) for pfx, urls in j['services'] for p in pfx]
        SOURCES['IANA RDAP bootstrap'] = dict(url='https://data.iana.org/rdap/ipv4.json', accessed_utc=now(), sha256=hashlib.sha256(b).hexdigest(), status='ok')
        self.cache = []; self.n_queries = 0; self.failures = []

    def lookup(self, a, reuse=True):
        if reuse:
            for r in self.cache:
                if r['_lo'] <= int(a) <= r['_hi']: return r
        base = max(((n, u) for n, u in self.boot if a in n), key=lambda x: x[0].prefixlen, default=(None, None))[1]
        if base is None: self.failures.append(str(a)); return None
        try:
            b, final = fetch(base.rstrip('/') + '/ip/' + str(a)); j = json.loads(b); self.n_queries += 1; time.sleep(0.3)
        except Exception as e:
            self.failures.append(f'{a}: {e}'); return None
        fns = []

        def walk(ents, depth=0):
            for e in ents or []:
                roles = e.get('roles', []); fn = [x[3] for x in (e.get('vcardArray') or [None, []])[1] if x[0] == 'fn']
                if 'registrant' in roles or depth == 0 and not fns: fns.extend([f'{"/".join(roles)}:{f}' for f in fn])
                walk(e.get('entities'), depth + 1)
        walk(j.get('entities'))
        desc = [d for rm in j.get('remarks', []) or [] for d in rm.get('description', [])][:4]
        lo, hi = j.get('startAddress'), j.get('endAddress')
        try: lo_i, hi_i = int(ipaddress.ip_address(lo)), int(ipaddress.ip_address(hi))
        except Exception: lo_i = hi_i = int(a)
        r = dict(query=str(a), server=final.split('/ip/')[0], handle=j.get('handle'), name=j.get('name'), start=lo, end=hi, registrant=fns[:4], remarks=desc, _lo=lo_i, _hi=hi_i)
        self.cache.append(r); return r


KEYWORDS = [('akamai', 'Akamai'), ('microsoft', 'Microsoft'), ('msft', 'Microsoft'), ('amazon', 'Amazon'), ('aws', 'Amazon'),
            ('google', 'Google'), ('cloudflare', 'Cloudflare'), ('fastly', 'Fastly'), ('edgecast', 'Edgio'), ('edgio', 'Edgio'), ('verizon', 'Edgio')]


def rdap_operator(r):
    if r is None: return 'unknown'
    txt = ' '.join([str(r.get('name') or ''), str(r.get('handle') or '')] + list(r.get('registrant') or []) + list(r.get('remarks') or [])).lower()
    for k, op in KEYWORDS:
        if k in txt: return op
    return 'other'


def attribute(L, a, rd, force_rdap):
    """-> dict(operator, service, frontend, matched_prefix, source, rdap)"""
    r = None
    if force_rdap: r = rd.lookup(a, reuse=False)
    if not a.is_global: out = dict(operator='private', service='private', matched_prefix=None, source='address')
    else:
        out = None
        for svc in ('Cloudflare', 'Fastly', 'Amazon/CloudFront', 'Amazon/AWS', 'Google/Cloud', 'Google/own', 'Microsoft/AzureFrontDoor', 'Microsoft/edge-13.107', 'Microsoft/Azure'):
            m = match(L, svc, a)
            if m is not None:
                out = dict(operator=svc.split('/')[0], service=svc, matched_prefix=str(m), source='published list'); break
        if out is None:
            if r is None: r = rd.lookup(a, reuse=True)
            op = rdap_operator(r)
            svc = {'Akamai': 'Akamai', 'Edgio': 'Edgio', 'Cloudflare': 'Cloudflare', 'Fastly': 'Fastly', 'Microsoft': 'Microsoft/other',
                   'Amazon': 'Amazon/other', 'Google': 'Google/other', 'other': 'other', 'unknown': 'unknown'}[op]
            out = dict(operator=op, service=svc, matched_prefix=(f'{r["start"]}-{r["end"]}' if r else None), source='RDAP')
    out['frontend'] = FRONTEND.get(out['service'], 'other')
    if r is not None: out['rdap'] = {k: v for k, v in r.items() if not k.startswith('_')}; out['rdap_operator'] = rdap_operator(r)
    return out


def shares(values):
    c = collections.Counter(values); n = sum(c.values())
    return {k: [int(v), round(100 * v / n, 2)] for k, v in c.most_common()}


def describe(D, ds, A, attr, rows):
    lab = (D.y_joint >= 0) if ds == 'genai' else (D.y_app >= 0)
    names = ASSIST if ds == 'genai' else [str(D.index['app'].values[np.where(lab & (D.y_app == c))[0][0]]) for c in range(D.n_app)]
    out = {}
    for pop, sel in rows.items():
        res = {}
        for c, nm in enumerate(names):
            m = sel & (D.y_app == c)
            ops = [attr[str(A[i])]['operator'] for i in np.where(m)[0]]; svc = [attr[str(A[i])]['service'] for i in np.where(m)[0]]
            fe = [attr[str(A[i])]['frontend'] for i in np.where(m)[0]]; mp = [f"{attr[str(A[i])]['service']} {attr[str(A[i])]['matched_prefix']}" for i in np.where(m)[0]]
            res[nm] = dict(n_flows=int(m.sum()), operator=shares(ops), service=shares(svc), frontend=shares(fe), top_matched=shares(mp) if ds == 'genai' else dict(list(shares(mp).items())[:8]))
        allm = sel
        res['_all'] = dict(n_flows=int(allm.sum()), operator=shares([attr[str(A[i])]['operator'] for i in np.where(allm)[0]]),
                           frontend=shares([attr[str(A[i])]['frontend'] for i in np.where(allm)[0]]))
        out[pop] = res
    return out


# ---------------------------------------------------------------- per-seed observer metrics
def metrics(y, pr, sid, nact):
    us, inv = np.unique(sid, return_inverse=True); nS = len(us)
    ys = np.array([np.bincount(y[inv == j]).argmax() for j in range(nS)]); NC = int(max(y.max(), pr.max()) + 1)
    ps = np.array([np.bincount(pr[inv == j], minlength=NC).argmax() for j in range(nS)])
    r = dict(flow=100 * f1_score(y, pr, average='macro'), sess_acc=100 * accuracy_score(ys, ps))
    if nact > 1:
        r['app'] = 100 * f1_score(y // nact, pr // nact, average='macro'); r['act'] = 100 * f1_score(y % nact, pr % nact, average='macro')
        pa = np.array([np.bincount(pr[inv == j] // nact, minlength=NC // nact).argmax() for j in range(nS)])
        r['sess_app_acc'] = 100 * accuracy_score(ys // nact, pa)
    return r


def agg(per):  # list of per-seed dicts -> {metric: [mean, min, max, per-seed]}
    return {k: [float(np.mean([p[k] for p in per])), float(np.min([p[k] for p in per])), float(np.max([p[k] for p in per])), [round(float(p[k]), 3) for p in per]] for k in per[0]}


def main():
    t0 = time.time(); OUT = dict(rules=__doc__.split('Writes results')[0].strip(), run_started_utc=now())
    L = load_lists(); rd = RDAP()
    ipb = json.load(open(os.path.join(ROOT, 'results', 'ip_baseline.json'))); ts = json.load(open(os.path.join(ROOT, 'results', 'threat_stats.json')))
    OUT['attribution'] = {}; OUT['address_observer'] = {}; OUT['accuracy_by_operator'] = {}; OUT['rdap_list_agreement_genai'] = {}
    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); chk = collections.Counter(); A = [server_addr(k, chk) for k in D.index.biflow_key.values]
        lab = (D.y_joint >= 0) if ds == 'genai' else (D.y_app >= 0)
        pre = '' if ds == 'genai' else 'ccma_'
        te0 = None
        for K in (1, 2, 4, 8):
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json'))
                if te0 is None: te0 = idx['test']
                assert np.array_equal(te0, idx['test'])
        test = np.zeros(len(A), bool); test[te0] = True
        # attribution of every distinct address of the evaluated flows
        attr = {}; seen24 = set()
        for i in np.where(lab)[0][np.argsort([int(A[i]) for i in np.where(lab)[0]])]:
            a = A[i]
            if str(a) in attr: continue
            p24 = str(ipaddress.ip_network(f'{a}/24', strict=False))
            force = ds == 'genai' and a.is_global and p24 not in seen24; seen24.add(p24)
            attr[str(a)] = attribute(L, a, rd, force)
        if ds == 'genai':  # RDAP on every /24 vs the published-list verdict
            agree = collections.Counter(); dis = []
            for a, v in attr.items():
                if v['source'] == 'published list' and 'rdap_operator' in v:
                    ok = v['rdap_operator'] == v['operator']; agree[ok] += 1
                    if not ok: dis.append([a, v['service'], v['rdap_operator'], v['rdap'].get('name'), v['rdap'].get('registrant')])
            OUT['rdap_list_agreement_genai'] = dict(n_checked_24s=int(sum(agree.values())), n_agree=int(agree[True]), disagreements=dis)
        # fill RDAP for GenAI /24 members not queried (reuse the /24's answer for the record)
        OUT['attribution'][ds] = dict(n_distinct_addresses=len(attr), flows=describe(D, ds, A, attr, {'evaluated': lab, 'test': test}),
                                      addresses={a: {k: v for k, v in x.items() if k != 'rdap'} | ({'rdap_name': x['rdap'].get('name'), 'rdap_registrant': x['rdap'].get('registrant')} if 'rdap' in x else {})
                                                 for a, x in attr.items()} if ds == 'genai' else None,
                                      services_ccma={} if ds == 'genai' else shares([attr[str(A[i])]['service'] for i in np.where(lab)[0]]))
        if ds == 'ccma':
            OUT['attribution'][ds]['rdap_other_names'] = shares([f"{x['rdap'].get('name')} | {(x['rdap'].get('registrant') or [''])[0]}" for x in attr.values() if x['operator'] in ('other', 'unknown') and 'rdap' in x])
        print(f'=== {ds}: {len(attr)} addresses attributed, RDAP queries so far {rd.n_queries}, failures {len(rd.failures)} ({time.time()-t0:.0f}s)', flush=True)
        for pop in ('evaluated', 'test'):
            for nm, r in OUT['attribution'][ds]['flows'][pop].items():
                print(f'  {pop:9s} {nm:12s} n={r["n_flows"]:5d} ops ' + ', '.join(f'{k} {v[1]:.1f}' for k, v in list(r['operator'].items())[:6]) + ' | fe ' + ', '.join(f'{k} {v[1]:.1f}' for k, v in r['frontend'].items()))
        # (b) address observer per seed + encoder/LightGBM per seed; accuracy by operator (GenAI)
        KK = keys_of(A); ylab = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app; nact = 2 if ds == 'genai' else 1
        OUT['address_observer'][ds] = {}; OUT['accuracy_by_operator'][ds] = {}
        opv = np.array([attr[str(A[i])]['operator'] if lab[i] else '' for i in range(len(A))], dtype=object)
        fev = np.array([attr[str(A[i])]['frontend'] if lab[i] else '' for i in range(len(A))], dtype=object)
        for K in (1, 2, 4, 8):
            per = collections.defaultdict(list); byop = collections.defaultdict(lambda: collections.defaultdict(list))
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json')); tr, te = idx['train'], idx['test']
                preds = {'backoff': predict(KK, tr, te, ylab, NC, 'backoff')[0]}
                for a, rid in (('ours', f'{pre}fs{K}_sslXL_s{s}'), ('lgbm', f'{pre}fs{K}_lgbm_meta64_s{s}')):
                    z = np.load(os.path.join(ROOT, 'results', 'runs', rid, 'test_preds.npz'))
                    assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], ylab[te]); preds[a] = z['pred']
                y = ylab[te]
                for a, pr in preds.items():
                    per[a].append(metrics(y, pr, D.session_id[te], nact))
                    ya, pa = (y // nact, pr // nact)
                    for grp, vec in (('operator', opv), ('frontend', fev)):
                        for g in np.unique(vec[te]):
                            m = vec[te] == g; byop[f'{grp}:{g}'][a].append(100 * float((ya[m] == pa[m]).mean()))
                            byop[f'{grp}:{g}']['n'] = [int(m.sum())]
            res = {a: agg(v) for a, v in per.items()}
            ref = ipb['closed_world'][ds][f'K{K}']['observers']
            for k in res['backoff']:
                assert abs(res['backoff'][k][0] - ref['backoff'][k][0]) < 1e-6, (ds, K, k)
            for a in ('ours', 'lgbm'):
                for k in res[a]:
                    assert abs(res[a][k][0] - ts[f'{ds}_K{K}']['attackers'][a][k][0]) < 1e-6, (ds, K, a, k)
            res['ours_minus_backoff_bootstrap'] = ref['backoff']['ours_minus']; res['ours_minus_lgbm_bootstrap'] = ref['lgbm']['ours_minus']
            res['n_test_sessions'] = ref.get('n_test_sessions', ipb['closed_world'][ds][f'K{K}']['n_test_sessions'])
            OUT['address_observer'][ds][f'K{K}'] = res
            OUT['accuracy_by_operator'][ds][f'K{K}'] = {g: {a: ([float(np.mean(v)), float(np.min(v)), float(np.max(v))] if a != 'n' else v[0]) for a, v in d.items()} for g, d in byop.items()}
            b, o = res['backoff'], res['ours']
            line = f'  {ds} K={K} backoff flow {b["flow"][0]:.1f} [{b["flow"][1]:.1f},{b["flow"][2]:.1f}] sess {b["sess_acc"][0]:.1f}'
            if nact > 1: line += f' app {b["app"][0]:.1f} [{b["app"][1]:.1f},{b["app"][2]:.1f}] sessApp {b["sess_app_acc"][0]:.1f} [{b["sess_app_acc"][1]:.1f},{b["sess_app_acc"][2]:.1f}] | ours app {o["app"][0]:.1f} sessApp {o["sess_app_acc"][0]:.1f}'
            print(line, flush=True)
    OUT['sources'] = SOURCES; OUT['rdap_queries'] = rd.n_queries; OUT['rdap_failures'] = rd.failures
    OUT['rdap_registrations'] = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rd.cache]
    OUT['run_finished_utc'] = now()
    json.dump(OUT, open(os.path.join(ROOT, 'results', 'cdn_attribution.json'), 'w'), indent=1, default=str)
    print(f'written results/cdn_attribution.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
