"""Operator attribution of the evaluated GenAI and CCMA server addresses recomputed from the ARCHIVED copies of the
operators' public IP-range lists (data/ip_ranges/2026-09-27/, written by scripts/archive_ip_ranges.py) instead of the
copies that scripts/cdn_attribution.py read over HTTPS into memory on 2026-09-26, and compared with
results/cdn_attribution.json. Descriptive; nothing is fitted or selected.

Rules:

Logic. scripts/cdn_attribution.py cannot read local files (load_lists() fetches URLs), so its functions are imported
unchanged (server_addr, nets, match, attribute, rdap_operator, shares, describe, FRONTEND; its per-operator accuracy
loop is copied line for line) and only two inputs are replaced:
  (1) the lists: read from the archived files and parsed exactly as load_lists() parses the downloaded bytes
      (Cloudflare ips-v4 lines; Fastly 'addresses'; AWS distinct ip_prefix, CLOUDFRONT vs all; Google cloud.json and
      goog.json ipv4Prefix; Azure service tags AzureFrontDoor.Frontend and AzureCloud IPv4; 13.107.0.0/16 fixed);
  (2) RDAP: no new network query is made. The 220 registrations of the retrieval of 2026-09-26
      (results/cdn_attribution.json ['rdap_registrations'], in their stored order) are replayed: a forced query
      (GenAI, lowest observed address of each /24) returns the stored registration whose query is that address; a reuse
      lookup returns the first stored registration whose range contains the address, as RDAP.lookup(reuse=True) does.
      An address that no archived list matches and that no stored registration contains would need a new RDAP
      query; it is attributed 'unknown', listed, and not queried.
Populations as in cdn_attribution.py: 'evaluated' (GenAI y_joint >= 0; CCMA y_app >= 0) and 'test' (test flows of the
few-shot splits, asserted identical over K and seeds).
Checks and comparisons (all are reported; none alters the attribution):
  a. Replay check: for every GenAI address, the RDAP name and RDAP operator recorded on 2026-09-26 must equal the
     replayed ones whenever the 2026-09-26 record has them.
  b. Content check for the lists whose SHA-256 differs from 2026-09-26: the list's own timestamp fields (Google
     syncToken and creationTime; AWS syncToken and createDate) are replaced in the archived bytes by the values recorded
     on 2026-09-26 and the SHA-256 is recomputed; equality proves the rest of the file (every prefix) byte-identical.
  c. GenAI per address (230 addresses; results/cdn_attribution.json stores each): operator, service, front-end class,
     matched prefix and source, 2026-09-26 vs archived; every address whose operator changes is listed, and separately
     every address whose service, front-end class or matched prefix changes.
  d. Per assistant (GenAI) and per app (CCMA), evaluated and test flows: operator, service and front-end shares,
     2026-09-26 vs archived, plus the '_all' rows. results/cdn_attribution.json stores no CCMA per-address attributions,
     so for CCMA the comparison is at the level of per-app flow counts per operator, service and front-end class, the
     'services_ccma' and 'rdap_other_names' counts, and the CCMA addresses that were RDAP queries of the CCMA phase on
     2026-09-26 (no list matched them then): any of them that an archived list matches has changed. CCMA-phase queries
     are the stored registrations whose query address is not a GenAI evaluated address (the GenAI phase queries every
     GenAI /24 whatever the lists say, and a CCMA address equal to a GenAI one reuses that answer, so it makes no query).
  e. Per-operator / per-front-end-class accuracy on the GenAI and CCMA test flows (encoder fs{K}_sslXL, LightGBM
     fs{K}_lgbm_meta64, address back-off), K = 1, 2, 4, 8, seeds 0-4, recomputed with the archived attribution by the
     same code as cdn_attribution.py and compared with its 'accuracy_by_operator' (max absolute difference).
Writes results/cdn_attribution_archived.json (a new file; results/cdn_attribution.json is not modified)."""
import collections, datetime, hashlib, ipaddress, json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import cdn_attribution as CA  # noqa: E402  (its main() is not run)
from data_mfr import GenAIData  # noqa: E402
from ip_baseline import server_addr, keys_of, predict  # noqa: E402
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ARCH = os.path.join(ROOT, 'data', 'ip_ranges', '2026-09-27')
REF = json.load(open(os.path.join(ROOT, 'results', 'cdn_attribution.json'), encoding='utf-8'))
MAN = json.load(open(os.path.join(ARCH, 'manifest.json'), encoding='utf-8'))
FILES = {r['name']: os.path.join(ARCH, r['file']) for r in MAN['lists'] if r['status'] == 'ok'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load_archived():
    """Same parsing as cdn_attribution.load_lists(), bytes read from the archive instead of HTTPS."""
    L = {}; src = {}
    rd = lambda n: open(FILES[n], 'rb').read()
    b = rd('Cloudflare'); p = [x.strip() for x in b.decode().split() if x.strip()]; L['Cloudflare'] = CA.nets(p); src['Cloudflare'] = len(p)
    b = rd('Fastly'); p = json.loads(b)['addresses']; L['Fastly'] = CA.nets(p); src['Fastly'] = len(p)
    b = rd('AWS'); j = json.loads(b)
    cf = sorted({x['ip_prefix'] for x in j['prefixes'] if x['service'] == 'CLOUDFRONT'}); al = sorted({x['ip_prefix'] for x in j['prefixes']})
    L['Amazon/CloudFront'] = CA.nets(cf); L['Amazon/AWS'] = CA.nets(al); src['AWS'] = len(al); src['AWS CLOUDFRONT'] = len(cf)
    for nm in ('Google/Cloud', 'Google/own'):
        b = rd(nm); j = json.loads(b); p = [x['ipv4Prefix'] for x in j['prefixes'] if 'ipv4Prefix' in x]; L[nm] = CA.nets(p); src[nm] = len(p)
    L['Microsoft/edge-13.107'] = CA.nets(['13.107.0.0/16'])
    b = rd('Azure service tags'); j = json.loads(b); tags = {v['name']: v['properties']['addressPrefixes'] for v in j['values']}
    afd = [p for p in tags.get('AzureFrontDoor.Frontend', []) if ':' not in p]; az = [p for p in tags.get('AzureCloud', []) if ':' not in p]
    L['Microsoft/AzureFrontDoor'] = CA.nets(afd); L['Microsoft/Azure'] = CA.nets(az); src['Azure AFD'] = len(afd); src['Azure AzureCloud'] = len(az)
    return L, src


def content_check():
    """Rule b: substitute the 2026-09-26 timestamp fields into the archived bytes and compare SHA-256."""
    out = {}
    for r in MAN['lists']:
        if r['status'] != 'ok' or r['sha256_equals_2026_09_26']: continue
        body = open(FILES[r['name']], 'rb').read(); new, old = r['list_timestamp'] or {}, r['ref_list_timestamp'] or {}
        sub = body; done = []
        for k in ('syncToken', 'creationTime', 'createDate'):
            if k in new and k in old:
                a = f'"{k}": "{new[k]}"'.encode(); bb = f'"{k}": "{old[k]}"'.encode()
                n = sub.count(a); sub = sub.replace(a, bb); done.append([k, n])
        sha = hashlib.sha256(sub).hexdigest()
        out[r['name']] = dict(substituted=done, bytes_after=len(sub), ref_bytes=r['ref_bytes'], sha256_after=sha, ref_sha256=r['ref_sha256'],
                              identical_apart_from_timestamps=(sha == r['ref_sha256']))
        print(f'  content check {r["name"]:14s}: timestamps {done} -> {len(sub)} B (09-26 {r["ref_bytes"]} B), identical apart from timestamps: {sha == r["ref_sha256"]}', flush=True)
    return out


class ReplayRDAP:
    def __init__(self, regs):
        self.cache = []
        for r in regs:
            try: lo, hi = int(ipaddress.ip_address(r['start'])), int(ipaddress.ip_address(r['end']))
            except Exception: lo = hi = int(ipaddress.ip_address(r['query']))
            self.cache.append(dict(r, _lo=lo, _hi=hi))
        self.missing = []; self.n_queries = 0; self.failures = []

    def lookup(self, a, reuse=True):
        if not reuse:
            for r in self.cache:
                if r['query'] == str(a): return r
            self.missing.append([str(a), 'forced query not stored']); return None
        for r in self.cache:
            if r['_lo'] <= int(a) <= r['_hi']: return r
        self.missing.append([str(a), 'no stored registration contains it (would need a new RDAP query)']); return None


def cmp_shares(old, new):
    keys = list(dict.fromkeys(list(old) + list(new)))
    return {k: dict(old=old.get(k, [0, 0.0]), archived=new.get(k, [0, 0.0])) for k in keys}


def main():
    t0 = time.time(); OUT = dict(rules=__doc__.split('Writes results')[0].strip(), run_started_utc=now(), archive=os.path.relpath(ARCH, ROOT).replace('\\', '/'))
    OUT['archive_manifest'] = [{k: r.get(k) for k in ('name', 'file', 'url', 'downloaded_utc', 'bytes', 'sha256', 'list_timestamp', 'sha256_equals_2026_09_26', 'ref_sha256')} for r in MAN['lists']]
    OUT['content_check'] = content_check()
    L, src = load_archived(); OUT['archived_ipv4_prefix_counts'] = src
    OUT['ref_ipv4_prefix_counts'] = {k: {kk: v.get(kk) for kk in ('n_ipv4_prefixes', 'n_cloudfront_ipv4_prefixes', 'n_afd_frontend', 'n_azurecloud') if kk in v} for k, v in REF['sources'].items()}
    rd = ReplayRDAP(REF['rdap_registrations'])
    OUT['attribution'] = {}; OUT['comparison'] = {}; OUT['accuracy_by_operator'] = {}; OUT['accuracy_by_operator_max_abs_diff'] = {}
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
        attr = {}; seen24 = set(); n_miss0 = len(rd.missing)
        for i in np.where(lab)[0][np.argsort([int(A[i]) for i in np.where(lab)[0]])]:  # same order and forcing as cdn_attribution.py
            a = A[i]
            if str(a) in attr: continue
            p24 = str(ipaddress.ip_network(f'{a}/24', strict=False))
            force = ds == 'genai' and a.is_global and p24 not in seen24; seen24.add(p24)
            attr[str(a)] = CA.attribute(L, a, rd, force)
        flows = CA.describe(D, ds, A, attr, {'evaluated': lab, 'test': test})
        ref = REF['attribution'][ds]
        addrs = {a: {k: v for k, v in x.items() if k != 'rdap'} | ({'rdap_name': x['rdap'].get('name'), 'rdap_registrant': x['rdap'].get('registrant')} if 'rdap' in x else {}) for a, x in attr.items()}
        OUT['attribution'][ds] = dict(n_distinct_addresses=len(attr), flows=flows, addresses=addrs,
                                      services_ccma={} if ds == 'genai' else CA.shares([attr[str(A[i])]['service'] for i in np.where(lab)[0]]),
                                      rdap_missing=rd.missing[n_miss0:])
        if ds == 'ccma':
            OUT['attribution'][ds]['rdap_other_names'] = CA.shares([f"{x['rdap'].get('name')} | {(x['rdap'].get('registrant') or [''])[0]}" for x in attr.values() if x['operator'] in ('other', 'unknown') and 'rdap' in x])
        C = dict(n_distinct_addresses=[ref['n_distinct_addresses'], len(attr)])
        # c. per-address (GenAI)
        if ds == 'genai':
            assert set(ref['addresses']) == set(attr), 'address sets differ'
            rep = dict(checked_name=0, name_mismatch=[], checked_operator=0, operator_mismatch=[])
            ch = {f: [] for f in ('operator', 'service', 'frontend', 'matched_prefix', 'source')}
            for a in sorted(attr, key=lambda s: int(ipaddress.ip_address(s))):
                o, n = ref['addresses'][a], addrs[a]
                if 'rdap_name' in o and 'rdap_name' in n:
                    rep['checked_name'] += 1
                    if o['rdap_name'] != n['rdap_name'] or o.get('rdap_registrant') != n.get('rdap_registrant'): rep['name_mismatch'].append([a, o['rdap_name'], n['rdap_name']])
                elif 'rdap_name' in o: rep['name_mismatch'].append([a, o['rdap_name'], 'not replayed'])
                if 'rdap_operator' in o and 'rdap_operator' in n:
                    rep['checked_operator'] += 1
                    if o['rdap_operator'] != n['rdap_operator']: rep['operator_mismatch'].append([a, o['rdap_operator'], n['rdap_operator']])
                for f in ch:
                    if o.get(f) != n.get(f):
                        nf = int(sum(1 for i in np.where(lab)[0] if str(A[i]) == a)); nt = int(sum(1 for i in np.where(lab & test)[0] if str(A[i]) == a))
                        ch[f].append(dict(address=a, old={k: o.get(k) for k in ('operator', 'service', 'frontend', 'matched_prefix', 'source')},
                                          archived={k: n.get(k) for k in ('operator', 'service', 'frontend', 'matched_prefix', 'source')}, n_evaluated_flows=nf, n_test_flows=nt))
            C['rdap_replay_check'] = rep; C['address_changes'] = ch
            agree = collections.Counter(); dis = []
            for a, v in attr.items():
                if v['source'] == 'published list' and 'rdap_operator' in v:
                    ok = v['rdap_operator'] == v['operator']; agree[ok] += 1
                    if not ok: dis.append([a, v['service'], v['rdap_operator'], v['rdap'].get('name')])
            C['rdap_list_agreement'] = dict(old=REF['rdap_list_agreement_genai'], archived=dict(n_checked_24s=int(sum(agree.values())), n_agree=int(agree[True]), disagreements=dis))
            print(f'=== genai: replay check names {rep["checked_name"]} checked, {len(rep["name_mismatch"])} mismatches; operators {rep["checked_operator"]} checked, {len(rep["operator_mismatch"])} mismatches', flush=True)
            print('    per-address changes: ' + ', '.join(f'{f} {len(v)}' for f, v in ch.items()), flush=True)
            for x in ch['operator']: print('    OPERATOR CHANGED', x, flush=True)
        else:
            qset = {r['query'] for r in REF['rdap_registrations']} - set(REF['attribution']['genai']['addresses'])
            C['n_ccma_phase_rdap_queries'] = len(qset)
            was_rdap_now_list = [dict(address=a, archived={k: attr[a].get(k) for k in ('operator', 'service', 'matched_prefix', 'source')}) for a in attr if a in qset and attr[a]['source'] == 'published list']
            C['ccma_rdap_queried_now_list_matched'] = was_rdap_now_list
            C['services_ccma'] = cmp_shares(ref['services_ccma'], OUT['attribution'][ds]['services_ccma'])
            C['rdap_other_names'] = cmp_shares(ref['rdap_other_names'], OUT['attribution'][ds]['rdap_other_names'])
            C['n_ccma_rdap_query_addresses_in_population'] = int(sum(a in qset for a in attr))
            print(f'=== ccma: {len(attr)} addresses; RDAP-queried on 09-26 and matched by an archived list: {len(was_rdap_now_list)}; missing RDAP {len(rd.missing) - n_miss0}', flush=True)
        # d. shares per assistant / app
        C['flows'] = {}; nd = 0
        for pop in ('evaluated', 'test'):
            C['flows'][pop] = {}
            for nm, r in flows[pop].items():
                o = ref['flows'][pop][nm]; e = dict(n_flows=[o['n_flows'], r['n_flows']])
                for f in ('operator', 'service', 'frontend'):
                    if f in r:
                        e[f] = cmp_shares(o[f], r[f]); diff = {k: v for k, v in e[f].items() if v['old'] != v['archived']}; e[f + '_differs'] = diff; nd += len(diff)
                C['flows'][pop][nm] = e
                print(f'  {pop:9s} {nm:12s} n={r["n_flows"]:5d} ' + ', '.join(f'{k} {v["old"][1]:.2f}->{v["archived"][1]:.2f}' for k, v in list(e['operator'].items())[:6])
                      + (f'  DIFFERS {e["operator_differs"] | e.get("service_differs", {}) | e["frontend_differs"]}' if (e['operator_differs'] or e.get('service_differs') or e['frontend_differs']) else ''), flush=True)
        C['n_share_cells_differing'] = nd
        OUT['comparison'][ds] = C
        # e. per-operator accuracy with the archived attribution (same code as cdn_attribution.py)
        KK = keys_of(A); ylab = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app; nact = 2 if ds == 'genai' else 1
        opv = np.array([attr[str(A[i])]['operator'] if lab[i] else '' for i in range(len(A))], dtype=object)
        fev = np.array([attr[str(A[i])]['frontend'] if lab[i] else '' for i in range(len(A))], dtype=object)
        OUT['accuracy_by_operator'][ds] = {}; mx = 0.0; notes = []
        for K in (1, 2, 4, 8):
            byop = collections.defaultdict(lambda: collections.defaultdict(list))
            for s in range(5):
                idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json')); tr, te = idx['train'], idx['test']
                preds = {'backoff': predict(KK, tr, te, ylab, NC, 'backoff')[0]}
                for a, rid in (('ours', f'{pre}fs{K}_sslXL_s{s}'), ('lgbm', f'{pre}fs{K}_lgbm_meta64_s{s}')):
                    z = np.load(os.path.join(ROOT, 'results', 'runs', rid, 'test_preds.npz'))
                    assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], ylab[te]); preds[a] = z['pred']
                y = ylab[te]
                for a, pr in preds.items():
                    ya, pa = (y // nact, pr // nact)
                    for grp, vec in (('operator', opv), ('frontend', fev)):
                        for g in np.unique(vec[te]):
                            m = vec[te] == g; byop[f'{grp}:{g}'][a].append(100 * float((ya[m] == pa[m]).mean()))
                            byop[f'{grp}:{g}']['n'] = [int(m.sum())]
            res = {g: {a: ([float(np.mean(v)), float(np.min(v)), float(np.max(v))] if a != 'n' else v[0]) for a, v in d.items()} for g, d in byop.items()}
            OUT['accuracy_by_operator'][ds][f'K{K}'] = res
            old = REF['accuracy_by_operator'][ds][f'K{K}']
            for g in set(old) | set(res):
                if g not in old or g not in res: notes.append(f'K{K} group {g} only in ' + ('archived' if g in res else '2026-09-26')); continue
                for a in set(old[g]) | set(res[g]):
                    if a == 'n':
                        if old[g].get('n') != res[g].get('n'): notes.append(f'K{K} {g} n {old[g].get("n")} -> {res[g].get("n")}')
                        continue
                    mx = max(mx, max(abs(x - y) for x, y in zip(old[g][a], res[g][a])))
            fe = res.get('frontend:shared_cdn', {})
            if fe: print(f'  {ds} K={K} shared_cdn n={fe["n"]} ours {fe["ours"][0]:.2f} backoff {fe["backoff"][0]:.2f} (09-26: ours {old["frontend:shared_cdn"]["ours"][0]:.2f} backoff {old["frontend:shared_cdn"]["backoff"][0]:.2f})', flush=True)
        OUT['accuracy_by_operator_max_abs_diff'][ds] = dict(max_abs_diff_pp=mx, group_notes=notes)
        print(f'  {ds}: accuracy_by_operator max |archived - 09-26| = {mx:.3g} pp; notes {notes}', flush=True)
    OUT['rdap_missing_total'] = rd.missing; OUT['run_finished_utc'] = now()
    path = os.path.join(ROOT, 'results', 'cdn_attribution_archived.json')
    assert not os.path.exists(path), 'output exists'
    json.dump(OUT, open(path, 'w', encoding='utf-8'), indent=1, default=str)
    print(f'written results/cdn_attribution_archived.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
