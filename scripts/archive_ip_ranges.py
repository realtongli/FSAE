"""Archive the operators' public IP-range lists that scripts/cdn_attribution.py read into memory on 2026-09-26.

Rules:
  * Exactly the lists that results/cdn_attribution.json ['sources'] records as read, at exactly the recorded URLs, and
    nothing else: Cloudflare ips-v4 (cdn_attribution.py does not read ips-v6, so it is not archived), Fastly
    public-ip-list, AWS ip-ranges.json, Google goog.json and cloud.json, and the Azure Service Tags file at the URL that
    the retrieval of 2026-09-26 resolved (ServiceTags_Public_20260921.json). The fixed Microsoft block 13.107.0.0/16 is
    not a download, and the IANA RDAP bootstrap and the RDAP answers are not IP-range lists; none of them is fetched
    here.
  * Same HTTP client as cdn_attribution.py (urllib, same User-Agent, 60 s timeout, up to 4 tries). The bytes are saved
    exactly as received; nothing is parsed or rewritten before the SHA-256 is taken.
  * Write-once: the target directory data/ip_ranges/<UTC date of the download> must not already contain a file of the
    same name; the script stops rather than overwrite an archived copy.
  * Recorded per list: URL, final URL after redirects, download time (UTC), size in bytes, SHA-256, the list's own
    timestamp field (Cloudflare and Fastly publish none; AWS createDate and syncToken; Google creationTime and
    syncToken; Azure changeNumber and cloud), the HTTP Last-Modified and ETag headers when sent, the number of IPv4
    prefixes parsed the way cdn_attribution.py parses them, and whether the SHA-256 equals the one recorded on
    2026-09-26. A mismatch is expected for lists that their operators regenerate often, and is reported, not resolved.
Writes data/ip_ranges/<date>/ (the raw files, manifest.json, README.md)."""
import datetime, hashlib, json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UA = {'User-Agent': 'Mozilla/5.0 (research; address-attribution)'}  # as in cdn_attribution.py

REF = json.load(open(os.path.join(ROOT, 'results', 'cdn_attribution.json'), encoding='utf-8'))['sources']
# (name in cdn_attribution.json sources, local file name)
LISTS = [('Cloudflare', 'cloudflare_ips-v4.txt'), ('Fastly', 'fastly_public-ip-list.json'), ('AWS', 'aws_ip-ranges.json'),
         ('Google/own', 'google_goog.json'), ('Google/Cloud', 'google_cloud.json'), ('Azure service tags', None)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch(url, tries=4):
    err = None
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read(), r.geturl(), dict(r.headers.items())
        except Exception as e:
            err = e
            if getattr(e, 'code', None) == 404: break
            time.sleep(2 * (t + 1))
    raise err


def own_timestamp(name, body):
    if name in ('Cloudflare', 'Fastly'): return None, [x.strip() for x in body.decode().split() if x.strip()] if name == 'Cloudflare' else json.loads(body)['addresses']
    j = json.loads(body)
    if name == 'AWS': return dict(createDate=j.get('createDate'), syncToken=j.get('syncToken')), sorted({x['ip_prefix'] for x in j['prefixes']})
    if name.startswith('Google'): return dict(creationTime=j.get('creationTime'), syncToken=j.get('syncToken')), [x['ipv4Prefix'] for x in j['prefixes'] if 'ipv4Prefix' in x]
    tags = {v['name']: v['properties']['addressPrefixes'] for v in j['values']}
    afd = [p for p in tags.get('AzureFrontDoor.Frontend', []) if ':' not in p]; az = [p for p in tags.get('AzureCloud', []) if ':' not in p]
    return dict(changeNumber=j.get('changeNumber'), cloud=j.get('cloud')), afd + az


def main():
    day = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
    out = os.path.join(ROOT, 'data', 'ip_ranges', day); os.makedirs(out, exist_ok=True)
    man = dict(rules=__doc__.split('Writes data')[0].strip(), reference='results/cdn_attribution.json [sources], retrieval of 2026-09-26', lists=[])
    for name, fn in LISTS:
        ref = REF[name]; url = ref['url']
        if fn is None: fn = 'azure_' + url.rsplit('/', 1)[1]
        path = os.path.join(out, fn)
        if os.path.exists(path): raise SystemExit(f'{path} exists; archive is write-once')
        rec = dict(name=name, url=url, file=fn)
        try:
            body, final, hdr = fetch(url); t = now()
        except Exception as e:
            rec.update(status=f'failed: {e}', attempted_utc=now()); man['lists'].append(rec); print(f'{name}: FAILED {e}', flush=True); continue
        with open(path, 'wb') as f: f.write(body)
        sha = hashlib.sha256(body).hexdigest(); stamp, pfx = own_timestamp(name, body)
        rec.update(status='ok', final_url=final, downloaded_utc=t, bytes=len(body), sha256=sha, list_timestamp=stamp,
                   http_last_modified=hdr.get('Last-Modified'), http_etag=hdr.get('ETag'), n_ipv4_prefixes=len(pfx),
                   ref_accessed_utc=ref.get('accessed_utc'), ref_sha256=ref.get('sha256'), ref_bytes=ref.get('bytes'),
                   ref_list_timestamp=ref.get('list_timestamp'), ref_n_ipv4_prefixes=ref.get('n_ipv4_prefixes'),
                   sha256_equals_2026_09_26=(sha == ref.get('sha256')))
        man['lists'].append(rec)
        print(f'{name:18s} {len(body):9d} B  sha {sha[:16]}  equal-to-09-26 {rec["sha256_equals_2026_09_26"]}  stamp {stamp}  n4 {len(pfx)} (09-26: {ref.get("n_ipv4_prefixes")})', flush=True)
        time.sleep(1)
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8'), indent=1)

    L = ['# Operators\' public IP-range lists, archived ' + day, '',
         'Raw copies, byte-for-byte as served, of the published address lists that `scripts/cdn_attribution.py` read into memory '
         'on 2026-09-26 (it records URL, access time and SHA-256 of each in `results/cdn_attribution.json` [`sources`] '
         'but does not keep the files). Downloaded by `scripts/archive_ip_ranges.py`, which states the rules; `manifest.json` '
         'holds the same information in machine-readable form. The Microsoft block 13.107.0.0/16 is fixed in the code and '
         'needs no file; Cloudflare `ips-v6` is not read and is not archived.', '',
         'The lists are regenerated by their operators, so a copy taken a day later can differ from the one used on 2026-09-26; '
         'the last column says whether the bytes are identical.', '',
         '| List | URL | Downloaded (UTC) | Size (bytes) | SHA-256 | List\'s own timestamp | IPv4 prefixes | SHA-256 = 2026-09-26 retrieval |',
         '|---|---|---|---|---|---|---|---|']
    for r in man['lists']:
        if r['status'] != 'ok':
            L.append(f"| {r['name']} (`{r['file']}`) | {r['url']} | {r['attempted_utc']} | - | - | - | - | download failed: {r['status']} |"); continue
        st = 'none published' if r['list_timestamp'] is None else ', '.join(f'{k}={v}' for k, v in r['list_timestamp'].items())
        rst = 'none published' if r['ref_list_timestamp'] is None else ', '.join(f'{k}={v}' for k, v in r['ref_list_timestamp'].items())
        eq = 'yes' if r['sha256_equals_2026_09_26'] else f"no (2026-09-26: `{r['ref_sha256']}`, {r['ref_bytes']} B, {rst}, {r['ref_n_ipv4_prefixes']} prefixes)"
        L.append(f"| {r['name']} (`{r['file']}`) | {r['url']} | {r['downloaded_utc']} | {r['bytes']} | `{r['sha256']}` | {st} | {r['n_ipv4_prefixes']} | {eq} |")
    L += ['', 'HTTP headers sent with each file (not part of the list itself):', '']
    for r in man['lists']:
        if r['status'] == 'ok': L.append(f"- {r['name']}: Last-Modified `{r['http_last_modified']}`, ETag `{r['http_etag']}`, final URL {r['final_url']}")
    L += ['', 'IPv4 prefixes are counted as `cdn_attribution.py` parses them: every line of Cloudflare ips-v4; Fastly `addresses`; '
          'distinct AWS `ip_prefix` (all services); Google `ipv4Prefix`; Azure service tags AzureFrontDoor.Frontend plus AzureCloud IPv4 prefixes.', '']
    open(os.path.join(out, 'README.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('written', out)


if __name__ == '__main__':
    main()
