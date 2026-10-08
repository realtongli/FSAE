"""Extract the UNMONITORED (background) biflows of both campaigns for open-world evaluation.

In each capture session the phone also produces traffic of apps other than the targeted one (Google Play services,
netd, Chrome, the store, etc.). build_dataset.py discards them by package name; this script keeps them, because for an
on-path observer they are exactly the traffic that is not in its monitored set. Session ids match the existing
<prefix>_sessions.csv, so the same session-level splits apply and no session leaks between train and test.

Outputs data/derived/<prefix>_background.npz: meta [N,64,4] (dir, IP bytes, payload bytes, iat), meta_len,
pkt_bytes [N,5,320] (for byte-level attackers), pkt_sni, session_id, pkg (the netstat package of the flow).
"""
import io, json, os, sys, time, glob, zipfile, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from build_dataset import find_sni_span, TARGET_PKG, K_PKT, PKT_BYTES, META_LEN
from build_dataset_ccma import APPS as CCMA_APPS

OUT = os.path.join(ROOT, 'data', 'derived')
MIN_PAY_PKTS = 1  # at least one payload-bearing packet, the same inclusion rule as the monitored flows
ANY_TARGET = set().union(*TARGET_PKG.values())  # every monitored package of the campaign


def extract(b, key):
    """-> (pkt_bytes [5,320], pkt_sni [5,2], meta [64,4], meta_len) or None if the flow carries no payload."""
    md, pdt = b['flow_metadata'], b['packet_data']
    lp = pdt['L4_payload_bytes']; raw = pdt['L4_raw_payload']; dirs = pdt['packet_dir']
    idx_pay = [i for i, l in enumerate(lp) if l > 0 and i < len(raw) and len(raw[i]) > 0]
    if len(idx_pay) < MIN_PAY_PKTS: return None
    pk = np.zeros((K_PKT, PKT_BYTES), np.uint8); psn = np.full((K_PKT, 2), -1, np.int32)
    for j, i in enumerate(idx_pay[:K_PKT]):
        b_ = bytes(raw[i]); n = min(len(b_), PKT_BYTES); pk[j, :n] = np.frombuffer(b_[:n], np.uint8)
        sp = find_sni_span(b_)
        if sp is not None and sp[0] < PKT_BYTES: psn[j] = (sp[0], min(sp[1], PKT_BYTES))
    m = np.zeros((META_LEN, 4), np.float32); nm = min(len(lp), META_LEN)
    m[:nm, 0] = np.where(np.array(dirs[:nm]) == 0, 1.0, -1.0)
    ipb = pdt.get('IP_packet_bytes')
    if ipb is None:  # MIRAGE-2019-style records without per-packet IP length
        hdr = 52.0 if key.split(',')[-1] == '6' else 28.0; m[:nm, 2] = lp[:nm]; m[:nm, 1] = m[:nm, 2] + hdr
    else:
        m[:nm, 1] = ipb[:nm]; m[:nm, 2] = lp[:nm]
    m[:nm, 3] = pdt['iat'][:nm]
    return pk, psn, m, nm


def build_genai():
    RAW = os.path.join(ROOT, 'data', 'mirage2025genai')
    ses = pd.read_csv(os.path.join(OUT, 'genai_sessions.csv')); file2sid = dict(zip(ses.file, ses.session_id))
    A = collections.defaultdict(list); pkgs = []; stats = collections.Counter(); t0 = time.time()
    for f in sorted(glob.glob(os.path.join(RAW, '*', '*', '*.json'))):
        rel = os.path.relpath(f, RAW).replace('\\', '/')
        if rel not in file2sid: continue  # Telegram/WhatsApp controlled sessions
        sid = file2sid[rel]
        d = json.load(open(f))
        for key, b in d.items():
            pkg = b['flow_metadata']['BF_label']
            if pkg in ANY_TARGET: continue  # monitored: exclude every monitored package in every session, not only the
            # session's own target, so that e.g. Google-app flows inside a ChatGPT session do not enter the unmonitored set
            r = extract(b, key)
            if r is None: stats['no_payload'] += 1; continue
            A['pkt_bytes'].append(r[0]); A['pkt_sni'].append(r[1]); A['meta'].append(r[2]); A['meta_len'].append(r[3])
            A['session_id'].append(sid); pkgs.append(pkg); stats['kept'] += 1
        if sid % 40 == 0: print(f'genai session {sid} kept {stats["kept"]} {time.time()-t0:.0f}s', flush=True)
    save('genai', A, pkgs, stats, t0)


def build_ccma():
    ZIP = os.path.join(ROOT, 'data', 'MIRAGE-COVID-CCMA-2022.zip')
    ses = pd.read_csv(os.path.join(OUT, 'ccma_sessions.csv')); ses = ses[ses.n_kept > 0]; file2sid = dict(zip(ses.file, ses.session_id))
    targets = set(CCMA_APPS.values())
    A = collections.defaultdict(list); pkgs = []; stats = collections.Counter(); t0 = time.time(); z = zipfile.ZipFile(ZIP)
    for app in CCMA_APPS:
        inner = zipfile.ZipFile(io.BytesIO(z.read(f'MIRAGE-COVID-CCMA-2022/Raw_JSON/{app}.zip')))
        for fn in sorted(n for n in inner.namelist() if n.endswith('.json')):
            if fn not in file2sid: continue
            sid = int(file2sid[fn]); d = json.loads(inner.read(fn))
            for key, b in d.items():
                pkg = b['flow_metadata']['BF_label']
                if pkg in targets: continue  # monitored (any of the nine apps)
                r = extract(b, key)
                if r is None: stats['no_payload'] += 1; continue
                A['pkt_bytes'].append(r[0]); A['pkt_sni'].append(r[1]); A['meta'].append(r[2]); A['meta_len'].append(r[3])
                A['session_id'].append(sid); pkgs.append(pkg); stats['kept'] += 1
        print(f'ccma {app} kept {stats["kept"]} {time.time()-t0:.0f}s', flush=True)
        del inner
    save('ccma', A, pkgs, stats, t0)


def save(prefix, A, pkgs, stats, t0):
    out = os.path.join(OUT, f'{prefix}_background.npz')
    np.savez_compressed(out, pkt_bytes=np.stack(A['pkt_bytes']), pkt_sni=np.stack(A['pkt_sni']), meta=np.stack(A['meta']),
                        meta_len=np.array(A['meta_len'], np.int32), session_id=np.array(A['session_id'], np.int64), pkg=np.array(pkgs))
    c = collections.Counter(pkgs)
    print(f'{prefix}: saved {out} kept={stats["kept"]} no_payload={stats["no_payload"]} packages={len(c)} sessions={len(set(A["session_id"]))} {time.time()-t0:.0f}s')
    print('top packages:', c.most_common(12))


if __name__ == '__main__':
    which = sys.argv[1:] or ['genai', 'ccma']
    if 'genai' in which: build_genai()
    if 'ccma' in which: build_ccma()
