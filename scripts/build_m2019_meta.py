"""Metadata-only extraction of MIRAGE-2019 (unlabelled pre-training corpus; labels kept only for bookkeeping).
Streams data/MIRAGE-2019_v2.tar.gz (nothing extracted to disk). MIRAGE-2019 packet lists are truncated at 32 packets and
carry no per-packet IP length, so IP bytes are reconstructed as L4 payload + 52 (TCP) / + 28 (UDP), the constant offsets
observed in MIRAGE-GenAI-2025 and CCMA-2022 (p10 = p90). Output data/derived/m2019_meta.npz: meta [N,64,4] (dir, IP bytes,
payload bytes, iat), meta_len, session_id (one JSON = one session), device, label (package)."""
import io, json, os, sys, tarfile, time, collections
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TAR = os.path.join(ROOT, 'data', 'MIRAGE-2019_v2.tar.gz'); OUT = os.path.join(ROOT, 'data', 'derived', 'm2019_meta.npz'); META_LEN = 64
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

metas, lens, sids, devs, labels, files = [], [], [], [], [], []
stats = collections.Counter(); t0 = time.time()
with tarfile.open(TAR, 'r:gz') as tf:
    for m in tf:
        if not m.isfile() or not m.name.endswith('.json'): continue
        d = json.load(tf.extractfile(m)); sid = len(files); dev = m.name.split('/')[1]; files.append(m.name)
        for key, b in d.items():
            pdt = b['packet_data']; lp = pdt['L4_payload_bytes']; dirs = pdt['packet_dir']; iat = pdt['iat']
            stats['total'] += 1
            if not any(l > 0 for l in lp): stats['no_payload'] += 1; continue
            hdr = 52.0 if key.split(',')[-1] == '6' else 28.0
            n = min(len(lp), META_LEN); mm = np.zeros((META_LEN, 4), np.float32)
            mm[:n, 0] = np.where(np.array(dirs[:n]) == 0, 1.0, -1.0); mm[:n, 2] = lp[:n]; mm[:n, 1] = mm[:n, 2] + hdr; mm[:n, 3] = iat[:n]
            metas.append(mm); lens.append(n); sids.append(sid); devs.append(dev); labels.append(b['flow_metadata']['BF_label']); stats['kept'] += 1
        if sid % 100 == 0: print(f'{sid} files, kept {stats["kept"]} / {stats["total"]} flows, {time.time() - t0:.0f}s', flush=True)
np.savez_compressed(OUT, meta=np.stack(metas), meta_len=np.array(lens, np.int32), session_id=np.array(sids, np.int32), device=np.array(devs), label=np.array(labels), files=np.array(files))
print('saved', OUT, dict(stats), 'sessions', len(files), 'devices', collections.Counter(devs), f'{time.time() - t0:.0f}s')
print('label counts (top 45):', collections.Counter(labels).most_common(45))
