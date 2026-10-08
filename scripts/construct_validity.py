"""Construct validity of the GenAI labels and decisions (training-free: reads saved test predictions; no model is trained).

Observers (session-scarce attackers): the pre-trained encoder (runs fs{K}_sslXL_s{s}), LightGBM on the 64-packet
metadata (fs{K}_lgbm_meta64_s{s}) and the input-space 1-NN on the first 10 packets (no prediction file; recomputed as in
scripts/threat_stats.py). K in {1,2,4,8}, seeds 0-4; the 48 test sessions (8 per class) are identical across seeds.
K=4 is the headline; every K is written.

Server name of a connection (ground truth only; no observer reads it), from the raw MIRAGE-GenAI-2025 payloads
(data/mirage2025genai, read-only): the SNI of the TLS ClientHello over TCP (handshake records reassembled over the first
upstream payloads), the SNI of the ClientHello carried in the QUIC client Initial packets (Initial protection keys derive
from the public Destination Connection ID, RFC 9001 sec. 5.2; AES-128 and HKDF below are checked against FIPS-197 and
RFC 9001 appendix A at start-up), or the Host header of a plaintext HTTP request. A connection whose handshake precedes
the capture (first upstream payload is TLS application data or a QUIC short header, or no upstream payload in the
first 40 packets) has no name. The coverage of the plaintext parser of scripts/leak_source.py (stored 1024-byte window)
is reported next to it, with the agreement on the connections both name.

Rules:
(1) Category of a connection of assistant A (first match wins, lower-cased server name):
    image    : ^files.oaiusercontent.com$ | ^lh3.googleusercontent.com$ | .mm.bing.net$   (image delivery)
    telemetry: analytics, crash reporting, experimentation, attribution, ads; first or third party (TEL_RE below)
    assistant: A's own endpoints. ChatGPT: openai.com, chatgpt.com, oaistatic, oaiusercontent; Copilot: copilot,
               sydney, bing.com, bingviz; Gemini: gemini, bard, makersuite, proactivebackend, generativelanguage,
               assistant
    vendor   : the vendor's generic, non-assistant services. Gemini: Google (google.*, googleapis, gstatic,
               googleusercontent, youtube, ytimg, ggpht, gvt1/2, googlevideo, android.com, 1e100.net); Copilot:
               Microsoft (microsoft.com, microsoftonline.com, office.com/.net, live.com, sfx.ms, msn.com, windows.net,
               skype.com, microsoftapp.net, azure); ChatGPT: none
    other    : any other named server (third-party content and CDNs, other vendors' platform services)
    none     : no server name.
    Per category: share of A's test connections; recall of the assistant decision (6-class prediction // 2) per
    observer (mean, std over seeds; K=4 also a 95% CI from 2000 resamples of A's 16 test sessions, seed-averaged); and
    the share of A's correctly assigned connections that fall in the category. Per host (digits folded to '#'): the
    same for every host with >= 2% of A's test connections. The 3x3 assistant confusion matrix (mean counts over seeds,
    row-normalised) and per-assistant F1. Session-level assistant vote (majority over the session's connections, ties
    to the lower index) on (a) all connections, (b) all but the 'vendor' category, (c) only 'assistant' + 'image'
    connections; a session left without connections counts as an error (accuracy on the remaining sessions is also
    given). Category shares over all labelled connections (all splits) are given for reference. Background check: for each Gemini-class host
    with >= 2% share, the share of the generic sessions of each assistant (all splits) in which the Google app package,
    and any package, contacts it.
(2) Session-level modality: score = share of the session's connections whose 6-class prediction is an
    image-generation class; AUROC over the 48 test sessions (pooled, and within each assistant, 8 vs 8), mean/std over
    seeds; 95% CI of the pooled AUROC from 2000 session resamples (seed-averaged). Majority vote (share > 0.5, ties to
    text) accuracy and flow-level modality macro-F1 for reference. Observer-free reference: the share of the session's
    connections to image-delivery hosts (ground-truth names).
(3) Label noise: test connections split into image-delivery-host connections vs all others; in each subset the share
    predicted image generation among connections of image-generation sessions (recall) and of text sessions
    (false-positive rate), the binary F1 of the image class, and the modality macro-F1 over the classes present in the
    subset; per assistant for the image-host subset; the share of image sessions' connections that are image fetches.
Writes results/construct_validity.json."""
import collections, glob, hashlib, hmac, json, os, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
from trivial_baselines import feats_knn

RAW = os.path.join(ROOT, 'data', 'mirage2025genai')
APPS = ['ChatGPT', 'Copilot', 'Gemini']; GOOGLE_APP = 'com.google.android.googlequicksearchbox'
RUNS = {'ours': 'fs{K}_sslXL_s{s}', 'lgbm': 'fs{K}_lgbm_meta64_s{s}'}; OBS = ['ours', 'lgbm', 'knn1']
KS, SEEDS, B, HEAD_K, HOST_MIN = (1, 2, 4, 8), range(5), 2000, 4, 0.02
CATS = ['assistant', 'image', 'telemetry', 'vendor', 'other', 'none']

# ---------------------------------------------------------------- server names: AES-128 / HKDF / QUIC Initial / TLS / HTTP
SBOX = np.array(bytearray.fromhex(
    '637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b27509832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cfd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2'
    'cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdbe0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9ee1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16'), np.uint8)
XT = np.array([((a << 1) ^ (0x1b if a & 0x80 else 0)) & 0xff for a in range(256)], np.uint8)
SHIFT = np.array([(i % 4) + 4 * (((i // 4) + (i % 4)) % 4) for i in range(16)])


def aes_keys(key):
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]; rc = 1
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = [int(SBOX[b]) for b in t[1:] + t[:1]]; t[0] ^= rc; rc = ((rc << 1) ^ (0x1b if rc & 0x80 else 0)) & 0xff
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return np.array([sum(w[4 * r:4 * r + 4], []) for r in range(11)], np.uint8)


def aes_enc(rk, blocks):  # uint8 [n, 16] -> [n, 16]
    s = blocks ^ rk[0]
    for r in range(1, 11):
        s = SBOX[s][:, SHIFT]
        if r < 10:
            c = s.reshape(-1, 4, 4); a0, a1, a2, a3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]; x0, x1, x2, x3 = XT[a0], XT[a1], XT[a2], XT[a3]
            s = np.stack([x0 ^ x1 ^ a1 ^ a2 ^ a3, a0 ^ x1 ^ x2 ^ a2 ^ a3, a0 ^ a1 ^ x2 ^ x3 ^ a3, x0 ^ a0 ^ a1 ^ a2 ^ x3], -1).reshape(-1, 16)
        s = s ^ rk[r]
    return s


def hkdf_label(secret, label, n):
    full = b'tls13 ' + label; info = n.to_bytes(2, 'big') + bytes([len(full)]) + full + b'\x00'; out, t, i = b'', b'', 1
    while len(out) < n: t = hmac.new(secret, t + info + bytes([i]), hashlib.sha256).digest(); out += t; i += 1
    return out[:n]


QV = {1: (bytes.fromhex('38762cf7f55934b34d179ae6a4c80cadccbb7f0a'), b'quic ', 0, 3),        # (salt, label prefix, Initial type, Retry type)
      0xff00001d: (bytes.fromhex('afbfec289993d24c9e9786f19c6111e04390a899'), b'quic ', 0, 3),
      0x6b3343cf: (bytes.fromhex('0dede3def700a6db819381be6e269dcbf9bd2ed9'), b'quicv2 ', 1, 0)}
_KC = {}


def client_keys(ver, dcid):
    if (ver, dcid) not in _KC:
        salt, pre = QV[ver][:2]; cs = hkdf_label(hmac.new(salt, dcid, hashlib.sha256).digest(), b'client in', 32)
        _KC[(ver, dcid)] = (aes_keys(hkdf_label(cs, pre + b'key', 16)), hkdf_label(cs, pre + b'iv', 12), aes_keys(hkdf_label(cs, pre + b'hp', 16)))
    return _KC[(ver, dcid)]


def selftest():
    rk = aes_keys(bytes(range(16)))
    assert aes_enc(rk, np.frombuffer(bytes.fromhex('00112233445566778899aabbccddeeff'), np.uint8)[None])[0].tobytes().hex() == '69c4e0d86a7b0430d8cdb78070b4c55a'
    cs = hkdf_label(hmac.new(QV[1][0], bytes.fromhex('8394c8f03e515708'), hashlib.sha256).digest(), b'client in', 32)
    assert cs.hex() == 'c00cf151ca5be075ed0ebfb5c80323c42d6b7db67881289af4008f1f6c357aea'
    assert hkdf_label(cs, b'quic key', 16).hex() == '1f369613dd76d5467730efcbe3b1a22d' and hkdf_label(cs, b'quic iv', 12).hex() == 'fa044b2f42a3fd3b46fb255c'
    hp = hkdf_label(cs, b'quic hp', 16); assert hp.hex() == '9f50449e04a0e810283a1e9933adedd2'
    assert aes_enc(aes_keys(hp), np.frombuffer(bytes.fromhex('d1b1c98dd7689fb8ec11d242b123dc9b'), np.uint8)[None])[0][:5].tobytes().hex() == '437b9aec36'


def varint(b, p):
    l = 1 << (b[p] >> 6); v = b[p] & 0x3f
    for i in range(1, l): v = (v << 8) | b[p + i]
    return v, p + l


def quic_initials(dg):  # (version, dcid, packet start, pn offset, packet end) of every long-header Initial in a datagram
    p = 0
    while p + 7 < len(dg) and dg[p] & 0x80:
        ver = int.from_bytes(dg[p + 1:p + 5], 'big')
        if ver not in QV: return
        q = p + 5; dl = dg[q]; dcid = bytes(dg[q + 1:q + 1 + dl]); q += 1 + dl; q += 1 + dg[q]; pt = (dg[p] >> 4) & 3
        if pt == QV[ver][3]: return  # Retry
        if pt == QV[ver][2]: tl, q = varint(dg, q); q += tl
        ln, q = varint(dg, q)
        if pt == QV[ver][2]: yield ver, dcid, p, q, q + ln
        p = q + ln


def decrypt_initial(dg, p0, pn_off, end, keys):
    key, iv, hp = keys
    if pn_off + 20 > len(dg): return None
    mask = aes_enc(hp, np.frombuffer(dg[pn_off + 4:pn_off + 20], np.uint8)[None])[0]
    pnl = ((dg[p0] ^ (int(mask[0]) & 0x0f)) & 3) + 1
    pn = int.from_bytes(bytes(a ^ int(m) for a, m in zip(dg[pn_off:pn_off + pnl], mask[1:1 + pnl])), 'big')
    ct = dg[pn_off + pnl:min(end, len(dg))]
    if end <= len(dg): ct = ct[:-16]  # AEAD tag (not verified: only the keystream is needed)
    nonce = bytes(a ^ b for a, b in zip(iv, pn.to_bytes(12, 'big'))); nb = (len(ct) + 15) // 16
    if nb == 0: return b''
    ctr = np.frombuffer(b''.join(nonce + (2 + j).to_bytes(4, 'big') for j in range(nb)), np.uint8).reshape(nb, 16)
    return (np.frombuffer(ct, np.uint8) ^ aes_enc(key, ctr).reshape(-1)[:len(ct)]).tobytes()


def crypto_frames(pt):  # CRYPTO frames of a client Initial; None if the first frame is not a known Initial frame
    p, out = 0, []
    try:
        while p < len(pt):
            t = pt[p]
            if t in (0x00, 0x01): p += 1
            elif t in (0x02, 0x03):
                p += 1; _, p = varint(pt, p); _, p = varint(pt, p); rc, p = varint(pt, p); _, p = varint(pt, p)
                for _ in range(2 * rc + (3 if t == 0x03 else 0)): _, p = varint(pt, p)
            elif t == 0x06:
                off, p = varint(pt, p + 1); ln, p = varint(pt, p); out.append((off, pt[p:p + ln])); p += ln
            else:
                return out if (out or t in (0x1c, 0x1d)) else None
    except IndexError:
        pass
    return out


def sni_clienthello(hs):  # handshake stream starting at the ClientHello header
    try:
        if len(hs) < 4 or hs[0] != 1: return None
        p = 4 + 2 + 32; p += 1 + hs[p]; p += 2 + int.from_bytes(hs[p:p + 2], 'big'); p += 1 + hs[p]
        end = min(p + 2 + int.from_bytes(hs[p:p + 2], 'big'), len(hs)); p += 2
        while p + 4 <= end:
            et, el = int.from_bytes(hs[p:p + 2], 'big'), int.from_bytes(hs[p + 2:p + 4], 'big')
            if et == 0:
                q = p + 6; nl = int.from_bytes(hs[q + 1:q + 3], 'big'); nm = hs[q + 3:q + 3 + nl]
                return nm.decode('ascii', 'replace').lower() if hs[q] == 0 and 0 < nl == len(nm) else None
            p += 4 + el
    except IndexError:
        return None
    return None


HTTP_M = (b'GET ', b'POST ', b'HEAD ', b'PUT ', b'OPTIONS ', b'CONNECT ', b'DELETE ', b'PATCH ')


def server_name(pdt, proto):
    """(name or None, how) from a biflow's first upstream payloads."""
    lp, raw, dr = pdt['L4_payload_bytes'], pdt['L4_raw_payload'], pdt['packet_dir']
    up = [bytes(raw[i]) for i in range(min(len(raw), 40)) if dr[i] == 0 and lp[i] > 0 and len(raw[i]) > 0][:8]
    if not up: return None, 'no_upstream_payload'
    if proto == 6:
        st = b''.join(up)
        if st[0] == 0x16:
            hs, p = b'', 0
            while p + 5 <= len(st) and st[p] == 0x16: ln = int.from_bytes(st[p + 3:p + 5], 'big'); hs += st[p + 5:p + 5 + ln]; p += 5 + ln
            nm = sni_clienthello(hs); return nm, ('tls' if nm else 'tls_no_sni')
        if st.startswith(HTTP_M):
            for line in st.split(b'\r\n\r\n', 1)[0].split(b'\r\n')[1:]:
                if line[:5].lower() == b'host:':
                    h = line[5:].strip().decode('ascii', 'replace').lower(); return re.sub(r':\d+$', '', h), 'http'
            return None, 'http_no_host'
        return None, ('handshake_missed' if st[0] in (0x14, 0x15, 0x17) and len(st) > 1 and st[1] == 0x03 else 'tcp_other')
    if proto == 17:
        first, frags = None, []
        for dg in up:
            for ver, dcid, p0, pn_off, end in quic_initials(dg):
                first = first or (ver, dcid)
                for kv in dict.fromkeys([first, (ver, dcid)]):
                    pt = decrypt_initial(dg, p0, pn_off, end, client_keys(*kv)); fr = crypto_frames(pt) if pt is not None else None
                    if fr: frags += fr; break
        if first is None: return None, ('handshake_missed' if (up[0][0] & 0xc0) == 0x40 else 'udp_other')
        if not frags: return None, 'quic_no_sni'
        mx = max(o + len(x) for o, x in frags); buf = bytearray(mx); have = np.zeros(mx, bool)
        for o, x in frags: buf[o:o + len(x)] = x; have[o:o + len(x)] = True
        nm = sni_clienthello(bytes(buf[:int(np.argmin(have)) if not have.all() else mx])); return nm, ('quic' if nm else 'quic_no_sni')
    return None, 'other_proto'


def host_plain(buf, span):  # scripts/leak_source.py host_from(): the name in the stored plaintext window
    s = int(span[0])
    if s < 0: return None
    b = bytes(buf)
    try:
        nl = int.from_bytes(b[s + 7:s + 9], 'big'); return b[s + 9:s + 9 + nl].decode('ascii', 'replace').lower() or None
    except Exception:
        return None


# ---------------------------------------------------------------- categories (see docstring)
IMG_RE = re.compile(r'^files\.oaiusercontent\.com$|^lh3\.googleusercontent\.com$|\.mm\.bing\.net$')
TEL_RE = re.compile(r'datadoghq|sentry\.io|statsig|^ab\.chatgpt\.com$|appcenter\.ms|adjust\.com|events\.data\.microsoft\.com|app-measurement|'
                    r'google-analytics|crashlytics|firebaseinstallations|firebaselogging|clienttracing|doubleclick|googleadservices|'
                    r'googlesyndication|mixpanel|amplitude|segment\.io|branch\.io|appsflyer|clarity\.ms')
ASSIST_RE = {0: re.compile(r'(^|\.)openai\.com$|(^|\.)chatgpt\.com$|oaistatic|oaiusercontent'),
             1: re.compile(r'copilot|sydney|(^|\.)bing\.com$|bingviz'),
             2: re.compile(r'gemini|bard|makersuite|proactivebackend|generativelanguage|assistant')}
VENDOR_RE = {0: None,
             1: re.compile(r'(^|\.)(microsoft\.com|microsoftonline\.com|office\.com|office\.net|live\.com|sfx\.ms|msn\.com|windows\.net|skype\.com|microsoftapp\.net)$|azure'),
             2: re.compile(r'(^|\.)google\.[a-z.]+$|(^|\.)(googleapis\.com|gstatic\.com|googleusercontent\.com|youtube\.com|ytimg\.com|ggpht\.com|gvt1\.com|gvt2\.com|googlevideo\.com|android\.com|1e100\.net)$')}


def category(name, app):
    if name is None: return 'none'
    if IMG_RE.search(name): return 'image'
    if TEL_RE.search(name): return 'telemetry'
    if ASSIST_RE[app].search(name): return 'assistant'
    if VENDOR_RE[app] is not None and VENDOR_RE[app].search(name): return 'vendor'
    return 'other'


fold = lambda h: re.sub(r'\d+', '#', h) if h else '<none>'
ms = lambda v: [float(np.mean(v)), float(np.std(v))]


def auc_boot(score, lab, W):  # AUROC under session resampling weights W [B, n]; ties count 1/2
    C = (score[:, None] > score[None, :]) + 0.5 * (score[:, None] == score[None, :]); pos, neg = lab == 1, lab == 0
    num = np.einsum('bi,ij,bj->b', W[:, pos], C[np.ix_(pos, neg)], W[:, neg]); den = W[:, pos].sum(1) * W[:, neg].sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def main():
    selftest()
    D = GenAIData(ROOT, prefix='genai'); N = len(D.meta); assert len(D.index) == N
    ses = D.sessions; files = dict(zip(ses.session_id, ses.file))
    row_of = {(files[s], k): i for i, (s, k) in enumerate(zip(D.index.session_id, D.index.biflow_key))}
    H = np.array([None] * N, dtype=object); HOW = np.array([''] * N, dtype=object)
    contact = collections.defaultdict(set)  # (file, package) -> set of folded hosts, every biflow of the session
    for f in ses.file:
        d = json.load(open(os.path.join(RAW, f)))
        for key, b in d.items():
            nm, how = server_name(b['packet_data'], int(key.split(',')[-1]))
            if nm: contact[(f, b['flow_metadata']['BF_label'])].add(fold(nm))
            if (f, key) in row_of: H[row_of[(f, key)]] = nm; HOW[row_of[(f, key)]] = how
    assert all(HOW != ''), 'every derived biflow must be matched to its raw record'
    Hp = []
    for i in range(N):
        h = host_plain(D.flat_bytes[i], D.flat_sni[i])
        for p in range(D.pkt_sni.shape[1]):
            if h: break
            h = host_plain(D.pkt_bytes[i, p], D.pkt_sni[i, p])
        Hp.append(h)
    Hp = np.array(Hp, dtype=object)
    lab = D.y_joint >= 0; ya_all = D.y_app
    both = lab & (Hp != None) & (H != None)  # noqa: E711
    names = dict(n_labelled=int(lab.sum()), share_named_plain_window=float(np.mean(Hp[lab] != None)),  # noqa: E711
                 share_named_raw=float(np.mean(H[lab] != None)), agreement_where_both=float(np.mean(H[both] == Hp[both])),  # noqa: E711
                 how_by_app={APPS[a]: dict(collections.Counter(HOW[lab & (ya_all == a)])) for a in range(3)})
    CAT = np.array([category(H[i], ya_all[i]) if lab[i] else '' for i in range(N)], dtype=object)
    FH = np.array([fold(h) for h in H], dtype=object); IMGH = np.array([bool(h and IMG_RE.search(h)) for h in H])
    print(f'server name: plaintext window {100*names["share_named_plain_window"]:.1f}%  raw (TLS+QUIC+HTTP) {100*names["share_named_raw"]:.1f}%  '
          f'agreement {100*names["agreement_where_both"]:.2f}%')

    # ------------------------------------------------ predictions (test sessions identical across seeds)
    Fk = feats_knn(D.meta, D.meta_len); y6 = D.y_joint
    preds, missing, T = {}, [], None
    for K in KS:
        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'fewshot_k{K}_s{s}.json')); tr, it = idx['train'], idx['test']
            if T is None: T = it
            assert np.array_equal(T, it), 'the test sessions must be identical across seeds and K'
            p1 = np.empty(len(it), int)
            for b0 in range(0, len(it), 512): p1[b0:b0 + 512] = y6[tr][np.abs(Fk[it][b0:b0 + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)]
            preds[('knn1', K, s)] = p1
            for o, fmt in RUNS.items():
                p = os.path.join(ROOT, 'results', 'runs', fmt.format(K=K, s=s), 'test_preds.npz')
                if not os.path.exists(p): missing.append(fmt.format(K=K, s=s)); continue
                z = np.load(p); assert np.array_equal(z['idx'], it) and np.array_equal(z['y'], y6[it]), (o, K, s)
                preds[(o, K, s)] = z['pred']
    obs = [o for o in OBS if all((o, K, s) in preds for K in KS for s in SEEDS)]
    y = y6[T]; ya, ym = y // 2, y % 2; sid = D.session_id[T]; us, inv = np.unique(sid, return_inverse=True); nS = len(us)
    s_app = np.array([ya[inv == j][0] for j in range(nS)]); s_mod = np.array([ym[inv == j][0] for j in range(nS)])
    cat, fh, imgh = CAT[T], FH[T], IMGH[T]
    rng = np.random.RandomState(12345)
    out = dict(protocol=__doc__, observers=obs, runs_missing=missing, n_test_connections=int(len(T)), n_test_sessions=int(nS),
               server_names=names, q1={}, q2={}, q3={})

    # ------------------------------------------------ (1) assistant decision by server-name category
    Wapp = {a: rng.multinomial(int((s_app == a).sum()), np.ones(int((s_app == a).sum())) / (s_app == a).sum(), size=B) for a in range(3)}
    for K in KS:
        rk = dict(per_app={}, confusion={}, confusion_rownorm={}, f1={}, session_vote={})
        for a in range(3):
            ma = ya == a; sa = np.where(s_app == a)[0]; loc = {j: t for t, j in enumerate(sa)}
            ra = dict(n=int(ma.sum()), categories={}, hosts={})
            for c in CATS:
                m = ma & (cat == c); e = dict(n=int(m.sum()), share=float(m.sum() / ma.sum()), recall={}, tp_share={})
                for o in obs:
                    rec, tps = [], []
                    for s in SEEDS:
                        pa = preds[(o, K, s)] // 2
                        rec.append(float(np.mean(pa[m] == a)) if m.any() else np.nan); tps.append(float(np.sum(pa[m] == a) / max(np.sum(pa[ma] == a), 1)))
                    e['recall'][o] = ms(rec) if m.any() else None; e['tp_share'][o] = float(np.mean(tps))
                    if K == HEAD_K and m.any():
                        n_j = np.zeros(len(sa)); h_j = np.zeros(len(sa))
                        np.add.at(n_j, [loc[j] for j in inv[m]], 1)
                        np.add.at(h_j, [loc[j] for j in inv[m]], np.mean([preds[(o, K, s)][m] // 2 == a for s in SEEDS], 0))
                        r = (Wapp[a] @ h_j) / np.where(Wapp[a] @ n_j > 0, Wapp[a] @ n_j, np.nan)
                        e.setdefault('recall_ci95', {})[o] = [float(v) for v in np.nanpercentile(r, [2.5, 97.5])]
                e['n_sessions'] = int(len(np.unique(sid[m])))
                la = lab & (ya_all == a); e['share_all_labelled'] = float(np.mean(CAT[la] == c))  # all splits, for reference
                ra['categories'][c] = e
            for h, n in collections.Counter(fh[ma]).most_common():
                if n / ma.sum() < HOST_MIN: break
                m = ma & (fh == h)
                ra['hosts'][h] = dict(n=int(n), share=float(n / ma.sum()), category=str(collections.Counter(cat[m]).most_common(1)[0][0]),
                                      n_sessions=int(len(np.unique(sid[m]))), n_sessions_text=int(len(np.unique(sid[m & (ym == 0)]))),
                                      n_sessions_image=int(len(np.unique(sid[m & (ym == 1)]))),
                                      recall={o: ms([np.mean(preds[(o, K, s)][m] // 2 == a) for s in SEEDS]) for o in obs})
            rk['per_app'][APPS[a]] = ra
        for o in obs:
            cms, f1s = [], []
            for s in SEEDS:
                pa = preds[(o, K, s)] // 2; cm = np.zeros((3, 3)); np.add.at(cm, (ya, pa), 1); cms.append(cm)
                f1s.append(100 * f1_score(ya, pa, labels=[0, 1, 2], average=None, zero_division=0))
            cm = np.mean(cms, 0); rk['confusion'][o] = cm.round(1).tolist(); rk['confusion_rownorm'][o] = (cm / cm.sum(1, keepdims=True)).round(4).tolist()
            f1s = np.array(f1s); rk['f1'][o] = dict({APPS[a]: ms(f1s[:, a]) for a in range(3)}, macro=ms(f1s.mean(1)))
            rules = {'all': np.ones(len(T), bool), 'without_vendor': cat != 'vendor', 'assistant_and_image_only': np.isin(cat, ['assistant', 'image'])}
            sv = {}
            for rn, keep in rules.items():
                accs = {APPS[a]: [] for a in range(3)}; empty = {APPS[a]: 0 for a in range(3)}
                for s in SEEDS:
                    pa = preds[(o, K, s)] // 2
                    for j in range(nS):
                        m = (inv == j) & keep
                        v = np.bincount(pa[m], minlength=3).argmax() if m.any() else -1
                        accs[APPS[s_app[j]]].append(v == s_app[j])
                        if s == 0 and not m.any(): empty[APPS[s_app[j]]] += 1
                nS_app = {APPS[a]: int((s_app == a).sum()) for a in range(3)}
                sv[rn] = dict(acc={a: 100 * float(np.mean(v)) for a, v in accs.items()}, sessions_without_connections=empty,
                              acc_on_sessions_with_connections={a: 100 * float(np.mean(v)) * nS_app[a] / (nS_app[a] - empty[a]) if nS_app[a] > empty[a] else None
                                                                for a, v in accs.items()})
            rk['session_vote'][o] = sv
        out['q1'][f'K{K}'] = rk
    # background check: who contacts the Gemini-class hosts, in which sessions (all splits, generic sessions)
    gen = ses[ses.part == 'generic']; bgc = {}
    for h in out['q1'][f'K{HEAD_K}']['per_app']['Gemini']['hosts']:
        if h == '<none>': continue
        r = {}
        for a, ad in enumerate(['Chatgpt', 'Copilot', 'Gemini']):
            fs = gen[gen.app_dir == ad].file.values
            r[APPS[a]] = dict(n_sessions=int(len(fs)), google_app=float(np.mean([h in contact.get((f, GOOGLE_APP), ()) for f in fs])),
                              any_package=float(np.mean([any(h in v for (ff, _), v in contact.items() if ff == f) for f in fs])))
        gm = gen[gen.app_dir == 'Gemini']
        for act in ('text', 'multi'):
            fs = gm[gm.activity == act].file.values; r[f'Gemini_{act}_google_app'] = float(np.mean([h in contact.get((f, GOOGLE_APP), ()) for f in fs]))
        bgc[h] = r
    out['q1']['background_contact'] = bgc

    # ------------------------------------------------ (2) session-level modality
    W = rng.multinomial(nS, np.ones(nS) / nS, size=B)
    ref = np.array([np.mean(imgh[inv == j]) for j in range(nS)])
    out['q2']['reference_image_host_share'] = dict(auroc=float(roc_auc_score(s_mod, ref)), ci95=[float(v) for v in np.nanpercentile(auc_boot(ref, s_mod, W), [2.5, 97.5])],
                                                   within_app={APPS[a]: float(roc_auc_score(s_mod[s_app == a], ref[s_app == a])) for a in range(3)})
    for K in KS:
        rk = {}
        for o in obs:
            au, wa, mv, ff, bs = [], {APPS[a]: [] for a in range(3)}, [], [], []
            for s in SEEDS:
                pm = preds[(o, K, s)] % 2; sh = np.array([np.mean(pm[inv == j]) for j in range(nS)])
                au.append(roc_auc_score(s_mod, sh)); bs.append(auc_boot(sh, s_mod, W)); mv.append(100 * np.mean((sh > 0.5) == s_mod))
                ff.append(100 * f1_score(ym, pm, average='macro'))
                for a in range(3): wa[APPS[a]].append(roc_auc_score(s_mod[s_app == a], sh[s_app == a]))
            rk[o] = dict(auroc=ms(au), auroc_ci95=[float(v) for v in np.nanpercentile(np.mean(bs, 0), [2.5, 97.5])],
                         auroc_within_app={a: ms(v) for a, v in wa.items()}, majority_vote_acc=ms(mv), flow_modality_macro_f1=ms(ff))
        out['q2'][f'K{K}'] = rk

    # ------------------------------------------------ (3) image-delivery connections vs all others
    out['q3']['image_fetch_share_of_image_session_connections'] = {APPS[a]: float(np.mean(imgh[(ya == a) & (ym == 1)])) for a in range(3)}
    out['q3']['image_fetch_share_of_text_session_connections'] = {APPS[a]: float(np.mean(imgh[(ya == a) & (ym == 0)])) for a in range(3)}
    subsets = {'image_host': imgh, 'other': ~imgh}
    for K in KS:
        rk = {}
        for o in obs:
            r = {}
            for sn, m in subsets.items():
                e = dict(n=int(m.sum()), n_image_sessions=int((m & (ym == 1)).sum()), n_text_sessions=int((m & (ym == 0)).sum()))
                rec, fpr, f1b, f1m = [], [], [], []
                for s in SEEDS:
                    pm = preds[(o, K, s)] % 2
                    rec.append(100 * np.mean(pm[m & (ym == 1)] == 1) if (m & (ym == 1)).any() else np.nan)
                    fpr.append(100 * np.mean(pm[m & (ym == 0)] == 1) if (m & (ym == 0)).any() else np.nan)
                    f1b.append(100 * f1_score(ym[m], pm[m], pos_label=1, average='binary', zero_division=0))
                    f1m.append(100 * f1_score(ym[m], pm[m], labels=sorted(set(ym[m].tolist())), average='macro', zero_division=0))
                e.update(recall_image=ms(rec) if not np.isnan(rec).all() else None, fpr_text=ms(fpr) if not np.isnan(fpr).all() else None,
                         f1_image=ms(f1b), macro_f1_present=ms(f1m))
                r[sn] = e
            pa_img = {}
            for a in range(3):
                m = imgh & (ya == a)
                if m.any():
                    pa_img[APPS[a]] = dict(n=int(m.sum()), pred_image=ms([100 * np.mean(preds[(o, K, s)][m] % 2 == 1) for s in SEEDS]),
                                           pred_exact_class=ms([100 * np.mean(preds[(o, K, s)][m] == y[m]) for s in SEEDS]))
            r['image_host_per_app'] = pa_img; rk[o] = r
        out['q3'][f'K{K}'] = rk

    json.dump(out, open(os.path.join(ROOT, 'results', 'construct_validity.json'), 'w'), indent=1, default=float)
    # ------------------------------------------------ summary
    print('runs missing:', missing or 'none')
    q = out['q1'][f'K{HEAD_K}']
    for a in APPS:
        ra = q['per_app'][a]; print(f'\n== K={HEAD_K} {a}: {ra["n"]} test connections')
        for c, e in ra['categories'].items():
            if e['n'] == 0: continue
            print(f'  {c:9s} n={e["n"]:4d} share {100*e["share"]:5.1f}% sess {e["n_sessions"]:2d}  ' +
                  '  '.join(f'{o} rec {100*e["recall"][o][0]:5.1f}±{100*e["recall"][o][1]:4.1f} [{100*e["recall_ci95"][o][0]:.0f},{100*e["recall_ci95"][o][1]:.0f}] tp {100*e["tp_share"][o]:4.1f}%' for o in obs))
        for h, e in ra['hosts'].items():
            print(f'    {h:42s} {e["category"]:9s} n={e["n"]:4d} {100*e["share"]:5.1f}% sess {e["n_sessions"]:2d} (t{e["n_sessions_text"]}/i{e["n_sessions_image"]})  ' +
                  '  '.join(f'{o} {100*e["recall"][o][0]:5.1f}' for o in obs))
    for o in obs:
        print(f'\n{o}: confusion (rows true ChatGPT/Copilot/Gemini) {q["confusion_rownorm"][o]}  F1 ' + ' '.join(f'{a} {v[0]:.1f}' for a, v in q['f1'][o].items()))
        print('   session vote ' + '  '.join(f'{rn}: ' + ','.join(f'{a[:3]} {v:.0f}' for a, v in sv['acc'].items()) + f' empty {sv["sessions_without_connections"]}' for rn, sv in q['session_vote'][o].items()))
    print('\nbackground contact', json.dumps(bgc, indent=None)[:2000])
    print('\n(2) session modality AUROC; reference image-host share', out['q2']['reference_image_host_share'])
    for K in KS:
        print(f'K={K} ' + '  '.join(f'{o} {out["q2"][f"K{K}"][o]["auroc"][0]:.3f}±{out["q2"][f"K{K}"][o]["auroc"][1]:.3f} [{out["q2"][f"K{K}"][o]["auroc_ci95"][0]:.2f},{out["q2"][f"K{K}"][o]["auroc_ci95"][1]:.2f}] '
                                    f'maj {out["q2"][f"K{K}"][o]["majority_vote_acc"][0]:.1f} within ' +
                                    '/'.join(f'{v[0]:.2f}' for v in out["q2"][f"K{K}"][o]["auroc_within_app"].values()) for o in obs))
    print('\n(3) image-fetch share of image-session connections', out['q3']['image_fetch_share_of_image_session_connections'])
    for K in KS:
        for o in obs:
            r = out['q3'][f'K{K}'][o]
            print(f'K={K} {o:5s} ' + '  '.join(f'{sn}: n={e["n"]} rec {e["recall_image"][0] if e["recall_image"] else float("nan"):5.1f} '
                                               f'fpr {e["fpr_text"][0] if e["fpr_text"] else float("nan"):5.1f} F1img {e["f1_image"][0]:5.1f} mF1 {e["macro_f1_present"][0]:5.1f}'
                                               for sn, e in r.items() if sn != 'image_host_per_app') +
                  '  per-app img ' + ' '.join(f'{a} {v["pred_image"][0]:.0f}' for a, v in r['image_host_per_app'].items()))
    print('\nwritten results/construct_validity.json')


if __name__ == '__main__':
    main()
