"""Where the leak is (training-free analyses on the existing data and saved predictions; no model is trained):
  1. transport mix per class (TCP / QUIC = UDP 443 / other UDP),
  2. server identity: the server name each flow's plaintext ClientHello carries (ground truth only; no attacker reads it),
     (a) a 'server-name lookup' attacker trained on the K labelled sessions (majority class per name) as the upper bound of
         what destination identity alone reveals, compared with the pre-trained encoder's predictions,
     (b) re-identification of the exact server name from the metadata of the first 10 packets by a 1-NN over the K sessions'
         flows (the same features as the input-space 1-NN attacker), on test flows whose name occurs in those sessions,
  3. the input-space 1-NN restricted to different 10-packet windows of the connection,
  4. on GenAI, the share of flows to the hosts that serve generated images.
Writes results/leak_source.json."""
import collections, json, os, re, sys
import numpy as np
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData

IMG = re.compile(r'(oaiusercontent|googleusercontent|mm\.bing\.net)')   # hosts that serve generated images (Montieri et al.)
NC = {'genai': 6, 'ccma': 9}


def host_from(buf, span):
    s = int(span[0])
    if s < 0: return None
    b = bytes(buf)
    try:
        nl = int.from_bytes(b[s + 7:s + 9], 'big'); h = b[s + 9:s + 9 + nl].decode('ascii', 'replace').lower()
        return h or None
    except Exception:
        return None


def hosts(d):
    out = []
    for i in range(len(d.meta)):
        h = host_from(d.flat_bytes[i], d.flat_sni[i])
        if h is None:
            for p in range(d.pkt_sni.shape[1]):
                h = host_from(d.pkt_bytes[i, p], d.pkt_sni[i, p])
                if h: break
        out.append(h or '<none>')
    return np.array(out, dtype=object)


def feats(meta, n, a, b):  # the input-space 1-NN features (scripts/trivial_baselines.py) on packets a..b-1
    m = meta[:, a:b]; valid = np.arange(a, b)[None, :] < n[:, None]
    x = np.stack([m[..., 0], np.log1p(m[..., 1]), np.log1p(m[..., 2]), np.log1p(m[..., 3] * 1000.0)], -1); x[~valid] = 0
    return x.reshape(len(x), -1).astype(np.float32)


def nn1(Ftr, Fq):
    out = np.empty(len(Fq), int)
    for b in range(0, len(Fq), 256): out[b:b + 256] = np.abs(Fq[b:b + 256, None, :] - Ftr[None]).sum(-1).argmin(1)
    return out


mf1 = lambda y, p, nc: 100 * f1_score(y, p, labels=list(range(nc)), average='macro', zero_division=0)
OUT = {}
for ds, tgt, pre in (('genai', 'joint', ''), ('ccma', 'app', 'ccma_')):
    d = GenAIData(ROOT, prefix=ds); y = d.y_joint if tgt == 'joint' else d.y_app; lab = y >= 0
    H = hosts(d); proto = d.index['proto'].values; n = d.meta_len
    first_dl = d.meta[:, 0, 1]  # IP size of the first packet (1200+ B for a QUIC Initial)
    quic = (proto == 17) & (first_dl >= 1200)
    res = dict(n_flows=int(lab.sum()), share_with_server_name=float(np.mean(H[lab] != '<none>')),
               transport={str(c): dict(tcp=float(np.mean(proto[lab & (y == c)] == 6)), quic=float(np.mean(quic[lab & (y == c)])),
                                       other_udp=float(np.mean((proto[lab & (y == c)] == 17) & ~quic[lab & (y == c)]))) for c in range(NC[ds])},
               transport_all=dict(tcp=float(np.mean(proto[lab] == 6)), quic=float(np.mean(quic[lab])), other_udp=float(np.mean((proto[lab] == 17) & ~quic[lab]))))
    if ds == 'genai':
        img = np.array([bool(IMG.search(h)) for h in H])
        res['image_host_share'] = {a: float(img[lab & (d.y_act == k)].mean()) for k, a in ((0, 'text'), (1, 'multimodal'))}
    wins = {'1-10': (0, 10), '11-20': (10, 20), '21-30': (20, 30), '11-64': (10, 64)}
    F = {k: feats(d.meta, n, a, b) for k, (a, b) in wins.items()}
    per_k = {}
    for K in (1, 4, 8):
        acc = collections.defaultdict(list)
        for s in range(5):
            idx, _ = d.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{pre}fewshot_k{K}_s{s}.json')); tr, te = idx['train'], idx['test']
            cnt = collections.defaultdict(collections.Counter)
            for i in tr: cnt[H[i]][y[i]] += 1
            maj = np.bincount(y[tr], minlength=NC[ds]).argmax()
            pl = np.array([cnt[H[i]].most_common(1)[0][0] if H[i] in cnt else maj for i in te])
            acc['lookup_f1'].append(mf1(y[te], pl, NC[ds]))
            if ds == 'genai': acc['lookup_app_f1'].append(100 * f1_score(y[te] // 2, pl // 2, average='macro'))
            z = np.load(os.path.join(ROOT, 'results', 'runs', f'{pre}fs{K}_sslXL_s{s}', 'test_preds.npz')); pos = {v: k for k, v in enumerate(z['idx'])}
            pe = z['pred'][[pos[i] for i in te]]
            acc['encoder_f1'].append(mf1(y[te], pe, NC[ds]))
            if ds == 'genai': acc['encoder_app_f1'].append(100 * f1_score(y[te] // 2, pe // 2, average='macro'))
            for k in F:
                nb = tr[nn1(F[k][tr], F[k][te])]; acc[f'knn_{k}_f1'].append(mf1(y[te], y[nb], NC[ds]))
                if k == '1-10':
                    named = set(H[tr]) - {'<none>'}; sel = np.array([H[i] in named for i in te])
                    acc['server_reid_first10'].append(100 * float(np.mean(H[nb][sel] == H[te][sel])))
                    acc['server_reid_coverage'].append(100 * float(sel.mean()))
                    # reference: the most frequent server name of the test connection's TRUE class (an oracle class prior)
                    tt = collections.defaultdict(collections.Counter)
                    for i in tr:
                        if H[i] != '<none>': tt[y[i]][H[i]] += 1
                    acc['server_reid_class_prior'].append(100 * float(np.mean([tt[y[i]].most_common(1)[0][0] == H[i] if tt[y[i]] else False for i in te[sel]])))
            acc['share_len_gt10'].append(100 * float(np.mean(n[te] > 10)))
        per_k[K] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
    res['per_K'] = per_k; OUT[ds] = res
    print(f'\n===== {ds}: flows {res["n_flows"]}, with server name {100*res["share_with_server_name"]:.1f}%  transport {res["transport_all"]}')
    if ds == 'genai': print('image-host share', res['image_host_share'])
    for K, r in per_k.items(): print(f'K={K} ' + '  '.join(f'{k} {v[0]:.1f}' for k, v in r.items()))
json.dump(OUT, open(os.path.join(ROOT, 'results', 'leak_source.json'), 'w'), indent=1)
print('\nwritten results/leak_source.json')
