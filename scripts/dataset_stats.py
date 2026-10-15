"""Statistics of the three datasets after filtering: the dataset table (capture sessions and connections of the classes
used) and the further numbers of the appendix (connections per session, TCP / QUIC / other UDP shares with the definition
of scripts/leak_source.py, i.e. QUIC = a UDP connection whose first packet carries at least 1,200 IP bytes, the session
split sizes, and the sessions of the third set per app and per phone).
Reads data/derived/ and configs/splits/ only; writes results/dataset_stats.json."""
import json, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from make_splits_appact import NINE  # noqa: E402

DER = os.path.join(ROOT, 'data', 'derived'); SPL = os.path.join(ROOT, 'configs', 'splits')
SETS = {'genai': ('session_split_s2026.json', 'y_joint'), 'ccma': ('ccma_session_split_s2026.json', 'y_app'),
        'appact': ('appact_session_split_s2026.json', None)}
OUT = {}
for ds, (split, col) in SETS.items():
    idx = pd.read_csv(os.path.join(DER, f'{ds}_index.csv')); z = np.load(os.path.join(DER, f'{ds}_biflows.npz'))
    lab = (idx.app.isin(NINE) if col is None else idx[col] >= 0).values
    proto = idx.proto.values; quic = (proto == 17) & (z['meta'][:, 0, 1] >= 1200)
    n_ses = int(idx.session_id[lab].nunique()); n_con = int(lab.sum())
    sp = json.load(open(os.path.join(SPL, split)))
    OUT[ds] = dict(sessions=n_ses, connections=n_con, connections_per_session=n_con / n_ses,
                   transport_pct=dict(tcp=100 * float(np.mean(proto[lab] == 6)), quic=100 * float(np.mean(quic[lab])),
                                      other_udp=100 * float(np.mean((proto[lab] == 17) & ~quic[lab]))),
                   split_sessions={k: len(sp[k]) for k in ('train', 'val', 'test')})
ses = pd.read_csv(os.path.join(DER, 'appact_sessions.csv')); ses = ses[ses.n_kept > 0]
OUT['appact']['sessions_per_app_all_eleven'] = {a: int(n) for a, n in ses.groupby('app').size().items()}
OUT['appact']['sessions_per_phone'] = {p: int(n) for p, n in ses[ses.app.isin(NINE)].device.str[:2].value_counts().items()}
json.dump(OUT, open(os.path.join(ROOT, 'results', 'dataset_stats.json'), 'w'), indent=1)
for ds, r in OUT.items():
    t = r['transport_pct']
    print(f"{ds:7s} sessions {r['sessions']:4d}  connections {r['connections']:6,d}  per session {r['connections_per_session']:.1f}  "
          f"TCP/QUIC/other UDP {t['tcp']:.1f}/{t['quic']:.1f}/{t['other_udp']:.1f}  split " + '/'.join(str(v) for v in r['split_sessions'].values()))
print('third set, sessions per app (all eleven apps):', OUT['appact']['sessions_per_app_all_eleven'])
print('third set, sessions per phone (nine apps):', OUT['appact']['sessions_per_phone'])
print('written results/dataset_stats.json')
