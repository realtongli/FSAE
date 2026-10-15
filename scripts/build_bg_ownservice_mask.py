"""Mark the GenAI background connections that belong to a monitored assistant's own service.

MIRAGE-GenAI-2025 attributes every biflow to a package. scripts/build_background.py keeps the biflows of all packages other
than the three monitored ones as 'background'. Part of them carry a server name of an assistant and end at its front-end
operator, in a capture session of that assistant, under a Google Play services package name. They are traffic of the
assistant, so the open world does not count them as negatives of the test traffic (scripts/ow_rescore.py).

Rule:
  A background connection is marked when (1) its server address is attributed (scripts/operator_confound.Attr: archived
  published lists, replayed RDAP) to Cloudflare, Microsoft or Akamai, the home operators of ChatGPT (Cloudflare) and
  Copilot (Microsoft, Akamai), and (2) its plaintext server name occurs among the monitored connections of that assistant
  anywhere in the campaign. Connections that end at Google are never marked, whatever their name: Google Play services
  talks to Google on its own account, and the Gemini class is the Google app.
Nothing else is changed. Writes data/derived/genai_background_ownservice.npy (bool, one entry per row of
genai_background.npz) and prints the counts."""
import collections, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import operator_confound as OC  # noqa: E402
from ip_baseline import server_addr  # noqa: E402
from data_mfr import GenAIData  # noqa: E402

HOME = {'Cloudflare': 0, 'Microsoft': 1, 'Akamai': 1}   # operator -> assistant (0 ChatGPT, 1 Copilot)
AT = OC.Attr(); bg = OC.background_addresses(); z = np.load(os.path.join(ROOT, 'data', 'derived', 'genai_background.npz'), allow_pickle=True)
chk = collections.Counter(); op = np.array([AT(server_addr(k, chk))['operator'] for k in bg['key']], dtype=object)
G = GenAIData(ROOT, prefix='genai'); y = G.y_joint; lab = y >= 0
Hm = np.array([OC.host_of(G.pkt_bytes[i], G.pkt_sni[i]) for i in range(len(G.meta))], dtype=object)
own = {a: set(Hm[lab & (y // 2 == a)]) - {'<none>'} for a in (0, 1)}
Hb = np.array([OC.host_of(z['pkt_bytes'][i], z['pkt_sni'][i]) for i in range(len(op))], dtype=object)
mask = np.array([op[i] in HOME and Hb[i] in own[HOME[op[i]]] for i in range(len(op))])
np.save(os.path.join(ROOT, 'data', 'derived', 'genai_background_ownservice.npy'), mask)
print(f'marked {int(mask.sum())} of {len(mask)} background connections ({100 * mask.mean():.1f}%)')
print('by operator:', dict(collections.Counter(op[mask])))
print('by package:', collections.Counter(z['pkg'][mask]).most_common(6))
print('by server name:', collections.Counter(Hb[mask]).most_common(8))
s2a = {int(s): int(np.bincount(y[lab & (G.session_id == s)]).argmax() // 2) for s in np.unique(G.session_id[lab])}
sa = np.array([s2a.get(int(s), -1) for s in z['session_id']])
print('assistant of the capture session (0 ChatGPT, 1 Copilot, 2 Gemini, -1 none):', dict(collections.Counter(sa[mask].tolist())))
print('marked connections in a session of ANOTHER assistant:', int(sum(sa[i] != HOME[op[i]] for i in np.where(mask)[0])))
