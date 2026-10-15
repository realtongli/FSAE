"""Build derived per-biflow arrays for the part of MIRAGE-AppAct-2024 that does NOT overlap MIRAGE-COVID-CCMA-2022.

MIRAGE-AppAct-2024 is a superset of CCMA-2022: 993 of the 998 CCMA capture sessions are in it (same app, same capture
timestamp). To keep the third evaluation set disjoint from CCMA, this script keeps only the eleven apps that CCMA does
not contain, and refuses any capture whose timestamp appears in data/derived/ccma_sessions.csv (run
scripts/build_dataset_ccma.py first). The evaluation uses the nine of them with at least 50 sessions (make_splits_appact.py).

Input   data/MIRAGE-AppAct-2024.zip  (read-only, streamed; layout MIRAGE-AppAct-2024/<AppDir>/<capture>.json)
Sample unit = biflow whose netstat label (BF_label) is the app of the capture, >= 1 payload packet.
Differences from CCMA handled here: BF_label holds the app *name* (e.g. "Signal"), not the package; BF_activity is a
composite string (e.g. "audiocall-chat", "video-streaming") or missing, so it is stored but never used to filter;
a few captures sit in another app's folder, so the app is taken from the package in the file name when it is known.

Same arrays as build_dataset_ccma.py (pkt_bytes 5x320, pkt_len, pkt_dir, pkt_sni, flat 1024, meta 64x4, meta_len).
Labels: y_app (index into APPS below), y_act = -1 (unused), y_joint = y_app, session_id (one JSON capture = one
session), device_id. Outputs data/derived/appact_biflows.npz, appact_index.csv, appact_sessions.csv,
appact_build_report.md.

Usage:  python scripts/build_dataset_appact.py
        python scripts/build_dataset_appact.py --sample-dir <dir with AppDir/*.json> --out <dir>   (smoke test)
"""
import argparse, collections, json, os, re, sys, time, zipfile
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from build_dataset import find_sni_span, K_PKT, PKT_BYTES, FLAT_BYTES, META_LEN

ZIP = os.path.join(ROOT, 'data', 'MIRAGE-AppAct-2024.zip')
OUT = os.path.join(ROOT, 'data', 'derived')
CCMA_SESSIONS = os.path.join(OUT, 'ccma_sessions.csv')
# folder in the archive -> (Android package, official app name); the eleven apps absent from CCMA-2022
APPS = {
    'ClashRoyale': ('com.supercell.clashroyale', 'Clash Royale'),
    'Crunchyroll': ('com.crunchyroll.crunchyroid', 'Crunchyroll'),
    'JitsiMeet': ('org.jitsi.meet', 'Jitsi Meet'),
    'KakaoTalk': ('com.kakao.talk', 'KakaoTalk'),
    'Line': ('jp.naver.line.android', 'Line'),
    'Omlet': ('mobisocial.arcade', 'Omlet Arcade'),
    'Signal': ('org.thoughtcrime.securesms', 'Signal'),
    'Telegram': ('org.telegram.messenger', 'Telegram'),
    'Trueconf': ('com.trueconf.videochat', 'Trueconf'),
    'Twitch': ('tv.twitch.android.app', 'Twitch'),
    'WhatsApp': ('com.whatsapp', 'WhatsApp'),
}
CCMA_PACKAGES = {'com.discord', 'com.gotomeeting', 'com.google.android.apps.meetings', 'com.facebook.orca', 'com.skype.raider',
                 'com.Slack', 'com.microsoft.teams', 'com.cisco.webex.meetings', 'us.zoom.videomeetings'}
APP_ID = {a: i for i, a in enumerate(APPS)}
PKG_TO_APP = {pkg: app for app, (pkg, _) in APPS.items()}
NAME_RE = re.compile(r'^(\d{10})_(?:((?:[0-9a-f]{2}[:\-]){5}[0-9a-f]{2})_)?(.*?)_mirage')


def norm(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


ALIASES = {app: {norm(app), norm(name), norm(pkg)} for app, (pkg, name) in APPS.items()}


def iter_captures(args):
    """Yield (app folder, file name, loader) for every capture JSON, from the archive or from a sample directory."""
    if args.sample_dir:
        for folder in sorted(os.listdir(args.sample_dir)):
            path = os.path.join(args.sample_dir, folder)
            if os.path.isdir(path):
                for fn in sorted(os.listdir(path)):
                    if fn.endswith('.json'):
                        yield folder, fn, (lambda p=os.path.join(path, fn): json.load(open(p, encoding='utf-8')))
    else:
        z = zipfile.ZipFile(args.zip)
        for name in sorted(z.namelist()):
            parts = name.split('/')
            if len(parts) == 3 and parts[2].endswith('.json'):
                yield parts[1], parts[2], (lambda n=name: json.loads(z.read(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=ZIP)
    ap.add_argument('--sample-dir', default=None)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ccma_keys = set()
    if os.path.exists(CCMA_SESSIONS):
        c = pd.read_csv(CCMA_SESSIONS)
        ccma_keys = set(c.session_ts.astype(int))
    t0 = time.time()
    sessions, rows = [], []
    A = collections.defaultdict(list); stats = collections.Counter(); labels_seen = collections.defaultdict(collections.Counter)
    for folder, fn, load in iter_captures(args):
        m = NAME_RE.match(fn)
        if m is None:
            stats[('_', 'bad_name')] += 1; continue
        ts, pkg_in_name = int(m.group(1)), m.group(3)
        if pkg_in_name in PKG_TO_APP:
            app = PKG_TO_APP[pkg_in_name]                       # a capture filed under another app's folder
            if app != folder: stats[(app, 'refiled_in')] += 1
        elif pkg_in_name in CCMA_PACKAGES or folder not in APPS:
            stats[(folder if folder in APPS else '_', 'skipped_ccma_app')] += 1; continue
        else:
            app = folder                                         # name carries 'None' or a system package
        if ts in ccma_keys:
            stats[(app, 'skipped_in_ccma')] += 1; continue      # never reuse a CCMA capture
        d = load(); sid = len(sessions); devs = collections.Counter(); acts = collections.Counter(); kept = 0
        stats[(app, 'files')] += 1
        for key, b in d.items():
            md = b['flow_metadata']; pdt = b['packet_data']
            stats[(app, 'total')] += 1
            label = md.get('BF_label'); labels_seen[app][str(label)] += 1
            if norm(label) not in ALIASES[app] and label != APPS[app][0]:
                stats[(app, 'other_label')] += 1; continue
            lp = pdt['L4_payload_bytes']; raw = pdt['L4_raw_payload']; dirs = pdt['packet_dir']
            idx_pay = [i for i, l in enumerate(lp) if l > 0 and i < len(raw) and len(raw[i]) > 0]
            if not idx_pay:
                stats[(app, 'no_payload')] += 1; continue
            pk = np.zeros((K_PKT, PKT_BYTES), np.uint8); pl = np.zeros(K_PKT, np.int32); pdr = np.full(K_PKT, -1, np.int8); psn = np.full((K_PKT, 2), -1, np.int32)
            for j, i in enumerate(idx_pay[:K_PKT]):
                b_ = bytes(raw[i]); n = min(len(b_), PKT_BYTES); pk[j, :n] = np.frombuffer(b_[:n], np.uint8); pl[j] = len(b_); pdr[j] = dirs[i]
                sp = find_sni_span(b_)
                if sp is not None and sp[0] < PKT_BYTES:
                    psn[j] = (sp[0], min(sp[1], PKT_BYTES)); stats[(app, 'sni')] += 1
            fl = np.zeros(FLAT_BYTES, np.uint8); pos = 0; fsn = np.array([-1, -1], np.int32)
            for i in idx_pay:
                if pos >= FLAT_BYTES: break
                b_ = bytes(raw[i]); n = min(len(b_), FLAT_BYTES - pos); fl[pos:pos + n] = np.frombuffer(b_[:n], np.uint8)
                sp = find_sni_span(b_)
                if sp is not None and fsn[0] < 0 and sp[0] < n: fsn[:] = (pos + sp[0], pos + min(sp[1], n))
                pos += n
            mt = np.zeros((META_LEN, 4), np.float32); nm = min(len(lp), META_LEN)
            mt[:nm, 0] = np.where(np.array(dirs[:nm]) == 0, 1.0, -1.0); mt[:nm, 1] = pdt['IP_packet_bytes'][:nm]; mt[:nm, 2] = lp[:nm]; mt[:nm, 3] = pdt['iat'][:nm]
            A['pkt_bytes'].append(pk); A['pkt_len'].append(pl); A['pkt_dir'].append(pdr); A['pkt_sni'].append(psn)
            A['flat_bytes'].append(fl); A['flat_len'].append(min(pos, FLAT_BYTES)); A['flat_sni'].append(fsn); A['meta'].append(mt); A['meta_len'].append(nm)
            act = str(md.get('BF_activity')); dev = str(md.get('BF_device'))
            rows.append(dict(session_id=sid, app=app, activity=act, device=dev, session_ts=ts, biflow_key=key, proto=key.split(',')[-1],
                             labeling_type=md.get('BF_labeling_type'), n_pkts=len(lp), n_pay_pkts=len(idx_pay), y_app=APP_ID[app]))
            devs[dev] += 1; acts[act] += 1; kept += 1; stats[(app, 'kept')] += 1
        sessions.append(dict(session_id=sid, app=app, folder=folder, file=f'{folder}/{fn}', session_ts=ts,
                             device=devs.most_common(1)[0][0] if devs else 'none', n_kept=kept, activities=','.join(sorted(acts))))
        if sid % 50 == 0: print(app, 'sessions', sid, 'biflows kept', len(rows), f'{time.time()-t0:.0f}s', flush=True)
    idx = pd.DataFrame(rows); ses = pd.DataFrame(sessions)
    dev_ids = {m: i for i, m in enumerate(sorted(idx.device.unique()))}; idx['device_id'] = idx.device.map(dev_ids)
    idx.to_csv(os.path.join(args.out, 'appact_index.csv'), index=False); ses.to_csv(os.path.join(args.out, 'appact_sessions.csv'), index=False)
    y = idx.y_app.values.astype(np.int64)
    np.savez_compressed(os.path.join(args.out, 'appact_biflows.npz'), pkt_bytes=np.stack(A['pkt_bytes']), pkt_len=np.stack(A['pkt_len']), pkt_dir=np.stack(A['pkt_dir']), pkt_sni=np.stack(A['pkt_sni']),
                        flat_bytes=np.stack(A['flat_bytes']), flat_len=np.array(A['flat_len'], np.int32), flat_sni=np.stack(A['flat_sni']), meta=np.stack(A['meta']), meta_len=np.array(A['meta_len'], np.int32),
                        y_app=y, y_act=np.full(len(y), -1, np.int64), y_joint=y, session_id=idx.session_id.values.astype(np.int64), device_id=idx.device_id.values.astype(np.int64))

    def table(frame):
        try: return frame.to_markdown()
        except Exception: return frame.to_string()

    cols = ['files', 'refiled_in', 'skipped_in_ccma', 'skipped_ccma_app', 'total', 'other_label', 'no_payload', 'kept', 'sni']
    L = ['# AppAct-2024 (apps not in CCMA) build report', f'sessions={len(ses)} sessions_with_kept_biflows={(ses.n_kept > 0).sum()} biflows={len(idx)} devices={dev_ids} elapsed={time.time()-t0:.0f}s',
         f'app ids: {APP_ID}', '', '| app | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    for a in list(APPS) + ['_']: L.append(f'| {a} | ' + ' | '.join(str(stats[(a, c)]) for c in cols) + ' |')
    L += ['', '## BF_label values seen per app (top 6) - check that the app name was matched']
    for a in APPS: L.append(f'- {a}: {dict(labels_seen[a].most_common(6))}')
    L += ['', '## sessions with kept biflows: app x device', table(idx.groupby(['app', 'device']).session_id.nunique().unstack(fill_value=0)),
          '', '## kept biflows: app x activity', table(idx.groupby(['app', 'activity']).size().unstack(fill_value=0)),
          '', '## capture year of sessions per app', table(ses.assign(year=pd.to_datetime(ses.session_ts, unit='s').dt.year).groupby(['app', 'year']).size().unstack(fill_value=0))]
    open(os.path.join(args.out, 'appact_build_report.md'), 'w', encoding='utf-8').write('\n'.join(L)); print('\n'.join(L))


if __name__ == '__main__':
    main()
