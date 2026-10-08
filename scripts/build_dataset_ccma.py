"""Build derived per-biflow arrays for MIRAGE-COVID-CCMA-2022 (second same-schema dataset), streaming the nested
per-app zips inside data/MIRAGE-COVID-CCMA-2022.zip (raw zip is read-only, nothing is extracted to disk).

Sample unit = biflow labelled (netstat BF_label) with the app's package, BF_activity in {audiocall, chat, videocall},
>= 1 payload packet. Same arrays as build_dataset.py (pkt_bytes 5x320, pkt_sni, flat 1024, meta 64x4), labels:
y_app (9), y_act (3), y_joint = app*3 + act (27), session_id (one JSON capture = one session), device_id (3 phones).
Outputs data/derived/ccma_biflows.npz, ccma_index.csv, ccma_sessions.csv, ccma_build_report.md.
"""
import io, json, os, sys, time, zipfile, collections
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from build_dataset import find_sni_span, K_PKT, PKT_BYTES, FLAT_BYTES, META_LEN

ZIP = os.path.join(ROOT, 'data', 'MIRAGE-COVID-CCMA-2022.zip')
OUT = os.path.join(ROOT, 'data', 'derived')
APPS = {'Discord': 'com.discord', 'GotoMeeting': 'com.gotomeeting', 'Meet': 'com.google.android.apps.meetings',
        'Messenger': 'com.facebook.orca', 'Skype': 'com.skype.raider', 'Slack': 'com.Slack', 'Teams': 'com.microsoft.teams',
        'Webex': 'com.cisco.webex.meetings', 'Zoom': 'us.zoom.videomeetings'}
APP_ID = {a: i for i, a in enumerate(APPS)}
ACT_ID = {'audiocall': 0, 'chat': 1, 'videocall': 2}


def main():
    t0 = time.time(); z = zipfile.ZipFile(ZIP)
    sessions, rows = [], []
    A = collections.defaultdict(list); stats = collections.Counter()
    for app, pkg in APPS.items():
        inner = zipfile.ZipFile(io.BytesIO(z.read(f'MIRAGE-COVID-CCMA-2022/Raw_JSON/{app}.zip')))
        files = sorted(n for n in inner.namelist() if n.endswith('.json'))
        for fi, fn in enumerate(files):
            d = json.loads(inner.read(fn)); sid = len(sessions)
            devs = collections.Counter(); kept = 0
            for key, b in d.items():
                md = b['flow_metadata']; pdt = b['packet_data']
                stats[(app, 'total')] += 1
                if md['BF_label'] != pkg:
                    stats[(app, 'other_pkg')] += 1; continue
                act = md.get('BF_activity')
                if act not in ACT_ID:
                    stats[(app, 'act_unknown')] += 1; continue
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
                m = np.zeros((META_LEN, 4), np.float32); nm = min(len(lp), META_LEN)
                m[:nm, 0] = np.where(np.array(dirs[:nm]) == 0, 1.0, -1.0); m[:nm, 1] = pdt['IP_packet_bytes'][:nm]; m[:nm, 2] = lp[:nm]; m[:nm, 3] = pdt['iat'][:nm]
                A['pkt_bytes'].append(pk); A['pkt_len'].append(pl); A['pkt_dir'].append(pdr); A['pkt_sni'].append(psn)
                A['flat_bytes'].append(fl); A['flat_len'].append(min(pos, FLAT_BYTES)); A['flat_sni'].append(fsn); A['meta'].append(m); A['meta_len'].append(nm)
                rows.append(dict(session_id=sid, app=app, activity=act, device=md['BF_device'], session_ts=int(os.path.basename(fn).split('_')[0]),
                                 biflow_key=key, proto=key.split(',')[-1], labeling_type=md['BF_labeling_type'], n_pkts=len(lp), n_pay_pkts=len(idx_pay),
                                 y_app=APP_ID[app], y_act=ACT_ID[act], y_joint=APP_ID[app] * 3 + ACT_ID[act]))
                devs[md['BF_device']] += 1; kept += 1; stats[(app, 'kept')] += 1
            sessions.append(dict(session_id=sid, app=app, file=fn, session_ts=int(os.path.basename(fn).split('_')[0]),
                                 device=devs.most_common(1)[0][0] if devs else 'none', n_kept=kept,
                                 activities=','.join(sorted(set(r['activity'] for r in rows if r['session_id'] == sid)))))
            if fi % 20 == 0: print(app, fi, '/', len(files), 'kept so far', stats[(app, 'kept')], f'{time.time()-t0:.0f}s', flush=True)
        del inner
    idx = pd.DataFrame(rows); dev_ids = {m: i for i, m in enumerate(sorted(idx.device.unique()))}; idx['device_id'] = idx.device.map(dev_ids)
    idx.to_csv(os.path.join(OUT, 'ccma_index.csv'), index=False); pd.DataFrame(sessions).to_csv(os.path.join(OUT, 'ccma_sessions.csv'), index=False)
    np.savez_compressed(os.path.join(OUT, 'ccma_biflows.npz'), pkt_bytes=np.stack(A['pkt_bytes']), pkt_len=np.stack(A['pkt_len']), pkt_dir=np.stack(A['pkt_dir']), pkt_sni=np.stack(A['pkt_sni']),
                        flat_bytes=np.stack(A['flat_bytes']), flat_len=np.array(A['flat_len'], np.int32), flat_sni=np.stack(A['flat_sni']), meta=np.stack(A['meta']), meta_len=np.array(A['meta_len'], np.int32),
                        y_app=idx.y_app.values.astype(np.int64), y_act=idx.y_act.values.astype(np.int64), y_joint=idx.y_joint.values.astype(np.int64),
                        session_id=idx.session_id.values.astype(np.int64), device_id=idx.device_id.values.astype(np.int64))
    L = ['# CCMA-2022 build report', f'sessions={len(sessions)} biflows={len(idx)} devices={dev_ids} elapsed={time.time()-t0:.0f}s', '',
         '| app | total | other_pkg | act_unknown | no_payload | kept | sni |', '|---|---|---|---|---|---|---|']
    for a in APPS: L.append(f'| {a} | ' + ' | '.join(str(stats[(a, c)]) for c in ['total', 'other_pkg', 'act_unknown', 'no_payload', 'kept', 'sni']) + ' |')
    L += ['', '## kept biflows app x activity', idx.groupby(['app', 'activity']).size().unstack(fill_value=0).to_markdown(), '', '## sessions app x device', idx.groupby(['app', 'device']).session_id.nunique().unstack(fill_value=0).to_markdown()]
    open(os.path.join(OUT, 'ccma_build_report.md'), 'w', encoding='utf-8').write('\n'.join(L)); print('\n'.join(L))


if __name__ == '__main__':
    main()
