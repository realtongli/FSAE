"""Build derived per-biflow arrays from MIRAGE-GenAI-2025 (raw JSON is read-only).

Sample unit = biflow labelled (by MIRAGE's netstat-based BF_label) with the package of the
GenAI app that the capture session targeted, and having >=1 packet with L4 payload.

Outputs (data/derived/):
  genai_biflows.npz   arrays described below
  genai_index.csv     one row per kept biflow (session, device, labels, sizes)
  build_report.md     counts + SNI parsing statistics

Arrays:
  pkt_bytes   uint8 [N, K_PKT, PKT_BYTES]  payload of first K_PKT payload-bearing packets (zero padded)
  pkt_len     int32 [N, K_PKT]             true payload length of those packets (0 = absent)
  pkt_dir     int8  [N, K_PKT]             direction of those packets (0=up,1=down), -1 absent
  pkt_sni     int32 [N, K_PKT, 2]          [start,end) of SNI extension inside the PKT_BYTES window, -1 none
  flat_bytes  uint8 [N, FLAT_BYTES]        payload bytes concatenated in packet order (first FLAT_BYTES)
  flat_len    int32 [N]
  flat_sni    int32 [N, 2]                 SNI span inside flat window, -1 none
  meta        float32 [N, META_LEN, 4]     per packet: dir(+1 up / -1 down), IP_packet_bytes, L4_payload_bytes, iat
  meta_len    int32 [N]
  y_app, y_act, y_joint  int64 [N]         app 0=Chatgpt 1=Copilot 2=Gemini ; act 0=text 1=multi (-1 controlled) ; joint=app*2+act
  session_id  int64 [N]  index into sessions list in genai_index.csv ; device_id int64 [N]
"""
import json, glob, os, time, collections
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'data', 'mirage2025genai')
OUT = os.path.join(ROOT, 'data', 'derived')
os.makedirs(OUT, exist_ok=True)

TARGET_PKG = {
    'Chatgpt': {'com.openai.chatgpt'},
    'Copilot': {'com.microsoft.copilot'},
    'Gemini': {'com.google.android.googlequicksearchbox'},  # Gemini is hosted by the Google app on Android
}
APP_ID = {'Chatgpt': 0, 'Copilot': 1, 'Gemini': 2}
ACT_ID = {'text': 0, 'multi': 1, 'controlled': -1}
K_PKT, PKT_BYTES, FLAT_BYTES, META_LEN = 5, 320, 1024, 64


def find_sni_span(payload):
    """Return [start, end) byte span of the server_name extension (incl. 4-byte ext header)
    inside a TLS ClientHello record, or None. Handles only a ClientHello starting at byte 0."""
    try:
        if len(payload) < 9 or payload[0] != 0x16 or payload[5] != 0x01:
            return None
        p = 9 + 2 + 32
        p += 1 + payload[p]                                   # session id
        p += 2 + int.from_bytes(payload[p:p + 2], 'big')      # cipher suites
        p += 1 + payload[p]                                   # compression
        ext_total = int.from_bytes(payload[p:p + 2], 'big'); p += 2
        end = min(len(payload), p + ext_total)
        while p + 4 <= end:
            et = int.from_bytes(payload[p:p + 2], 'big'); el = int.from_bytes(payload[p + 2:p + 4], 'big')
            if et == 0:
                return (p, min(p + 4 + el, len(payload)))
            p += 4 + el
    except Exception:
        return None
    return None


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(RAW, '*', '*', '*.json')))
    sessions = []  # list of dict
    rows = []
    A_pkt, A_len, A_dir, A_sni, A_flat, A_flen, A_fsni, A_meta, A_mlen = [], [], [], [], [], [], [], [], []
    stats = collections.Counter()
    for f in files:
        rel = os.path.relpath(f, RAW).replace('\\', '/')
        part, cls_dir, fname = rel.split('/')
        if part == 'generic':
            app_dir, act = cls_dir.split('_')
        else:
            app_dir, act = cls_dir, 'controlled'
        if app_dir not in TARGET_PKG:
            continue  # Telegram / Whatsapp controlled sessions are not GenAI apps
        ts, mac = fname.split('_')[0], fname.split('_')[1]
        sid = len(sessions)
        sessions.append(dict(session_id=sid, part=part, class_dir=cls_dir, app_dir=app_dir, activity=act,
                             file=rel, session_ts=int(ts), device=mac))
        with open(f) as fh:
            d = json.load(fh)
        for key, b in d.items():
            md = b['flow_metadata']; pdt = b['packet_data']
            stats[(cls_dir, 'biflows_total')] += 1
            if md['BF_label'] not in TARGET_PKG[app_dir]:
                stats[(cls_dir, 'dropped_other_pkg')] += 1
                continue
            lp = pdt['L4_payload_bytes']; raw = pdt['L4_raw_payload']; dirs = pdt['packet_dir']
            idx_pay = [i for i, l in enumerate(lp) if l > 0 and i < len(raw) and len(raw[i]) > 0]
            if not idx_pay:
                stats[(cls_dir, 'dropped_no_payload')] += 1
                continue
            stats[(cls_dir, 'kept')] += 1
            stats[(cls_dir, 'kept_' + md['BF_labeling_type'])] += 1
            # first K payload packets
            pk = np.zeros((K_PKT, PKT_BYTES), np.uint8); pl = np.zeros(K_PKT, np.int32)
            pdr = np.full(K_PKT, -1, np.int8); psn = np.full((K_PKT, 2), -1, np.int32)
            for j, i in enumerate(idx_pay[:K_PKT]):
                b_ = bytes(raw[i]); n = min(len(b_), PKT_BYTES)
                pk[j, :n] = np.frombuffer(b_[:n], np.uint8); pl[j] = len(b_); pdr[j] = dirs[i]
                sp = find_sni_span(b_)
                if sp is not None:
                    s, e = sp
                    if s < PKT_BYTES:
                        psn[j] = (s, min(e, PKT_BYTES)); stats[(cls_dir, 'sni_in_pkt_window')] += 1
                    else:
                        stats[(cls_dir, 'sni_beyond_pkt_window')] += 1
                elif b_[:1] == b'\x16' and len(b_) > 5 and b_[5:6] == b'\x01':
                    stats[(cls_dir, 'clienthello_sni_not_found')] += 1
            # flat bytes
            fl = np.zeros(FLAT_BYTES, np.uint8); pos = 0; fsn = np.array([-1, -1], np.int32)
            for i in idx_pay:
                if pos >= FLAT_BYTES:
                    break
                b_ = bytes(raw[i]); n = min(len(b_), FLAT_BYTES - pos)
                fl[pos:pos + n] = np.frombuffer(b_[:n], np.uint8)
                sp = find_sni_span(b_)
                if sp is not None and fsn[0] < 0 and sp[0] < n:
                    fsn[:] = (pos + sp[0], pos + min(sp[1], n))
                pos += n
            # metadata sequence (all packets incl. zero-payload ones)
            m = np.zeros((META_LEN, 4), np.float32); nm = min(len(lp), META_LEN)
            m[:nm, 0] = np.where(np.array(dirs[:nm]) == 0, 1.0, -1.0)
            m[:nm, 1] = pdt['IP_packet_bytes'][:nm]; m[:nm, 2] = lp[:nm]; m[:nm, 3] = pdt['iat'][:nm]
            A_pkt.append(pk); A_len.append(pl); A_dir.append(pdr); A_sni.append(psn)
            A_flat.append(fl); A_flen.append(min(pos, FLAT_BYTES)); A_fsni.append(fsn); A_meta.append(m); A_mlen.append(nm)
            rows.append(dict(session_id=sid, part=part, class_dir=cls_dir, app_dir=app_dir, activity=act, device=mac,
                             session_ts=int(ts), biflow_key=key, proto=key.split(',')[-1], bf_label=md['BF_label'],
                             labeling_type=md['BF_labeling_type'], n_pkts=len(lp), n_pay_pkts=len(idx_pay),
                             bf_payload_bytes=md['BF_L4_payload_bytes'], bf_duration=md['BF_duration'],
                             y_app=APP_ID[app_dir], y_act=ACT_ID[act],
                             y_joint=(APP_ID[app_dir] * 2 + ACT_ID[act]) if ACT_ID[act] >= 0 else -1,
                             has_sni_pkt=int((psn[:, 0] >= 0).any()), has_sni_flat=int(fsn[0] >= 0)))
        print(f'{rel}: kept {stats[(cls_dir, "kept")]} so far, {time.time()-t0:.0f}s', flush=True)

    idx = pd.DataFrame(rows)
    dev_ids = {m: i for i, m in enumerate(sorted(idx.device.unique()))}
    idx['device_id'] = idx.device.map(dev_ids)
    idx.to_csv(os.path.join(OUT, 'genai_index.csv'), index=False)
    pd.DataFrame(sessions).to_csv(os.path.join(OUT, 'genai_sessions.csv'), index=False)
    np.savez_compressed(os.path.join(OUT, 'genai_biflows.npz'),
                        pkt_bytes=np.stack(A_pkt), pkt_len=np.stack(A_len), pkt_dir=np.stack(A_dir), pkt_sni=np.stack(A_sni),
                        flat_bytes=np.stack(A_flat), flat_len=np.array(A_flen, np.int32), flat_sni=np.stack(A_fsni),
                        meta=np.stack(A_meta), meta_len=np.array(A_mlen, np.int32),
                        y_app=idx.y_app.values.astype(np.int64), y_act=idx.y_act.values.astype(np.int64),
                        y_joint=idx.y_joint.values.astype(np.int64), session_id=idx.session_id.values.astype(np.int64),
                        device_id=idx.device_id.values.astype(np.int64))
    lines = ['# build_dataset report', '', f'files={len(files)} sessions_kept={len(sessions)} biflows_kept={len(idx)}', '',
             '| class_dir | ' + ' | '.join(['biflows_total', 'dropped_other_pkg', 'dropped_no_payload', 'kept', 'kept_exact', 'kept_most-common', 'sni_in_pkt_window', 'sni_beyond_pkt_window', 'clienthello_sni_not_found']) + ' |',
             '|---|' + '---|' * 9]
    for cd in sorted({k[0] for k in stats}):
        lines.append(f'| {cd} | ' + ' | '.join(str(stats[(cd, c)]) for c in ['biflows_total', 'dropped_other_pkg', 'dropped_no_payload', 'kept', 'kept_exact', 'kept_most-common', 'sni_in_pkt_window', 'sni_beyond_pkt_window', 'clienthello_sni_not_found']) + ' |')
    lines += ['', '## kept biflows per class_dir x device', idx.groupby(['class_dir', 'device']).size().unstack(fill_value=0).to_markdown(),
              '', '## sessions per class_dir x device', idx.groupby(['class_dir', 'device']).session_id.nunique().unstack(fill_value=0).to_markdown(),
              '', '## proto per class_dir', idx.groupby(['class_dir', 'proto']).size().unstack(fill_value=0).to_markdown(),
              '', '## n_pay_pkts quantiles', idx.n_pay_pkts.describe(percentiles=[.1, .25, .5, .75, .9]).to_frame().to_markdown(),
              '', f'device ids: {dev_ids}', f'elapsed {time.time()-t0:.0f}s']
    with open(os.path.join(OUT, 'build_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
