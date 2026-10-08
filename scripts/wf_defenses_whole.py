"""Whole-connection versions of the two published defences of scripts/wf_defenses.py, applied to the complete packet trace
of a connection (every packet MIRAGE records for the biflow, not only the first 64), plus the whole-connection quantities
an observer of flow records sees. Nothing here reads labels.

Source of the whole-connection quantities. Every MIRAGE JSON record (one biflow) holds `packet_data` with one entry per
packet of the connection (`timestamp`, `packet_dir` 0 = from the phone / 1 = to the phone, `IP_packet_bytes`,
`L4_payload_bytes`, `iat`) and `flow_metadata` with the flow totals `BF_num_packets`, `BF_IP_packet_bytes`,
`BF_L4_payload_bytes`, `BF_duration` and their per-direction counterparts `UF_*` (upstream) and `DF_*` (downstream). The
quantities are computed from `packet_data` (so that the defences can be simulated on the same packets) and checked against
the `flow_metadata` totals (extract() records the largest disagreement).

Undefended observables (raw_whole): per direction the number of packets, the IP bytes and the L4 payload bytes, the
connection duration (last minus first packet) and the per-direction duration (last minus first packet of that direction).

TAMARAW over the whole connection (tamaraw_whole; tamaraw_prepad and tamaraw_pad). Every packet has one fixed size, each
direction d sends one packet every rho_d seconds, each direction is padded to a multiple of L packets, and one real packet
occupies one slot. Two configurations are defined:
  a double-rate configuration (1452-B packets every 20 ms up / 5 ms down, L = 20): 1452 IP bytes with 1400 B of payload;
    constants PKT_IP, PKT_PAY, RHO_UP, RHO_DN, LMUL, the default arguments of the functions below;
  the published main configuration of Tamaraw (1500-B packets every 40 ms up / 12 ms down, L = 100): Cai et al., CCS 2014,
    Sec. 6.2, MTU packets of 1500 IP bytes with 1448 B of payload; constants PUB_PKT_IP, PUB_PKT_PAY, PUB_RHO_UP,
    PUB_RHO_DN, PUB_L, equal to the defaults of wf_defenses.tamaraw.
tamaraw_whole simulates the double-rate configuration (its packet size is PKT_IP / PKT_PAY); tamaraw_prepad (the part that
does not depend on L) and tamaraw_pad (the padding) do the same arithmetic with the rates, L and the packet size as
arguments, for either configuration. Simulated causally: slot k of direction d is at k*rho_d (k >= 1, the first slot one
period after the connection's first packet); a real packet leaves in the first free slot at or after its arrival, so
s_j = max(s_{j-1} + 1, ceil(t_j / rho_d)). Constant-rate transmission runs in both directions until the last real packet
of the connection has left (T_real = max_d s_last,d * rho_d), as in Cai et al.'s design and Wang's reference simulation;
then each direction is padded to the next multiple of L at its own rate. Observables: padded packets per direction N_d
(and their bytes, N_d * packet IP bytes and N_d * packet payload bytes), the defended duration max_d N_d * rho_d and the
per-direction spans (N_d - 1) * rho_d. Costs: extra bytes = sum_d N_d * packet IP bytes - real IP bytes; queueing delay of
every real packet s_j * rho_d - t_j; added duration = defended duration - original duration.

FRONT over the whole connection (front_whole). The parameters of wf_defenses.front: per direction a dummy budget
k ~ U{1..min(N, n)} with N = 700 upstream and 1400 downstream and n the connection's real packet count, a window
w ~ U(1, 14) s scaled to the connection's duration (w * dur / 14), dummy times ~ Rayleigh(scaled window) clipped to
[0, dur], and each dummy copies the IP and payload size of a uniformly drawn real packet of the same direction (of any
direction if that one has none). Real packets are not delayed. Observables: per direction packets, IP bytes and payload
bytes including dummies, the connection duration and the per-direction spans including dummies. Cost: dummy IP bytes."""
import io, json, os, zipfile
import numpy as np
import pandas as pd

# A double-rate configuration (1452-B packets every 20 ms up / 5 ms down, L = 20); the default arguments of the functions
# below, which take the rates and packet sizes as arguments.
PKT_IP, PKT_PAY = 1452.0, 1400.0          # = wf_defenses.DUMMY_IP / DUMMY_PAY
RHO_UP, RHO_DN, LMUL = 0.020, 0.005, 20
# The published main configuration of Tamaraw (1500-B packets every 40 ms up / 12 ms down, L = 100): Cai et al., CCS 2014,
# Sec. 6.2, = wf_defenses.tamaraw defaults; MTU packets (1500 IP / 1448 payload bytes, see wf_defenses.TAM_PKT_IP).
PUB_PKT_IP, PUB_PKT_PAY = 1500.0, 1448.0
PUB_RHO_UP, PUB_RHO_DN, PUB_L = 0.040, 0.012, 100
FRONT_NC, FRONT_NS, FRONT_WMIN, FRONT_WMAX = 700, 1400, 1.0, 14.0   # wf_defenses.front defaults

# column order of the whole-connection observable vector W (same meaning for undefended and defended traffic)
W_COLS = ['pkts_up', 'pkts_dn', 'ipbytes_up', 'ipbytes_dn', 'paybytes_up', 'paybytes_dn', 'duration', 'span_up', 'span_dn']


def _span(t):
    return float(t[-1] - t[0]) if len(t) else 0.0


def raw_whole(t, d, ip, pay):
    up, dn = d > 0, d < 0
    return np.array([up.sum(), dn.sum(), ip[up].sum(), ip[dn].sum(), pay[up].sum(), pay[dn].sum(),
                     float(t[-1] - t[0]), _span(t[up]), _span(t[dn])], np.float64)


def _slots(td, rho):
    """Causal constant-rate queue: slot index of every real packet (arrival times td, ascending, relative to the
    connection's first packet). s_j = max(s_{j-1} + 1, ceil(td_j / rho)), s_{-1} = 0  =>  s_j = j + max(1, cummax_i<=j(a_i - i))."""
    if len(td) == 0: return np.zeros(0, np.int64)
    a = np.ceil(td / rho - 1e-9).astype(np.int64); j = np.arange(len(td), dtype=np.int64)
    return j + np.maximum(1, np.maximum.accumulate(a - j))


def tamaraw_whole(t, d, ip, rho_up=RHO_UP, rho_dn=RHO_DN, L=LMUL):
    t = t - t[0]; out = {}; s_last = {}; delay_sum = 0.0; delay_n = 0
    for key, sgn, rho in (('up', 1, rho_up), ('dn', -1, rho_dn)):
        td = t[d == sgn]; s = _slots(td, rho)
        s_last[key] = int(s[-1]) if len(s) else 0
        if len(s): delay_sum += float((s * rho - td).sum()); delay_n += len(s)
    T_real = max(s_last['up'] * rho_up, s_last['dn'] * rho_dn)
    N = {}
    for key, rho in (('up', rho_up), ('dn', rho_dn)):
        c = max(s_last[key], int(np.floor(T_real / rho + 1e-9)))
        N[key] = int(np.ceil(max(c, 1) / L) * L)
    dur_def = max(N['up'] * rho_up, N['dn'] * rho_dn)
    W = np.array([N['up'], N['dn'], N['up'] * PKT_IP, N['dn'] * PKT_IP, N['up'] * PKT_PAY, N['dn'] * PKT_PAY,
                  dur_def, (N['up'] - 1) * rho_up, (N['dn'] - 1) * rho_dn], np.float64)
    real_ip = float(ip.sum())
    out.update(W=W, extra_bytes=float((N['up'] + N['dn']) * PKT_IP - real_ip), real_bytes=real_ip, delay_sum=delay_sum,
               delay_n=delay_n, dur_orig=float(t[-1]), dur_def=float(dur_def))
    return out


def tamaraw_prepad(t, d, rho_up=RHO_UP, rho_dn=RHO_DN):
    """The part of tamaraw_whole that does not depend on the padding multiple L (same code, same arithmetic): the slot
    count c_d of each direction when constant-rate transmission stops (the connection's last real packet has left,
    T_real = max_d s_last,d * rho_d; c_d = max(s_last,d, floor(T_real / rho_d))) and the queueing delay of the real
    packets. Padding to a multiple of L starts only after the last real packet has left, so the queueing delay does not
    depend on L. tamaraw_pad(c_up, c_dn, L) then gives tamaraw_whole's observables for any L."""
    t = t - t[0]; s_last = {}; delay_sum = 0.0; delay_n = 0
    for key, sgn, rho in (('up', 1, rho_up), ('dn', -1, rho_dn)):
        td = t[d == sgn]; s = _slots(td, rho)
        s_last[key] = int(s[-1]) if len(s) else 0
        if len(s): delay_sum += float((s * rho - td).sum()); delay_n += len(s)
    T_real = max(s_last['up'] * rho_up, s_last['dn'] * rho_dn)
    c = {key: max(s_last[key], int(np.floor(T_real / rho + 1e-9))) for key, rho in (('up', rho_up), ('dn', rho_dn))}
    return c['up'], c['dn'], delay_sum, delay_n


def tamaraw_pad(c_up, c_dn, L, rho_up=RHO_UP, rho_dn=RHO_DN, pkt_ip=PKT_IP, pkt_pay=PKT_PAY):
    """Vectorised over flows: tamaraw_whole's observable vector W (W_COLS) and defended duration for padding multiple L,
    from the pre-padding slot counts of tamaraw_prepad. Each direction is padded to the next multiple of L,
    N_d = ceil(max(c_d, 1) / L) * L (Cai et al., Sec. 6.1: 'AL < I <= (A+1)L, then we pad to (A+1)L'), exactly as in
    tamaraw_whole (integer arithmetic here; equal to tamaraw_whole's float ceil for these magnitudes, checked by the caller).
    pkt_ip / pkt_pay: the fixed packet size of the configuration (default: the double-rate configuration)."""
    cu = np.maximum(np.asarray(c_up, np.int64), 1); cd = np.maximum(np.asarray(c_dn, np.int64), 1)
    Nu = ((cu + L - 1) // L) * L; Nd = ((cd + L - 1) // L) * L
    dur_def = np.maximum(Nu * rho_up, Nd * rho_dn)
    W = np.stack([Nu, Nd, Nu * pkt_ip, Nd * pkt_ip, Nu * pkt_pay, Nd * pkt_pay, dur_def, (Nu - 1) * rho_up, (Nd - 1) * rho_dn], 1).astype(np.float64)
    return W, dur_def.astype(np.float64)


def extract_tamaraw_rates(root, prefix, rates, log=print):
    """As extract_tamaraw_grid, for several rate configurations in ONE pass over the raw JSON: for each
    name -> (rho_up, rho_dn) of `rates`, the L-independent quantities of tamaraw_prepad under '<name>_c_up', '<name>_c_dn',
    '<name>_delay_sum', '<name>_delay_n'; plus the capture edges start_off, end_gap, cap_len, got, sorted_ts, computed
    exactly as in extract_tamaraw_grid. Nothing here reads labels."""
    index = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_index.csv'))
    sessions = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_sessions.csv'))
    N = len(index)
    G = dict(start_off=np.zeros(N), end_gap=np.zeros(N), cap_len=np.zeros(N), got=np.zeros(N, bool), sorted_ts=np.zeros(N, bool))
    for name in rates:
        G.update({f'{name}_c_up': np.zeros(N, np.int64), f'{name}_c_dn': np.zeros(N, np.int64),
                  f'{name}_delay_sum': np.zeros(N), f'{name}_delay_n': np.zeros(N)})
    rows_by_sid = {sid: g for sid, g in index.reset_index().groupby('session_id')}
    for c, (sid, js) in enumerate(_iter_sessions(root, prefix, sessions)):
        if sid not in rows_by_sid: continue
        firsts = [float(min(b['packet_data']['timestamp'])) for b in js.values() if len(b['packet_data']['timestamp'])]
        lasts = [float(max(b['packet_data']['timestamp'])) for b in js.values() if len(b['packet_data']['timestamp'])]
        t0, t1 = min(firsts), max(lasts)
        for row, key in zip(rows_by_sid[sid]['index'].values, rows_by_sid[sid]['biflow_key'].values):
            ts, d, ip, pay, iat = _flow_arrays(js[key])
            for name, (ru, rd) in rates.items():
                cu, cd, dsum, dn = tamaraw_prepad(ts - ts[0], d, ru, rd)
                G[f'{name}_c_up'][row], G[f'{name}_c_dn'][row], G[f'{name}_delay_sum'][row], G[f'{name}_delay_n'][row] = cu, cd, dsum, dn
            G['start_off'][row] = float(ts[0] - t0); G['end_gap'][row] = float(t1 - ts[-1]); G['cap_len'][row] = t1 - t0
            G['sorted_ts'][row] = bool(np.all(np.diff(ts) >= 0)); G['got'][row] = True
        if c % 25 == 0: log(f'{prefix} (Tamaraw rates {list(rates)}): session {c + 1}/{len(sessions)}, flows done {int(G["got"].sum())}/{N}')
        del js
    return G


def extract_tamaraw_grid(root, prefix, log=print):
    """Per row of data/derived/<prefix>_index.csv, the L-independent Tamaraw quantities (tamaraw_prepad: c_up, c_dn,
    delay_sum, delay_n) and the capture edges: start_off = connection's first packet minus the capture's first packet,
    end_gap = the capture's last packet minus the connection's last packet, cap_len = capture length (capture = the
    session JSON; its first/last packet over every biflow it holds, labelled or not). Nothing here reads labels."""
    index = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_index.csv'))
    sessions = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_sessions.csv'))
    N = len(index)
    G = dict(c_up=np.zeros(N, np.int64), c_dn=np.zeros(N, np.int64), delay_sum=np.zeros(N), delay_n=np.zeros(N),
             start_off=np.zeros(N), end_gap=np.zeros(N), cap_len=np.zeros(N), got=np.zeros(N, bool), sorted_ts=np.zeros(N, bool))
    rows_by_sid = {sid: g for sid, g in index.reset_index().groupby('session_id')}
    for c, (sid, js) in enumerate(_iter_sessions(root, prefix, sessions)):
        if sid not in rows_by_sid: continue
        firsts = [float(min(b['packet_data']['timestamp'])) for b in js.values() if len(b['packet_data']['timestamp'])]
        lasts = [float(max(b['packet_data']['timestamp'])) for b in js.values() if len(b['packet_data']['timestamp'])]
        t0, t1 = min(firsts), max(lasts)
        for row, key in zip(rows_by_sid[sid]['index'].values, rows_by_sid[sid]['biflow_key'].values):
            ts, d, ip, pay, iat = _flow_arrays(js[key])
            cu, cd, dsum, dn = tamaraw_prepad(ts - ts[0], d)
            G['c_up'][row], G['c_dn'][row], G['delay_sum'][row], G['delay_n'][row] = cu, cd, dsum, dn
            G['start_off'][row] = float(ts[0] - t0); G['end_gap'][row] = float(t1 - ts[-1]); G['cap_len'][row] = t1 - t0
            G['sorted_ts'][row] = bool(np.all(np.diff(ts) >= 0)); G['got'][row] = True
        if c % 25 == 0: log(f'{prefix} (Tamaraw L grid): session {c + 1}/{len(sessions)}, flows done {int(G["got"].sum())}/{N}')
        del js
    return G


def front_whole(t, d, ip, pay, rng, N_c=FRONT_NC, N_s=FRONT_NS, W_min=FRONT_WMIN, W_max=FRONT_WMAX):
    t = t - t[0]; n = len(t); dur = max(float(t[-1]), 1e-3)
    W = raw_whole(t, d, ip, pay); dummy_ip = 0.0; dummy_n = 0; t_max = float(t[-1])
    for col, sgn, budget in ((0, 1, N_c), (1, -1, N_s)):
        cap = max(1, int(round(min(budget, n))))
        k = rng.randint(1, cap + 1)
        w = rng.uniform(W_min, W_max); w_eff = w / W_max * dur
        ts = np.clip(rng.rayleigh(max(w_eff, 1e-4), k), 0, dur)
        same = np.where(d == sgn)[0]; pool = same if len(same) else np.arange(n)
        pick = pool[rng.randint(len(pool), size=k)]
        W[col] += k; W[2 + col] += ip[pick].sum(); W[4 + col] += pay[pick].sum()
        tt = np.concatenate([t[same], ts]); W[7 + col] = float(tt.max() - tt.min())
        dummy_ip += float(ip[pick].sum()); dummy_n += k; t_max = max(t_max, float(ts.max()))
    W[6] = t_max   # dummies are clipped to [0, max(dur, 1 ms)], so the duration changes only for connections under 1 ms
    return dict(W=W, dummy_bytes=dummy_ip, dummy_pkts=dummy_n, real_bytes=float(ip.sum()), real_pkts=n)


def _flow_arrays(rec):
    p = rec['packet_data']
    ts = np.asarray(p['timestamp'], np.float64)
    d = np.where(np.asarray(p['packet_dir']) == 0, 1, -1).astype(np.int8)
    ip = np.asarray(p['IP_packet_bytes'], np.float64); pay = np.asarray(p['L4_payload_bytes'], np.float64)
    iat = np.asarray(p['iat'], np.float64)
    return ts, d, ip, pay, iat


def _iter_sessions(root, prefix, sessions):
    """Yields (session_id, parsed JSON dict) in the order of the sessions table."""
    if prefix == 'genai':
        raw = os.path.join(root, 'data', 'mirage2025genai')
        for sid, f in zip(sessions.session_id, sessions.file):
            with open(os.path.join(raw, f), encoding='utf-8') as fh: yield int(sid), json.load(fh)
    else:
        z = zipfile.ZipFile(os.path.join(root, 'data', 'MIRAGE-COVID-CCMA-2022.zip'))
        for app in sessions.app.unique():
            inner = zipfile.ZipFile(io.BytesIO(z.read(f'MIRAGE-COVID-CCMA-2022/Raw_JSON/{app}.zip')))
            sub = sessions[sessions.app == app]
            for sid, f in zip(sub.session_id, sub.file):
                yield int(sid), json.loads(inner.read(f))
            del inner


def extract(root, prefix, meta, meta_len, seeds=(0, 1, 2, 3, 4), log=print):
    """Per row of data/derived/<prefix>_index.csv: undefended observables W_raw, whole-connection Tamaraw observables and
    costs, whole-connection FRONT observables and costs per seed, the start of the connection relative to the first
    packet of its capture session (any biflow of the capture), plus consistency checks against flow_metadata and against
    the derived 64-packet array meta. FRONT's random stream is seeded per (seed, row), so the result does not depend on
    the order in which flows are processed."""
    index = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_index.csv'))
    sessions = pd.read_csv(os.path.join(root, 'data', 'derived', f'{prefix}_sessions.csv'))
    N = len(index); S = len(seeds)
    R = dict(W_raw=np.zeros((N, 9)), W_tam=np.zeros((N, 9)), W_front=np.zeros((S, N, 9)), start_off=np.zeros(N),
             tam_extra=np.zeros(N), tam_delay_sum=np.zeros(N), tam_delay_n=np.zeros(N), dur_orig=np.zeros(N), dur_def=np.zeros(N),
             real_bytes=np.zeros(N), real_pkts=np.zeros(N), front_dummy_bytes=np.zeros((S, N)), front_dummy_pkts=np.zeros((S, N)),
             md_W=np.zeros((N, 9)), md_n=np.zeros(N), got=np.zeros(N, bool), meta_ok=np.zeros(N, bool))
    rows_by_sid = {sid: g for sid, g in index.reset_index().groupby('session_id')}
    for c, (sid, js) in enumerate(_iter_sessions(root, prefix, sessions)):
        if sid not in rows_by_sid: continue
        t_sess = min(float(b['packet_data']['timestamp'][0]) for b in js.values() if len(b['packet_data']['timestamp']))
        for row, key in zip(rows_by_sid[sid]['index'].values, rows_by_sid[sid]['biflow_key'].values):
            rec = js[key]; md = rec['flow_metadata']
            ts, d, ip, pay, iat = _flow_arrays(rec)
            t = ts - ts[0]
            R['W_raw'][row] = raw_whole(t, d, ip, pay)
            R['md_W'][row] = [md['UF_num_packets'], md['DF_num_packets'], md['UF_IP_packet_bytes'], md['DF_IP_packet_bytes'],
                              md['UF_L4_payload_bytes'], md['DF_L4_payload_bytes'], md['BF_duration'], md['UF_duration'], md['DF_duration']]
            R['md_n'][row] = md['BF_num_packets']
            n64 = int(meta_len[row]); m = meta[row, :n64]
            R['meta_ok'][row] = (n64 == min(len(t), 64) and np.array_equal(m[:, 0], d[:n64].astype(np.float32))
                                 and np.allclose(m[:, 1], ip[:n64]) and np.allclose(m[:, 2], pay[:n64]) and np.allclose(m[:, 3], iat[:n64].astype(np.float32)))
            R['start_off'][row] = float(ts[0] - t_sess)
            tw = tamaraw_whole(t, d, ip)
            R['W_tam'][row] = tw['W']; R['tam_extra'][row] = tw['extra_bytes']; R['tam_delay_sum'][row] = tw['delay_sum']
            R['tam_delay_n'][row] = tw['delay_n']; R['dur_orig'][row] = tw['dur_orig']; R['dur_def'][row] = tw['dur_def']
            R['real_bytes'][row] = float(ip.sum()); R['real_pkts'][row] = len(t)
            for si, s in enumerate(seeds):
                rng = np.random.RandomState((1_000_003 * (40_000 + s) + row) % (2 ** 32 - 1))
                fw = front_whole(t, d, ip, pay, rng)
                R['W_front'][si, row] = fw['W']; R['front_dummy_bytes'][si, row] = fw['dummy_bytes']; R['front_dummy_pkts'][si, row] = fw['dummy_pkts']
            R['got'][row] = True
        if c % 25 == 0: log(f'{prefix}: session {c + 1}/{len(sessions)}, flows done {int(R["got"].sum())}/{N}')
        del js
    return R
