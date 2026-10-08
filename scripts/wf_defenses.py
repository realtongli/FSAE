"""Two published traffic-analysis defences, applied to the per-packet metadata view of a biflow.

FRONT   (Gong and Wang, USENIX Security 2020): randomised dummy injection, front-loaded. For each flow and each
        direction, draw a dummy budget n ~ U(1, N) and a window w ~ U(W_min, W_max), draw n dummy timestamps from
        Rayleigh(w) relative to the start of the flow, and insert dummy packets at those times. Real packets are never
        delayed, so the defence costs bandwidth but no latency. Tor cells all have one size, so FRONT's dummies cannot
        be told apart from real cells; on TLS a fixed dummy size would give them away (no real CCMA packet is 1452 B).
        With dummy_size='mimic' (the default) each dummy therefore copies the IP and payload size of a random real
        packet of the same direction in the same flow; with dummy_size='fixed' every dummy is a full-size packet
        (DUMMY_IP IP bytes, DUMMY_PAY payload bytes).
TAMARAW (Cai et al., CCS 2014): constant-rate transmission. Each direction sends one packet of a fixed size every
        rho seconds; real packets queue for the next slot and each direction is padded until its packet count is a
        multiple of L. Timing is fully regularised, so both bandwidth and latency are paid. tamaraw() is a causal
        simulation with the published main configuration of Cai et al. (CCS 2014, Sec. 6.2); see its docstring.

Both take the raw metadata array m [N, T, 4] = (dir +1 up / -1 down, IP bytes, payload bytes, iat seconds) with the
per-flow lengths, and return the shaped array, the new lengths, and the mean bandwidth and latency overheads.
"""
import numpy as np

DUMMY_PAY = 1400.0   # a full-size TLS record of padding
DUMMY_IP = 1452.0

# Tamaraw, published main configuration (Cai, Nithyanand, Wang, Johnson, Goldberg, CCS 2014, Sec. 6.2: 'Here we set L to
# 100 ... With MTU packets, rho_out = 0.04 and rho_in = 0.012'). MTU = 1500 B (their testbed: 'MTU set at 1500'; Fig. 3
# treats 1500-B packets as MTU packets; the largest IP packet in both of our campaigns is 1500 B). In our (IP bytes,
# payload bytes) representation an MTU packet is 1500 IP bytes and 1448 payload bytes (IPv4 20 B + TCP 32 B with the
# timestamp option, the most common header in both campaigns); the payload channel is a constant, so its value carries
# no information, and every cost is counted in IP bytes.
TAM_PKT_IP, TAM_PKT_PAY = 1500.0, 1448.0
TAM_RHO_OUT, TAM_RHO_IN, TAM_L = 0.04, 0.012, 100


def _to_abs(m, n):
    """Per-packet absolute times from the inter-arrival channel."""
    return np.cumsum(m[:n, 3])


def front(m_all, len_all, seed=0, N_c=700, N_s=1400, W_min=1.0, W_max=14.0, T=64, dummy_size='mimic'):
    """Padding budgets follow the paper; they are scaled by the flow's own duration because a biflow here is one
    connection rather than a whole page load, so a budget of hundreds of dummies would dwarf a short flow. We keep the
    paper's Rayleigh schedule and cap the budget at the flow's real packet count."""
    rng = np.random.RandomState(20000 + seed)
    out = np.zeros_like(m_all); lens = np.zeros_like(len_all); bw = []; N = len(m_all)
    for i in range(N):
        n = int(min(len_all[i], m_all.shape[1]))
        if n == 0: continue
        m = m_all[i, :n]; t = _to_abs(m_all[i], n); dur = max(float(t[-1]), 1e-3)
        pkts = [(t[j], m[j, 0], m[j, 1], m[j, 2], 0) for j in range(n)]
        real_bytes = float(m[:, 1].sum())
        for d, budget in ((1.0, N_c), (-1.0, N_s)):
            cap = max(1, int(round(min(budget, n))))
            k = rng.randint(1, cap + 1)
            w = rng.uniform(W_min, W_max); w_eff = w / W_max * dur          # Rayleigh window scaled to this flow
            ts = np.clip(rng.rayleigh(max(w_eff, 1e-4), k), 0, dur)
            same = np.where(m[:, 0] == d)[0]
            pool = same if len(same) else np.arange(n)  # a direction with no real packet borrows the flow's other sizes
            for tt in ts:
                if dummy_size == 'mimic': j = pool[rng.randint(len(pool))]; ipb, pay = float(m[j, 1]), float(m[j, 2])
                else: ipb, pay = DUMMY_IP, DUMMY_PAY
                pkts.append((float(tt), d, ipb, pay, 1))
        pkts.sort(key=lambda p: p[0]); pkts = pkts[:T]
        arr = np.zeros((T, 4), np.float32); prev = 0.0
        for j, (tt, d, ipb, pay, _) in enumerate(pkts):
            arr[j] = (d, ipb, pay, max(tt - prev, 0.0)); prev = tt
        out[i] = arr; lens[i] = len(pkts)
        dummy_bytes = sum(p[2] for p in pkts if p[4] == 1)
        bw.append(dummy_bytes / max(real_bytes, 1.0))
    return out, lens, dict(bandwidth_overhead=float(np.mean(bw)) if bw else 0.0, latency_overhead=0.0)


def tamaraw(m_all, len_all, seed=0, rho_out=TAM_RHO_OUT, rho_in=TAM_RHO_IN, Lmul=TAM_L, T=64, pkt_ip=TAM_PKT_IP, pkt_pay=TAM_PKT_PAY):
    """Tamaraw, causal simulation with the published main configuration of Cai et al. (CCS 2014, Sec. 6.2: MTU packets,
    rho_out = 0.04 s, rho_in = 0.012 s, L = 100; MTU = 1500 IP / 1448 payload bytes, see TAM_PKT_IP) as the default
    arguments. Rules:
      Input. The observed real packets of each flow (the first min(len, T) packets of m, i.e. the first 64 packets of the
        connection); their times t_j = cumulative inter-arrival times, relative to the flow's first packet.
      Causal queue, per direction d (up: rho_out, down: rho_in). Slot k of direction d is at k * rho_d (k >= 1, the first
        slot one period after the flow's first packet). The j-th real packet of d leaves in the first free slot at or after
        its arrival, s_j = max(s_{j-1} + 1, ceil(t_j / rho_d)); one real packet per slot; no packet leaves before it
        arrives. This is wf_defenses_whole._slots, called through wf_defenses_whole.tamaraw_prepad (the same code as the
        whole-connection simulation).
      Stop and padding. Both directions send in every slot, a real packet if one is waiting and a dummy otherwise, until
        the flow's last real packet has left (T_real = max_d s_last,d * rho_d; c_d = max(s_last,d, floor(T_real / rho_d)));
        then each direction is padded at its own rate to N_d = ceil(max(c_d, 1) / L) * L packets (Sec. 6.1: 'if AL < I <=
        (A+1)L, then we pad to (A+1)L'; deterministic; Wang's reference code adds a random geometric number of further
        multiples of L, which is not used). Same arithmetic as wf_defenses_whole.tamaraw_pad.
      Defended trace. The N_up + N_dn packets at their slot times, merged in time order; at equal times the downstream
        packet goes first (Wang's reference sends an outgoing packet only if its slot is strictly earlier). Every packet
        has pkt_ip IP bytes and pkt_pay payload bytes.
      Observation. The first T defended packets: (direction, pkt_ip, pkt_pay, gap to the previous defended packet), the
        first packet's gap 0 (as in the undefended data, whose first inter-arrival time is always 0); length min(N_up +
        N_dn, T). With L = 100 each direction sends at least 100 packets, so the first 64 defended packets are the same
        for every flow; this follows from the rules and is checked in scripts/defence_stats.py (distinct inputs).
      Overheads (returned). Over the flows with at least one packet: bandwidth_overhead = mean of per-flow ratios
        (sum_d N_d * pkt_ip - real IP bytes) / real IP bytes of the simulated (observed) part of the flow, and
        bandwidth_overhead_aggregate = the same as sum over sum; latency_overhead_s = mean over flows of the mean queueing
        delay s_j * rho_d - t_j of its real packets, latency_pooled_s = mean over all real packets; params.
    seed is unused (the defence is deterministic)."""
    import wf_defenses_whole as WW
    out = np.zeros_like(m_all); lens = np.zeros_like(len_all); N = len(m_all)
    ratios, extra_sum, real_sum, flow_delay, dsum_all, dn_all = [], 0.0, 0.0, [], 0.0, 0
    for i in range(N):
        n = int(min(len_all[i], m_all.shape[1]))
        if n == 0: continue
        m = m_all[i, :n].astype(np.float64); t = np.cumsum(m[:, 3]); t = t - t[0]; d = m[:, 0]
        c_up, c_dn, dsum, dn = WW.tamaraw_prepad(t, d, rho_out, rho_in)
        Nu = int(np.ceil(max(c_up, 1) / Lmul) * Lmul); Nd = int(np.ceil(max(c_dn, 1) / Lmul) * Lmul)
        ku = np.arange(1, min(Nu, T) + 1); kd = np.arange(1, min(Nd, T) + 1)
        times = np.concatenate([kd * rho_in, ku * rho_out]); dirs = np.concatenate([-np.ones(len(kd)), np.ones(len(ku))])
        order = np.lexsort((dirs, np.round(times, 9)))[:T]          # time, then downstream (-1) before upstream (+1)
        tt, dd = times[order], dirs[order]; k = len(order)
        out[i, :k, 0] = dd; out[i, :k, 1] = pkt_ip; out[i, :k, 2] = pkt_pay; out[i, :k, 3] = np.diff(tt, prepend=tt[0])
        lens[i] = k
        real = float(m[:, 1].sum()); extra = (Nu + Nd) * pkt_ip - real
        ratios.append(extra / max(real, 1.0)); extra_sum += extra; real_sum += real
        flow_delay.append(dsum / max(dn, 1)); dsum_all += dsum; dn_all += dn
    return out, lens, dict(bandwidth_overhead=float(np.mean(ratios)) if ratios else 0.0,
                           bandwidth_overhead_aggregate=float(extra_sum / max(real_sum, 1.0)),
                           latency_overhead_s=float(np.mean(flow_delay)) if flow_delay else 0.0,
                           latency_pooled_s=float(dsum_all / max(dn_all, 1)),
                           params=dict(rho_out=rho_out, rho_in=rho_in, L=Lmul, pkt_ip=pkt_ip, pkt_pay=pkt_pay, T=T, causal=True))


def ech(m_all, len_all, seed=0, T=64, mss=1388.0, pct=99.0, ref=None):
    """Proxy for Encrypted ClientHello. ECH pads the inner ClientHello so that its length does not reveal the server
    name; what remains visible is a ClientHello of a fixed size. We therefore replace the first upstream flight of every
    flow (the consecutive upstream payload packets before the first downstream payload packet, i.e. the ClientHello and
    any continuation segment) by a flight of one constant total size C, re-segmented at the MSS, so neither its size nor
    its segment count carries information. C is the pct-th percentile of ClientHello sizes in `ref` (default: this
    data; 640 B on GenAI, 1081 B on CCMA); the rare larger ones are rounded up to a multiple of 256 B. Nothing is
    delayed; the cost is the padding bytes."""
    def first_flight(m, n):
        j0 = next((j for j in range(n) if m[j, 2] > 0), None)
        # only TCP connections captured from their handshake: the first payload packet follows the zero-payload
        # SYN/SYN-ACK/ACK and is upstream, so it is the ClientHello. UDP (header 28/48 B: QUIC, whose Initial is already
        # padded to >= 1200 B, and media) and connections captured mid-stream are left unchanged.
        if j0 is None or j0 == 0 or m[j0, 0] != 1.0 or (m[j0, 1] - m[j0, 2]) in (28.0, 48.0): return None
        j1 = j0
        while j1 + 1 < n and not (m[j1 + 1, 0] == -1.0 and m[j1 + 1, 2] > 0):
            j1 += 1
        idx = [j for j in range(j0, j1 + 1) if m[j, 0] == 1.0 and m[j, 2] > 0]
        return j0, j1, idx
    src_m, src_l = (m_all, len_all) if ref is None else ref
    tot = []
    for i in range(len(src_m)):
        n = int(min(src_l[i], src_m.shape[1])); ff = first_flight(src_m[i], n) if n else None
        if ff: tot.append(float(src_m[i, ff[2], 2].sum()))
    C = float(np.percentile(tot, pct)) if tot else mss
    out = np.zeros_like(m_all); lens = len_all.copy(); added = []; N = len(m_all)
    for i in range(N):
        n = int(min(len_all[i], m_all.shape[1])); m = m_all[i, :n]
        ff = first_flight(m, n) if n else None
        if ff is None:
            out[i, :n] = m; continue
        j0, j1, idx = ff; hdr = float(m[j0, 1] - m[j0, 2]); total = float(m[idx, 2].sum())
        new_total = C if total <= C else float(np.ceil(total / 256.0) * 256.0)
        segs = [mss] * int(new_total // mss) + ([new_total % mss] if new_total % mss else [])
        flight = [(1.0, s + hdr, s, float(m[j0, 3]) if k == 0 else 0.0) for k, s in enumerate(segs)]
        others = [tuple(m[j]) for j in range(j0, j1 + 1) if j not in idx]      # ACKs inside the flight stay
        rows = [tuple(m[j]) for j in range(j0)] + flight + others + [tuple(m[j]) for j in range(j1 + 1, n)]
        rows = rows[:T]; out[i, :len(rows)] = np.array(rows, np.float32); lens[i] = len(rows)
        added.append((new_total - total) / max(float(m[:, 1].sum()), 1.0))
    return out, lens, dict(bandwidth_overhead=float(np.mean(added)) if added else 0.0, latency_overhead=0.0, clienthello_bytes=C,
                           share_of_flows=len(added) / max(N, 1))


def ech_full(m_all, len_all, seed=0, T=64, mss=1388.0, pct=99.0):
    """ECH with server-side handshake padding, as RFC 9849 recommends ('clients and servers will also need to pad all other
    handshake messages that have sensitive-length fields'). On top of ech() (the client's first flight padded to a constant
    C_c), the server's first flight (the consecutive downstream payload packets that follow the ClientHello, i.e.
    ServerHello to Finished, up to the next upstream payload packet) is padded to one constant total size C_s, its
    pct-th percentile in this data (the rare larger ones rounded up to a multiple of 256 B), and re-segmented at the MSS.
    The padded segments are sent back to back at the start of the original flight; the client's acknowledgements inside
    the flight are regenerated to match the padded flight (one ACK per two padded segments), so their number does not
    reveal the original flight size, and every later packet keeps its original time. Only TCP connections captured from
    their handshake are changed. The cost is the padding bytes."""
    m1, l1, ov1 = ech(m_all, len_all, seed, T, mss, pct)

    def flights(m, n):  # -> (client flight end j1, server flight [s0, s1], downstream payload indices) or None
        j0 = next((j for j in range(n) if m[j, 2] > 0), None)
        if j0 is None or j0 == 0 or m[j0, 0] != 1.0 or (m[j0, 1] - m[j0, 2]) in (28.0, 48.0): return None
        s0 = next((j for j in range(j0 + 1, n) if m[j, 0] == -1.0 and m[j, 2] > 0), None)
        if s0 is None: return None
        s1 = s0
        while s1 + 1 < n and not (m[s1 + 1, 0] == 1.0 and m[s1 + 1, 2] > 0):
            s1 += 1
        idx = [j for j in range(s0, s1 + 1) if m[j, 0] == -1.0 and m[j, 2] > 0]
        return s0, s1, idx
    tot = []
    for i in range(len(m1)):
        n = int(min(l1[i], m1.shape[1])); f = flights(m1[i], n) if n else None
        if f: tot.append(float(m1[i, f[2], 2].sum()))
    C = float(np.percentile(tot, pct)) if tot else mss
    out = np.zeros_like(m1); lens = l1.copy(); added = []
    for i in range(len(m1)):
        n = int(min(l1[i], m1.shape[1])); m = m1[i, :n]; f = flights(m, n) if n else None
        if f is None:
            out[i, :n] = m; continue
        s0, s1, idx = f; hdr = float(m[s0, 1] - m[s0, 2]); total = float(m[idx, 2].sum())
        new_total = C if total <= C else float(np.ceil(total / 256.0) * 256.0)
        segs = [mss] * int(new_total // mss) + ([new_total % mss] if new_total % mss else [])
        up_ack = [float(m[j, 1]) for j in range(1, n) if m[j, 0] == 1.0 and m[j, 2] == 0]   # after the SYN
        ack_ip = max(set(up_ack), key=up_ack.count) if up_ack else hdr   # the client's usual pure-ACK size in this connection
        flight = []
        for k, s in enumerate(segs):   # back to back from the flight's start; one client ACK per two segments
            flight.append((-1.0, s + hdr, s, float(m[s0, 3]) if k == 0 else 0.0))
            if k % 2 == 1 or k == len(segs) - 1: flight.append((1.0, ack_ip, 0.0, 0.0))
        span = float(m[s0 + 1:s1 + 1, 3].sum())   # the original flight's duration
        rest = [tuple(m[j]) for j in range(s1 + 1, n)]
        if rest: rest[0] = (rest[0][0], rest[0][1], rest[0][2], rest[0][3] + span)   # later packets keep their original times
        rows = [tuple(m[j]) for j in range(s0)] + flight + rest
        rows = rows[:T]; out[i, :len(rows)] = np.array(rows, np.float32); lens[i] = len(rows)
        added.append((new_total - total) / max(float(m[:, 1].sum()), 1.0))
    return out, lens, dict(bandwidth_overhead=float(ov1['bandwidth_overhead']) + (float(np.mean(added)) if added else 0.0), latency_overhead=0.0,
                           clienthello_bytes=ov1['clienthello_bytes'], server_flight_bytes=C, share_of_flows=len(added) / max(len(m1), 1))


def apply_named(kind, m_all, len_all, seed=0):
    if kind == 'front': return front(m_all, len_all, seed)
    if kind == 'tamaraw': return tamaraw(m_all, len_all, seed)            # causal, published configuration
    if kind == 'ech': return ech(m_all, len_all, seed)
    if kind == 'ech_full': return ech_full(m_all, len_all, seed)
    raise ValueError(kind)
