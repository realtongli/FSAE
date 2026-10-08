"""Cost and effect of the defences as implemented in scripts/wf_defenses.py, over every labelled flow of each campaign
(the attacker observes the first 64 packets). No model is involved. Writes results/defence_stats.json.
  Tamaraw : the published configuration with a causal queue (wf_defenses.tamaraw): extra bytes within the observed window,
            aggregated over flows (sum extra / sum real), per-flow median, and the mean of per-flow ratios; mean per-packet
            queueing delay; the number of distinct defended inputs. Key 'tamaraw', over every labelled flow, the observed
            real packets being the first min(len, 64) of the flow:
            window     extra_bytes_aggregate / _median / _mean_of_ratios: (64 defended packets x 1500 B - IP bytes of the
                       flow's observed real packets) / those real bytes, as sum over sum, per-flow median, per-flow mean;
            simulated  sim_extra_bytes_aggregate / _median: ((N_up + N_dn) x 1500 B - real IP bytes) / real IP bytes of the
                       simulated flow (its observed real packets sent causally, both directions until the last has left,
                       then each padded to a multiple of L = 100), sum over sum and per-flow median;
            delay      causal queueing delay s_j * rho_d - t_j of every observed real packet: delay_ms = mean over flows of
                       the per-flow mean (the figure wf_defenses.tamaraw returns), delay_ms_pooled = mean over all real
                       packets, delay_ms_median_flow / delay_ms_p90_flow = median and 90th percentile of per-flow means;
                       share_real_sent_before_arrival (must be 0);
            window     distinct_inputs (distinct defended 64-packet arrays) and largest_input_share;
            content    share_real_in_window: share of the observed real packets that have left by the time of the 64th
                       defended packet (they ride inside the observed window), pooled;
            function_overheads (the overheads wf_defenses.tamaraw itself returns, over the simulated flow); params.
  FRONT   : dummies inserted per real packet, dummy bytes over real bytes, growth of the observed window, share of real
            packets pushed out of the first 64 by dummies.
  ECH     : share of flows whose first upstream flight is replaced.
  jitter  : the Exp(20 ms) gap added before every packet accumulates along the flow; mean added delay per packet."""
import collections, json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
import wf_defenses as W
import wf_defenses_whole as WW

T = 64; OUT = {}
for ds in ('genai', 'ccma'):
    d = GenAIData(ROOT, prefix=ds); lab = (d.y_joint if ds == 'genai' else d.y_app) >= 0
    m = d.meta[lab, :T].astype(np.float64); n = np.minimum(d.meta_len[lab], T)
    r = {}
    real = np.array([m[i, :n[i], 1].sum() for i in range(len(m))])
    # Tamaraw, published configuration, causal
    out, ol, ov = W.tamaraw(m, n); P = ov['params']
    sent = ol * P['pkt_ip']; per = (sent - real) / np.maximum(real, 1)
    distinct = collections.Counter(out[i, :ol[i]].tobytes() for i in range(len(m)))
    sim_extra, flow_mean, all_delay, early, in_win, n_real = [], [], [], 0, 0, 0
    for i in range(len(m)):
        k_ = int(n[i]); t = np.cumsum(m[i, :k_, 3]); t = t - t[0]; dd = m[i, :k_, 0]
        cu, cd, _, _ = WW.tamaraw_prepad(t, dd, P['rho_out'], P['rho_in'])
        Nu = int(np.ceil(max(cu, 1) / P['L']) * P['L']); Nd = int(np.ceil(max(cd, 1) / P['L']) * P['L'])
        sim_extra.append((Nu + Nd) * P['pkt_ip'] - real[i])
        t64 = float(np.cumsum(out[i, :ol[i], 3])[-1]) + (P['rho_in'] if out[i, 0, 0] < 0 else P['rho_out'])   # absolute time of the last observed defended packet
        dl = []
        for sgn, rho in ((1.0, P['rho_out']), (-1.0, P['rho_in'])):
            td = t[dd == sgn]; s = WW._slots(td, rho); dep = s * rho
            dl += list(dep - td); early += int((dep < td - 1e-9).sum()); in_win += int((dep <= t64 + 1e-9).sum())
        flow_mean.append(float(np.mean(dl))); all_delay += dl; n_real += k_
    sim_extra = np.array(sim_extra); flow_mean = np.array(flow_mean)
    r['tamaraw'] = dict(extra_bytes_aggregate=float((sent - real).sum() / real.sum()), extra_bytes_median=float(np.median(per)),
                        extra_bytes_mean_of_ratios=float(np.mean(per)),
                        sim_extra_bytes_aggregate=float(sim_extra.sum() / real.sum()), sim_extra_bytes_median=float(np.median(sim_extra / np.maximum(real, 1))),
                        delay_ms=1000 * float(np.mean(flow_mean)), delay_ms_pooled=1000 * float(np.mean(all_delay)),
                        delay_ms_median_flow=1000 * float(np.median(flow_mean)), delay_ms_p90_flow=1000 * float(np.percentile(flow_mean, 90)),
                        share_real_sent_before_arrival=early / n_real,
                        distinct_inputs=len(distinct), largest_input_share=float(distinct.most_common(1)[0][1] / len(m)),
                        share_real_in_window=in_win / n_real,
                        function_overheads=dict(bandwidth_overhead=float(ov['bandwidth_overhead']), bandwidth_overhead_aggregate=float(ov['bandwidth_overhead_aggregate']),
                                                latency_overhead_ms=1000 * float(ov['latency_overhead_s']), latency_pooled_ms=1000 * float(ov['latency_pooled_s'])),
                        params=dict(packet_bytes=P['pkt_ip'], payload_bytes=P['pkt_pay'], rho_up_ms=1000 * P['rho_out'], rho_down_ms=1000 * P['rho_in'],
                                    pad_multiple=P['L'], causal=True, source='Cai et al., CCS 2014, Sec. 6.2 (MTU packets, 0.04/0.012 s, L = 100)'))
    # FRONT: replicates wf_defenses.front's draws, which the observers apply to ALL flows of the dataset in index order, so
    # the random stream runs over every flow; the statistics are aggregated over the labelled flows only
    mA = d.meta[:, :T].astype(np.float64); nA = np.minimum(d.meta_len, T)
    rng = np.random.RandomState(20000); ins = realn = pushed = 0; dbytes = rbytes = 0.0; growth = []
    for i in range(len(mA)):
        k_ = int(nA[i])
        if k_ == 0: continue
        mm = mA[i, :k_]; t = np.cumsum(mm[:, 3]); dur = max(float(t[-1]), 1e-3); pk = [(t[j], 0, mm[j, 1]) for j in range(k_)]
        ins_i = 0; db_i = 0.0
        for dd, budget in ((1.0, 700), (-1.0, 1400)):
            cap = max(1, int(round(min(budget, k_)))); kk = rng.randint(1, cap + 1); w = rng.uniform(1.0, 14.0)
            ts = np.clip(rng.rayleigh(max(w / 14.0 * dur, 1e-4), kk), 0, dur)
            same = np.where(mm[:, 0] == dd)[0]; pool = same if len(same) else np.arange(k_)
            for tt in ts:
                jj = pool[rng.randint(len(pool))]; pk.append((float(tt), 1, mm[jj, 1])); ins_i += 1; db_i += mm[jj, 1]
        if not lab[i]: continue
        ins += ins_i; dbytes += db_i
        pk.sort(key=lambda p: p[0]); win = pk[:T]; realn += k_; rbytes += mm[:, 1].sum()
        pushed += k_ - sum(1 for p in win if p[1] == 0); growth.append(len(win) / k_ - 1)
    r['front'] = dict(dummies_per_real_packet=ins / realn, dummy_bytes_over_real=dbytes / rbytes, window_growth=float(np.mean(growth)),
                      real_pushed_out=pushed / realn, params=dict(N_client=700, N_server=1400, W_min_s=1, W_max_s=14, budget_cap='flow length', window='scaled to flow duration'))
    # ECH proxy
    _, _, ove = W.ech(m, n); r['ech'] = dict(clienthello_bytes=float(ove['clienthello_bytes']), share_of_flows_padded=float(ove['share_of_flows']))
    # jitter
    dj = np.random.RandomState(10000).exponential(0.020, size=(len(m), T)) * (m[:, :, 1] > 0); cum = np.cumsum(dj, 1)
    valid = np.arange(T)[None, :] < n[:, None]; r['jitter'] = dict(mean_added_delay_ms=1000 * float(cum[valid].mean()))
    OUT[ds] = r
    print(f'\n== {ds}'); [print(' ', k, v) for k, v in r.items()]
json.dump(OUT, open(os.path.join(ROOT, 'results', 'defence_stats.json'), 'w'), indent=1)
print('\nwritten results/defence_stats.json')
