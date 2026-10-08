"""Tamaraw and FRONT against observers that see the WHOLE connection, and their whole-connection costs.

Every defence cost and observer elsewhere in the paper is limited to the first 64 packets of a connection. Here both
published defences are simulated over the complete packet trace of each connection (functions in
scripts/wf_defenses_whole.py, whose docstring documents the MIRAGE fields used), and LightGBM is given the flow-level
quantities the defended connection still exposes (what a flow record shows). FRONT has the parameters of
scripts/wf_defenses.py. Tamaraw has two configurations (constants in scripts/wf_defenses_whole.py), used by three modes:
  default mode         the double-rate configuration (1452-B packets every 20 ms up / 5 ms down, L = 20), together with
                       FRONT and the undefended references; output under the keys 'genai' and 'ccma'. Its 64-packet
                       Tamaraw features (observers 64 and (ii)) are those of wf_defenses.apply_named('tamaraw'), the
                       64-packet simulation at the published configuration.
  --tamaraw_L_grid     the double-rate configuration with the padding multiple L varied; key 'tamaraw_L_grid'.
  --tamaraw_published  the published main configuration of Tamaraw (1500-B packets every 40 ms up / 12 ms down; primary
                       L = 100, sensitivity L = 1000); key 'tamaraw_published'. This is the configuration the paper
                       reports.

Analysis rules:
  Data and protocol. K=4 labelled sessions per class, seeds 0-4, splits configs/splits/fewshot_k4_s{s}.json (GenAI,
    six assistant x modality classes, y_joint) and ccma_fewshot_k4_s{s}.json (CCMA, nine apps, y_app): the splits of
    the FRONT/Tamaraw table. Training uses the K labelled sessions only (split 'train'); validation sessions are not used at all; the
    test sessions are only scored.
  Observer. LightGBM with the fixed recipe of scripts/train_baselines.py (1,600 rounds, learning rate 0.03, 31 leaves,
    subsample 0.8, colsample 0.8, at least 5 samples per leaf, random_state = seed, features standardised with the
    training mean and s.d.), no tuning, no model selection. Feature sets:
      W   = whole-connection observables (9): packets, IP bytes and payload bytes per direction, connection duration,
            per-direction duration; for Tamaraw the padded counts, their fixed-size bytes, the defended duration and spans;
            for FRONT the totals including dummies (wf_defenses_whole.W_COLS).
      64  = the paper's 45 LightGBM features of the first 64 packets (train_baselines.stat_features), computed from the
            paper's own 64-packet simulation of the same defence (wf_defenses.apply_named(defence, meta, meta_len, seed),
            applied to every flow as in the paper's runs).
      start = start of the connection relative to the first packet of its capture session (the connection start time
            the defences leave in place; needs the capture-session boundary, so it is a secondary observer).
    Observers per defence D in {Tamaraw, FRONT}, all reported:
      (i)   adaptive W        : trained and tested on defended traffic, features W  [primary]
      (ii)  adaptive W + 64   : trained and tested on defended traffic, features W and the paper's 64-packet features
      (iii) unaware W         : trained on undefended W, tested on defended W
      (iv)  adaptive W + start: secondary (uses the capture-session boundary)
      checks: adaptive 64 and unaware 64 must reproduce the paper's locked LightGBM runs (<D>_adv1 / _adv0).
    References without defence: 64 (must reproduce fs4_lgbm_meta64 / ccma_fs4_lgbm_meta64), W, W + 64, W + start.
    FRONT uses the realisation of seed s (per flow, per seed) for both training and test flows of seed s; Tamaraw is
    deterministic.
  Metrics (as scripts/robust_assistant.py). Task macro-F1 (GenAI 6 classes, CCMA 9 apps; sklearn macro over the labels
    present, zero_division 0); GenAI assistant macro-F1 of pred//2 against y//2; session vote: the majority of the
    predicted assistant (GenAI) or app (CCMA) over a test session's connections (ties to the lowest index) against the
    session's label, as accuracy over test sessions. Baselines: uniform guess (session 1/3 or 1/9; per-connection F1 by
    the expected-count approximation of robust_assistant.uniform_f1) and majority (a constant guess of the most frequent
    test class: its macro-F1 per connection and its session accuracy). Mean and seed range (min-max) over seeds 0-4.
  Overheads, over every labelled flow (as scripts/defence_stats.py), whole connection: extra bytes aggregated
    (sum extra / sum real IP bytes) and per-flow median; Tamaraw: mean queueing delay per real packet (all packets pooled,
    the median and the 90th percentile of per-flow means, and the share of flows whose mean exceeds 1 s), added duration
    (defended minus original: pooled sum over sum of original durations, median in seconds, median ratio
    defended/original); FRONT: dummy bytes over real bytes (aggregate, per-flow median), dummies per real packet, no
    delay; mean and range over the five FRONT seeds.
Checks recorded in the output: every index row found in the raw JSON; the first 64 packets of each extracted trace equal
the derived meta array; the per-direction totals equal MIRAGE's UF_/DF_/BF_ flow_metadata fields.
Writes results/whole_connection_defences.json only (an optional --cache_dir keeps the extracted per-flow arrays). The
default mode keeps every top-level key of an existing output file that it does not produce itself (e.g. 'tamaraw_L_grid').

TAMARAW PADDING-MULTIPLE GRID (--tamaraw_L_grid 20,100,1000; output under the key 'tamaraw_L_grid').
  Why. The default mode pads each direction to a multiple of L = 20 packets. Over a whole connection this leaves the
    defended duration visible to L * rho_d (0.4 s up, 0.1 s down).
  Published L. Cai, Nithyanand, Wang, Johnson, Goldberg, CCS 2014 (doi 10.1145/2660267.2660362), Sec. 6.2: 'Here we set
    L to 100'; Table 3 (rho_out 0.02, rho_in 0.006, 750-B packets) and Figs. 1 and 4 use L = 100, on whole page loads.
    Wang's reference simulation (https://www.cs.sfu.ca/~taowang/wf/defenses/tamaraw.py) calls AnoaPad(list2, list3, 100, 0),
    i.e. padL = 100. The published L, 100, is in the grid.
  Grid: L in {20, 100, 1000}. The L = 20 row is the default mode's configuration and must reproduce its stored observers
    (i) and (iii) exactly. Every value of the grid is reported.
  Everything else as in the default mode: 1452-B packets (1400 B payload), rho 20 ms up / 5 ms down, one real packet per
    slot, the causal queue, both directions run until the connection's last real packet has left, then each direction is
    padded deterministically to ceil(max(c_d, 1) / L) * L (the rule of Cai et al.; Wang's code adds a random geometric
    number of further multiples of L, which is not used for any L). Same data, splits (K = 4, seeds 0-4, both campaigns),
    LightGBM recipe, metrics and baselines as above.
  Observers per L: (i) adaptive W (trained and tested on Tamaraw-L traffic) and (iii) unaware W (trained on undefended W,
    tested on Tamaraw-L W). Recomputed with the same code as references and required to equal the stored numbers: no
    defence W, and FRONT (i) (used only in the truncation check).
  Costs per L, over every labelled flow: extra bytes (aggregate and per-flow median), number of distinct defended inputs
    and the largest share of flows with one input, duration resolution L * rho_d, added duration (aggregate, median, mean,
    median ratio defended/original, share of flows whose duration more than doubles); queueing delay (pooled mean per real
    packet, median and p90 of per-flow means, share of flows whose mean exceeds 1 s): the same for every L, because
    padding starts after the connection's last real packet has left.
  Truncation check. A test connection is inside the capture iff its first packet is more than 1 s after the capture's
    first packet and its last packet more than 1 s before the capture's last packet (capture = the session JSON, first and
    last packet over all its biflows). The session vote is counted over inside connections only, with the same trained
    model (training flows are not filtered); sessions without an inside connection are dropped and the number of retained
    sessions is reported. Also reported: the shares of labelled connections that start or end within 1 s of the capture
    edges, and the median capture length per class.
  Leak removed (descriptor): 100 * (no defence W - Tamaraw-L (i)) / (no defence W - uniform guess), on seed means, for the
    task F1, the GenAI assistant F1 and the session vote.
  Paired two-level bootstrap (scripts/boot2.Cell: B = 4000, RandomState(20260926), draws x test sessions) of test macro-F1
    differences of observer (i): Tamaraw-L minus no defence W for each L; L=100 minus L=20; L=1000 minus L=20; L=1000
    minus L=100; the task F1 on both campaigns and the assistant F1 on GenAI.

TAMARAW, PUBLISHED CONFIGURATION (--tamaraw_published 100,1000; output under the key 'tamaraw_published'; every other key
is kept).
  Configuration. The default mode and tamaraw_L_grid use the double-rate configuration, 1452-B packets every 20 ms up /
    5 ms down, about twice the bandwidth of Cai et al.'s configurations. This mode uses the published main configuration
    (Cai et al., CCS 2014, Sec. 6.2: 'With MTU packets, rho_out = 0.04 and rho_in = 0.012', 'Here we set L to 100'): MTU
    packets of 1500 IP bytes (1448 payload bytes; wf_defenses.TAM_PKT_IP / TAM_PKT_PAY, = wf_defenses_whole.PUB_PKT_IP /
    PUB_PKT_PAY), rho 40 ms up / 12 ms down, L = 100 (published; primary) and L = 1000 (a stronger multiple; sensitivity).
  Everything else as in tamaraw_L_grid: the causal queue (wf_defenses_whole._slots, one real packet per slot, s_j =
    max(s_{j-1}+1, ceil(t_j / rho_d))), both directions run until the connection's last real packet has left, then each
    direction is padded deterministically to ceil(max(c_d, 1) / L) * L at its own rate; same data, splits (K = 4, seeds
    0-4, both campaigns), LightGBM recipe (lgbm_predict), feature set W, metrics, uniform and majority baselines and
    truncation rule (inside the capture: first packet > 1 s after the capture's first packet and last packet > 1 s
    before its last).
  Extraction. One pass over the raw JSON (wf_defenses_whole.extract_tamaraw_rates) computes the L-independent quantities
    at the published rates ('pub') and at the double-rate configuration's 20/5 ms ('old'); the 'old' ones must equal the
    cached tamaraw_L_grid extraction exactly (c_up, c_dn, delay_sum, delay_n; start_off, end_gap), which checks that this
    pass reads the same packets with the same code.
  Observers per L (published configuration): (i) adaptive W (trained and tested on defended W) [primary] and (iii)
    unaware W (trained on undefended W, tested on defended W). Recomputed with the same code as references, each required
    to equal its stored per-seed numbers: no defence W (key <ds>.observers.none.W), and (i) at the double-rate
    configuration for the same L ('double-rate L=<L>: (i) W adaptive', against tamaraw_L_grid.<ds>.observers); the latter
    serves the paired published-minus-double-rate comparison.
  Metrics per observer: task macro-F1, GenAI assistant macro-F1, session vote, session vote over inside-capture
    connections (sessions without one dropped; their number reported); mean, seed range and per-seed values; the number
    of seeds in which (i)'s task F1 and assistant F1 exceed the per-seed uniform-guess F1.
  Session vote against chance. Two-level bootstrap with the resampling of boot2 (B = 4000, np.random.RandomState(20260926),
    drawn once per campaign in this order: draw indices (B x 5) uniformly from the five seeds, then test-session
    multiplicities (B x n_sessions) ~ Multinomial(n_sessions, uniform)); statistic = the session-weighted vote accuracy
    averaged over the five selected draws; 95% percentile interval; 'above chance' iff its lower end exceeds 100 / (number
    of assistants or apps). The inside-capture vote uses the same resamples, a dropped session weighing 0 in numerator
    and denominator.
  Costs per L over every labelled flow (tam_costs with the published packet size and rates): extra bytes (aggregate and
    per-flow median), defended/real bytes, distinct inputs and the largest share of flows with one input, duration
    resolution L * rho_d, added duration (aggregate, median, mean, median ratio, share of flows more than doubled);
    queueing delay (pooled mean per real packet, median and p90 of per-flow means, share of flows whose mean exceeds 1 s),
    the same for every L. Also the bandwidth per direction (pkt_ip / rho_d, kB/s) of the published and of the double-rate
    configuration, and the ratio of the latter to the former.
  Leak removed (descriptor): as tamaraw_L_grid, for (i), on seed means.
  Paired two-level bootstrap (boot2.Cell, B = 4000, RandomState(20260926)) of test macro-F1 differences (task on both
    campaigns, assistant on GenAI): published-L (i) minus no defence W for each L; published L=1000 minus published L=100;
    published-L (i) minus double-rate-L (i) for each L. Every number is reported."""
import argparse, json, os, sys, time
import numpy as np
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
import wf_defenses_whole as WW
from wf_defenses import apply_named
from train_baselines import stat_features
import lightgbm as lgb

SEEDS = [0, 1, 2, 3, 4]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def lgbm_predict(Xtr, ytr, Xte, seed):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = lgb.LGBMClassifier(n_estimators=1600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                             min_child_samples=5, random_state=seed, verbose=-1)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.predict((Xte - mu) / sd).astype(int)


def mf1(y, p): return 100 * f1_score(y, p, average='macro', zero_division=0)


def uniform_f1(y, C):  # scripts/robust_assistant.uniform_f1
    p = np.bincount(y, minlength=C) / len(y); p = p[p > 0]
    return 100 * float(np.mean(2 * p / C / (p + 1 / C)))


def sess_vote(yv, pv, inv, nS, C):
    ys = np.array([np.bincount(yv[inv == j], minlength=C).argmax() for j in range(nS)])
    ps = np.array([np.bincount(pv[inv == j], minlength=C).argmax() for j in range(nS)])
    return 100 * float((ys == ps).mean())


def summ(v):
    v = [float(x) for x in v]
    return dict(mean=round(float(np.mean(v)), 2), min=round(min(v), 2), max=round(max(v), 2), per_seed=[round(x, 2) for x in v])


def locked_records():
    L = {}
    for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
        j = json.loads(l)
        if not str(j.get('tag', '')).startswith('smoke'): L[j['run_id']] = j['test'].get('macro_f1')
    return L


def load_or_extract(ds, D, cache_dir):
    p = os.path.join(cache_dir, f'whole_{ds}.npz') if cache_dir else None
    if p and os.path.exists(p):
        log(f'{ds}: loading cached extraction {p}'); z = np.load(p); return {k: z[k] for k in z.files}
    R = WW.extract(ROOT, ds, D.meta, D.meta_len, SEEDS, log=log)
    if p: os.makedirs(cache_dir, exist_ok=True); np.savez_compressed(p, **R); log(f'{ds}: cached extraction to {p}')
    return R


def overheads(R, lab, ds):
    o = {}
    real = R['real_bytes'][lab]; extra = R['tam_extra'][lab]; do = R['dur_orig'][lab]; dd = R['dur_def'][lab]
    pos = do > 0
    per_flow_delay = R['tam_delay_sum'][lab] / np.maximum(R['tam_delay_n'][lab], 1)
    Wt = R['W_tam'][lab]; uniq, cnt = np.unique(Wt, axis=0, return_counts=True)
    o['tamaraw_whole'] = dict(
        extra_bytes_aggregate=round(float(extra.sum() / real.sum()), 2),
        extra_bytes_median_per_flow=round(float(np.median(extra / np.maximum(real, 1))), 2),
        extra_bytes_mean_of_ratios=round(float(np.mean(extra / np.maximum(real, 1))), 2),
        queue_delay_ms_per_packet_pooled=round(1000 * float(R['tam_delay_sum'][lab].sum() / R['tam_delay_n'][lab].sum()), 1),
        queue_delay_ms_median_of_flow_means=round(1000 * float(np.median(per_flow_delay)), 1),
        queue_delay_ms_p90_of_flow_means=round(1000 * float(np.percentile(per_flow_delay, 90)), 1),
        share_flows_mean_queue_delay_over_1s=round(float(np.mean(per_flow_delay > 1.0)), 4),
        added_duration_aggregate=round(float((dd - do).sum() / do.sum()), 3),
        added_duration_median_s=round(float(np.median(dd - do)), 3),
        added_duration_mean_s=round(float(np.mean(dd - do)), 2),
        duration_ratio_median=round(float(np.median(dd[pos] / do[pos])), 3),
        share_flows_duration_doubled=round(float(np.mean(dd[pos] > 2 * do[pos])), 4),
        original_duration_median_s=round(float(np.median(do)), 3), original_duration_mean_s=round(float(np.mean(do)), 2),
        padded_pkts_up_median=float(np.median(Wt[:, 0])), padded_pkts_dn_median=float(np.median(Wt[:, 1])),
        distinct_inputs=int(len(uniq)), largest_input_share=round(float(cnt.max() / len(Wt)), 4))
    fb = R['front_dummy_bytes'][:, lab]; fp = R['front_dummy_pkts'][:, lab]
    agg = fb.sum(1) / real.sum(); med = np.median(fb / np.maximum(real, 1)[None], axis=1); dpp = fp.sum(1) / R['real_pkts'][lab].sum()
    o['front_whole'] = dict(dummy_bytes_aggregate=summ(agg), dummy_bytes_median_per_flow=summ(med), dummies_per_real_packet=summ(dpp),
                            added_delay_s=0.0)
    return o


OUT_PATH = os.path.join(ROOT, 'results', 'whole_connection_defences.json')


def read_json():
    return json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}


def write_json(obj):
    json.dump(obj, open(OUT_PATH, 'w'), indent=1)


# ============================================================================ Tamaraw padding-multiple grid
PUBLISHED_L = dict(
    value=100, in_grid=True,
    paper='Cai, Nithyanand, Wang, Johnson, Goldberg. A Systematic Approach to Developing and Evaluating Website Fingerprinting '
          'Defenses. ACM CCS 2014, doi 10.1145/2660267.2660362. Sec. 6.2: "Here we set L to 100" (with MTU packets, rho_out 0.04 s, '
          'rho_in 0.012 s); Table 3 uses rho_out 0.02, rho_in 0.006, 750-B packets and L = 100; Fig. 1 ("Tamaraw with L = 100") and '
          'Fig. 4 ("the padding mandated by L = 100"). Traces are whole page loads.',
    padding_rule_paper='Sec. 6.1: if AL < I <= (A+1)L, pad to (A+1)L, separately per direction, at that direction\'s rate '
                       '(deterministic; used here for every L).',
    reference_code='Wang, https://www.cs.sfu.ca/~taowang/wf/defenses/tamaraw.py: AnoaPad(list2, list3, 100, 0), '
                   'i.e. padL = 100; rates 0.04 s out / 0.012 s in; DATASIZE = 800. Its AnoaPad pads to (n // L + g) * L with '
                   'g >= 1 geometric (P(g = k) = 2^-k), a randomised variant of the paper\'s rule that is not used here.',
    this_paper_parameters='the double-rate configuration: 1452-B packets, rho 20 ms up / 5 ms down (scripts/wf_defenses_whole), '
                          'L from the grid; L = 20 in the default mode.')


def load_or_extract_grid(ds, cache_dir):
    p = os.path.join(cache_dir, f'tamgrid_{ds}.npz') if cache_dir else None
    if p and os.path.exists(p):
        log(f'{ds}: loading cached Tamaraw-grid extraction {p}'); z = np.load(p); return {k: z[k] for k in z.files}
    G = WW.extract_tamaraw_grid(ROOT, ds, log=log)
    if p: os.makedirs(cache_dir, exist_ok=True); np.savez_compressed(p, **G); log(f'{ds}: cached Tamaraw-grid extraction to {p}')
    return G


def tam_costs(W, dur_def, real, do, dsum, dn, L, pkt_ip=WW.PKT_IP, rho_up=WW.RHO_UP, rho_dn=WW.RHO_DN):
    """Packet size and rates are arguments; the defaults are the double-rate configuration."""
    extra = (W[:, 0] + W[:, 1]) * pkt_ip - real
    per_flow_delay = dsum / np.maximum(dn, 1); pos = do > 0
    uniq, cnt = np.unique(W, axis=0, return_counts=True)
    return extra, dict(
        L=L, duration_resolution_s=dict(up=round(L * rho_up, 3), dn=round(L * rho_dn, 3)),
        extra_bytes_aggregate=round(float(extra.sum() / real.sum()), 2),
        extra_bytes_median_per_flow=round(float(np.median(extra / np.maximum(real, 1))), 2),
        defended_over_real_bytes_aggregate=round(float((extra.sum() + real.sum()) / real.sum()), 2),
        padded_pkts_up_median=float(np.median(W[:, 0])), padded_pkts_dn_median=float(np.median(W[:, 1])),
        distinct_inputs=int(len(uniq)), largest_input_share=round(float(cnt.max() / len(W)), 4),
        queue_delay_ms_per_packet_pooled=round(1000 * float(dsum.sum() / dn.sum()), 1),
        queue_delay_ms_median_of_flow_means=round(1000 * float(np.median(per_flow_delay)), 1),
        queue_delay_ms_p90_of_flow_means=round(1000 * float(np.percentile(per_flow_delay, 90)), 1),
        share_flows_mean_queue_delay_over_1s=round(float(np.mean(per_flow_delay > 1.0)), 4),
        added_duration_aggregate=round(float((dur_def - do).sum() / do.sum()), 3),
        added_duration_median_s=round(float(np.median(dur_def - do)), 3),
        added_duration_mean_s=round(float(np.mean(dur_def - do)), 2),
        duration_ratio_median=round(float(np.median(dur_def[pos] / do[pos])), 3),
        share_flows_duration_doubled=round(float(np.mean(dur_def[pos] > 2 * do[pos])), 4))


def sess_vote_subset(yv, pv, inv, C, keep):
    """Session vote over the connections with keep=True; sessions without one are dropped. Returns (accuracy %, n sessions)."""
    js = np.unique(inv[keep])
    ys = np.array([np.bincount(yv[keep & (inv == j)], minlength=C).argmax() for j in js])
    ps = np.array([np.bincount(pv[keep & (inv == j)], minlength=C).argmax() for j in js])
    return 100 * float((ys == ps).mean()), int(len(js))


def run_L_grid(Ls, cache_dir, datasets):
    import boot2
    J = read_json()
    doc = __doc__.split('TAMARAW PADDING-MULTIPLE GRID')[1]
    GR = dict(rules='TAMARAW PADDING-MULTIPLE GRID' + doc.split('TAMARAW, PUBLISHED CONFIGURATION')[0].rstrip(),
              published_L=PUBLISHED_L, L_grid=Ls, seeds=SEEDS, K=4, observers=['(i) W adaptive', '(iii) W unaware'])
    for ds in datasets:
        D = GenAIData(ROOT, prefix=ds)
        y = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app
        nact = 2 if ds == 'genai' else 1; NA = NC // nact
        lab = y >= 0
        R = load_or_extract(ds, D, cache_dir)
        G = load_or_extract_grid(ds, cache_dir)
        stored = J[ds]['observers']
        # ---------------- checks: the L-independent part reproduces the stored L = 20 extraction exactly
        W20, dd20 = WW.tamaraw_pad(G['c_up'], G['c_dn'], 20)
        ex20 = (W20[:, 0] + W20[:, 1]) * WW.PKT_IP - R['real_bytes']
        chk = dict(all_rows_found=bool(G['got'].all()), share_flows_timestamps_sorted=round(float(G['sorted_ts'].mean()), 6),
                   L20_W_equal_to_stored=bool(np.array_equal(W20, R['W_tam'])),
                   L20_defended_duration_equal_to_stored=bool(np.array_equal(dd20, R['dur_def'])),
                   L20_extra_bytes_equal_to_stored=bool(np.array_equal(ex20, R['tam_extra'])),
                   delay_sum_equal_to_stored=bool(np.array_equal(G['delay_sum'], R['tam_delay_sum'])),
                   delay_n_equal_to_stored=bool(np.array_equal(G['delay_n'], R['tam_delay_n'])),
                   start_offset_max_abs_diff_to_stored=float(np.abs(G['start_off'] - R['start_off']).max()))
        log(ds, 'grid checks', json.dumps(chk))
        assert chk['L20_W_equal_to_stored'] and chk['delay_sum_equal_to_stored'], 'L = 20 does not reproduce the stored extraction'
        real, do = R['real_bytes'][lab], R['dur_orig'][lab]
        WL, costs = {}, {}
        for L in Ls:
            WL[L], ddL = WW.tamaraw_pad(G['c_up'], G['c_dn'], L)
            _, costs[L] = tam_costs(WL[L][lab], ddL[lab], real, do, G['delay_sum'][lab], G['delay_n'][lab], L)
            log(ds, f'L={L} costs', json.dumps(costs[L]))
        c20 = J[ds]['overheads_whole']['tamaraw_whole']
        chk['L20_costs_equal_to_stored'] = all(costs[20][k] == c20[k] for k in costs[20] if k in c20) if 20 in costs else None
        # ---------------- truncation descriptors
        so, eg = G['start_off'], G['end_gap']; inside_all = (so > 1.0) & (eg > 1.0)
        trunc = dict(share_start_within_1s=round(float(np.mean(so[lab] <= 1.0)), 4), share_end_within_1s=round(float(np.mean(eg[lab] <= 1.0)), 4),
                     share_inside=round(float(np.mean(inside_all[lab])), 4))
        # median capture length per class: GenAI by assistant (app_dir) and by capture folder (class_dir); CCMA by app
        for tag, cls_col in ((('assistant', 'app_dir'), ('capture_folder', 'class_dir')) if ds == 'genai' else (('app', 'app'),)):
            sess_len = {}
            for sid in np.unique(D.session_id[lab]):
                m = lab & (D.session_id == sid); cname = D.index[cls_col].values[m][0]
                sess_len.setdefault(str(cname), []).append(float(G['cap_len'][m][0]))
            trunc[f'capture_length_s_median_by_{tag}'] = {k: round(float(np.median(v)), 1) for k, v in sorted(sess_len.items())}
            trunc[f'n_captures_by_{tag}'] = {k: len(v) for k, v in sorted(sess_len.items())}
        # ---------------- observers
        split = 'fewshot_k4' if ds == 'genai' else 'ccma_fewshot_k4'
        acc, preds, te0 = {}, {}, None
        uni = dict(task_f1=[], asst_f1=[], sess_vote=[])

        def score(name, p, yt, ya, inv, nS, inside):
            r = acc.setdefault(name, {})
            r.setdefault('task_f1', []).append(mf1(yt, p))
            if ds == 'genai': r.setdefault('asst_f1', []).append(mf1(ya, p // nact))
            r.setdefault('sess_vote', []).append(sess_vote(ya, p // nact, inv, nS, NA))
            v, n = sess_vote_subset(ya, p // nact, inv, NA, inside)
            r.setdefault('sess_vote_inside_capture', []).append(v); r.setdefault('n_test_sessions_inside_capture', []).append(n)
            preds.setdefault(name, []).append(p)
            log(f'{ds} s{len(r["task_f1"]) - 1} {name:34s} task {r["task_f1"][-1]:5.1f}' + (f' asst {r["asst_f1"][-1]:5.1f}' if ds == 'genai' else '')
                + f' sess {r["sess_vote"][-1]:5.1f} inside {v:5.1f} (n={n})')

        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json'))
            tr, it = idx['train'], idx['test']; yt = y[it]; ya = yt // nact
            if te0 is None: te0 = it
            assert np.array_equal(te0, it), 'test connections differ across draws'
            us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
            inside = inside_all[it]
            uni['task_f1'].append(uniform_f1(yt, NC)); uni['asst_f1'].append(uniform_f1(ya, NA)); uni['sess_vote'].append(100 / NA)
            # one model on undefended W: the no-defence reference on undefended test flows, and observer (iii) on every L
            Wraw = R['W_raw']; nte = len(it)
            p_all = lgbm_predict(Wraw[tr], y[tr], np.concatenate([Wraw[it]] + [WL[L][it] for L in Ls]), s)
            score('none: W', p_all[:nte], yt, ya, inv, nS, inside)
            for k, L in enumerate(Ls):
                score(f'tamaraw L={L}: (iii) W unaware', p_all[(k + 1) * nte:(k + 2) * nte], yt, ya, inv, nS, inside)
            for L in Ls:
                score(f'tamaraw L={L}: (i) W adaptive', lgbm_predict(WL[L][tr], y[tr], WL[L][it], s), yt, ya, inv, nS, inside)
            Wfr = R['W_front'][s]
            score('front: (i) W adaptive', lgbm_predict(Wfr[tr], y[tr], Wfr[it], s), yt, ya, inv, nS, inside)
        # ---------------- reproduction of the stored numbers (per seed, 2 decimals)
        pairs = [('none: W', 'none', 'W'), ('front: (i) W adaptive', 'front', '(i) W adaptive')]
        if 20 in Ls: pairs += [('tamaraw L=20: (i) W adaptive', 'tamaraw', '(i) W adaptive'), ('tamaraw L=20: (iii) W unaware', 'tamaraw', '(iii) W unaware')]
        repro = {}
        for name, st, ob in pairs:
            for met in ('task_f1', 'asst_f1', 'sess_vote'):
                if met not in acc[name]: continue
                here = [round(float(x), 2) for x in acc[name][met]]; there = stored[st][ob][met]['per_seed']
                repro[f'{name} | {met}'] = dict(equal=here == there, here=here, stored=there)
        chk['reproduces_stored_observers'] = all(v['equal'] for v in repro.values())
        log(ds, 'reproduction of stored observers:', chk['reproduces_stored_observers'])
        obs = {name: {k: (summ(v) if not k.startswith('n_') else v) for k, v in r.items()} for name, r in acc.items()}
        U = {k: summ(v) for k, v in uni.items() if v and (k != 'asst_f1' or ds == 'genai')}
        # ---------------- leak removed (seed means)
        leak = {}
        for L in Ls:
            nm = f'tamaraw L={L}: (i) W adaptive'
            leak[f'L={L}'] = {met: round(100 * (np.mean(acc['none: W'][met]) - np.mean(acc[nm][met])) / (np.mean(acc['none: W'][met]) - np.mean(uni[met])), 1)
                              for met in ('task_f1', 'asst_f1', 'sess_vote') if met in acc[nm]}
        # ---------------- paired two-level bootstrap (boot2.Cell)
        yte, sid = y[te0], D.session_id[te0]
        cells = {'task': (boot2.Cell(yte, sid, NC), lambda p: p)}
        if ds == 'genai': cells['assistant'] = (boot2.Cell(yte // nact, sid, NA), lambda p: p // nact)
        comps = [(f'tamaraw L={L}: (i) W adaptive', 'none: W') for L in Ls]
        comps += [(f'tamaraw L={b}: (i) W adaptive', f'tamaraw L={a}: (i) W adaptive') for a, b in ((20, 100), (20, 1000), (100, 1000)) if a in Ls and b in Ls]
        boot = {}
        for metric, (cell, f) in cells.items():
            for name in {n for c in comps for n in c}: cell.add(name, [f(p) for p in preds[name]])
            for a_, b_ in comps:
                c = cell.compare(a_, b_)
                boot[f'{metric} | {a_} minus {b_}'] = dict(diff=round(c['diff'], 2), ci95=[round(x, 2) for x in c['ci']], verdict=c['verdict'],
                                                           per_draw=[round(x, 2) for x in c['per_draw']], n_test_sessions=c['n_test_sessions'])
        GR[ds] = dict(checks=chk, costs={f'L={L}': costs[L] for L in Ls}, truncation=trunc, uniform_guess=U,
                      majority_guess_stored=dict(task_f1=J[ds]['references']['majority_task_f1'], sess_vote=J[ds]['references']['majority_sess']),
                      observers=obs, leak_removed_vs_no_defence_W_pct=leak, bootstrap_two_level=boot, reproduction=repro)
        J = read_json(); J['tamaraw_L_grid'] = GR; write_json(J)
        log(f'{ds} grid done; written results/whole_connection_defences.json')
        for name, m in obs.items():
            log(f'  {ds} {name:34s} task {m["task_f1"]["mean"]:5.1f} [{m["task_f1"]["min"]:.1f}-{m["task_f1"]["max"]:.1f}]'
                + (f' asst {m["asst_f1"]["mean"]:5.1f} [{m["asst_f1"]["min"]:.1f}-{m["asst_f1"]["max"]:.1f}]' if 'asst_f1' in m else '')
                + f' sess {m["sess_vote"]["mean"]:5.1f} [{m["sess_vote"]["min"]:.1f}-{m["sess_vote"]["max"]:.1f}]'
                + f' inside {m["sess_vote_inside_capture"]["mean"]:5.1f} [{m["sess_vote_inside_capture"]["min"]:.1f}-{m["sess_vote_inside_capture"]["max"]:.1f}]')
        for k, v in boot.items(): log(f'  {ds} boot {k}: {v["diff"]:+.2f} {v["ci95"]} {v["verdict"]}')


# ============================================================================ Tamaraw, published configuration
PUB = dict(rho_up=WW.PUB_RHO_UP, rho_dn=WW.PUB_RHO_DN, pkt_ip=WW.PUB_PKT_IP, pkt_pay=WW.PUB_PKT_PAY)
OLD = dict(rho_up=WW.RHO_UP, rho_dn=WW.RHO_DN, pkt_ip=WW.PKT_IP, pkt_pay=WW.PKT_PAY)


def load_or_extract_rates(ds, cache_dir):
    p = os.path.join(cache_dir, f'tamrates_{ds}.npz') if cache_dir else None
    if p and os.path.exists(p):
        log(f'{ds}: loading cached Tamaraw-rates extraction {p}'); z = np.load(p); return {k: z[k] for k in z.files}
    G = WW.extract_tamaraw_rates(ROOT, ds, dict(pub=(PUB['rho_up'], PUB['rho_dn']), old=(OLD['rho_up'], OLD['rho_dn'])), log=log)
    if p: os.makedirs(cache_dir, exist_ok=True); np.savez_compressed(p, **G); log(f'{ds}: cached Tamaraw-rates extraction to {p}')
    return G


def boot_vote(C, keep=None, B=4000, seed=20260926):
    """Two-level bootstrap of the session-vote accuracy (%): C [5, nS] per-draw correctness (1/0), keep [5, nS] or None."""
    R, nS = C.shape; rs = np.random.RandomState(seed)
    Ss = rs.randint(0, R, size=(B, R)); Ws = rs.multinomial(nS, np.ones(nS) / nS, size=B)
    K = np.ones_like(C) if keep is None else keep.astype(float)
    num = np.einsum('brs,bs->b', (C * K)[Ss], Ws); den = np.einsum('brs,bs->b', K[Ss], Ws)
    acc = 100 * num / np.maximum(den, 1e-12); lo, hi = np.percentile(acc, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def run_tamaraw_published(Ls, cache_dir, datasets):
    import boot2
    J = read_json()
    doc = __doc__.split('TAMARAW, PUBLISHED CONFIGURATION')[1]
    TP = dict(rules='TAMARAW, PUBLISHED CONFIGURATION' + doc.rstrip(), L_values=Ls, seeds=SEEDS, K=4,
              configuration=dict(packet_ip_bytes=PUB['pkt_ip'], packet_payload_bytes=PUB['pkt_pay'], rho_up_s=PUB['rho_up'], rho_dn_s=PUB['rho_dn'],
                                 source='Cai, Nithyanand, Wang, Johnson, Goldberg, CCS 2014, Sec. 6.2 (MTU packets, rho_out 0.04 s, rho_in 0.012 s, L = 100)'),
              bandwidth_kB_per_s=dict(published=dict(up=round(PUB['pkt_ip'] / PUB['rho_up'] / 1000, 2), dn=round(PUB['pkt_ip'] / PUB['rho_dn'] / 1000, 2)),
                                      double_rate=dict(up=round(OLD['pkt_ip'] / OLD['rho_up'] / 1000, 2), dn=round(OLD['pkt_ip'] / OLD['rho_dn'] / 1000, 2)),
                                      double_rate_over_published=dict(up=round((OLD['pkt_ip'] / OLD['rho_up']) / (PUB['pkt_ip'] / PUB['rho_up']), 3),
                                                                      dn=round((OLD['pkt_ip'] / OLD['rho_dn']) / (PUB['pkt_ip'] / PUB['rho_dn']), 3))),
              observers=['(i) W adaptive', '(iii) W unaware'])
    for ds in datasets:
        D = GenAIData(ROOT, prefix=ds)
        y = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app
        nact = 2 if ds == 'genai' else 1; NA = NC // nact
        lab = y >= 0
        R = load_or_extract(ds, D, cache_dir)            # rate-independent: W_raw, real bytes, durations (cached extraction of the default mode)
        Gold = load_or_extract_grid(ds, cache_dir)       # double-rate L-independent quantities (cached extraction of the L grid)
        G = load_or_extract_rates(ds, cache_dir)
        chk = dict(all_rows_found=bool(G['got'].all()))
        for k in ('c_up', 'c_dn', 'delay_sum', 'delay_n'):
            chk[f'old_{k}_equal_to_cached_grid'] = bool(np.array_equal(G[f'old_{k}'], Gold[k]))
        for k in ('start_off', 'end_gap', 'cap_len'):
            chk[f'{k}_equal_to_cached_grid'] = bool(np.array_equal(G[k], Gold[k]))
        log(ds, 'checks', json.dumps(chk))
        assert all(v for k, v in chk.items()), 'the rates extraction does not reproduce the cached grid extraction'
        real, do = R['real_bytes'][lab], R['dur_orig'][lab]
        WP, WO, costs = {}, {}, {}
        for L in Ls:
            WP[L], ddL = WW.tamaraw_pad(G['pub_c_up'], G['pub_c_dn'], L, PUB['rho_up'], PUB['rho_dn'], PUB['pkt_ip'], PUB['pkt_pay'])
            _, costs[L] = tam_costs(WP[L][lab], ddL[lab], real, do, G['pub_delay_sum'][lab], G['pub_delay_n'][lab], L, PUB['pkt_ip'], PUB['rho_up'], PUB['rho_dn'])
            WO[L], _ = WW.tamaraw_pad(G['old_c_up'], G['old_c_dn'], L)
            log(ds, f'published L={L} costs', json.dumps(costs[L]))
        so, eg = G['start_off'], G['end_gap']; inside_all = (so > 1.0) & (eg > 1.0)
        split = 'fewshot_k4' if ds == 'genai' else 'ccma_fewshot_k4'
        acc, preds, corr, keepm, te0 = {}, {}, {}, {}, None
        uni = dict(task_f1=[], asst_f1=[], sess_vote=[])

        def score(name, p, yt, ya, inv, nS, inside):
            r = acc.setdefault(name, {})
            r.setdefault('task_f1', []).append(mf1(yt, p))
            if ds == 'genai': r.setdefault('asst_f1', []).append(mf1(ya, p // nact))
            r.setdefault('sess_vote', []).append(sess_vote(ya, p // nact, inv, nS, NA))
            v, n = sess_vote_subset(ya, p // nact, inv, NA, inside)
            r.setdefault('sess_vote_inside_capture', []).append(v); r.setdefault('n_test_sessions_inside_capture', []).append(n)
            preds.setdefault(name, []).append(p)
            ys = np.array([np.bincount(ya[inv == j], minlength=NA).argmax() for j in range(nS)])
            ps = np.array([np.bincount((p // nact)[inv == j], minlength=NA).argmax() for j in range(nS)])
            ok = np.zeros(nS); kp = np.zeros(nS, bool)
            for j in range(nS):
                mm = (inv == j) & inside
                if mm.any(): kp[j] = True; ok[j] = float(np.bincount((p // nact)[mm], minlength=NA).argmax() == ys[j])
            corr.setdefault(name, []).append((ys == ps).astype(float)); keepm.setdefault(name, []).append((ok, kp))
            log(f'{ds} s{len(r["task_f1"]) - 1} {name:40s} task {r["task_f1"][-1]:5.1f}' + (f' asst {r["asst_f1"][-1]:5.1f}' if ds == 'genai' else '')
                + f' sess {r["sess_vote"][-1]:5.1f} inside {v:5.1f} (n={n})')

        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json'))
            tr, it = idx['train'], idx['test']; yt = y[it]; ya = yt // nact
            if te0 is None: te0 = it
            assert np.array_equal(te0, it), 'test connections differ across draws'
            us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
            inside = inside_all[it]
            uni['task_f1'].append(uniform_f1(yt, NC)); uni['asst_f1'].append(uniform_f1(ya, NA)); uni['sess_vote'].append(100 / NA)
            Wraw = R['W_raw']; nte = len(it)
            p_all = lgbm_predict(Wraw[tr], y[tr], np.concatenate([Wraw[it]] + [WP[L][it] for L in Ls]), s)
            score('none: W', p_all[:nte], yt, ya, inv, nS, inside)
            for k, L in enumerate(Ls):
                score(f'published L={L}: (iii) W unaware', p_all[(k + 1) * nte:(k + 2) * nte], yt, ya, inv, nS, inside)
            for L in Ls:
                score(f'published L={L}: (i) W adaptive', lgbm_predict(WP[L][tr], y[tr], WP[L][it], s), yt, ya, inv, nS, inside)
                score(f'double-rate L={L}: (i) W adaptive', lgbm_predict(WO[L][tr], y[tr], WO[L][it], s), yt, ya, inv, nS, inside)
        # ---------------- reproduction of stored numbers
        repro = {}
        for name, there in [('none: W', J[ds]['observers']['none']['W'])] + \
                [(f'double-rate L={L}: (i) W adaptive', J.get('tamaraw_L_grid', {}).get(ds, {}).get('observers', {}).get(f'tamaraw L={L}: (i) W adaptive')) for L in Ls]:
            if there is None: repro[name] = 'no stored record'; continue
            for met in ('task_f1', 'asst_f1', 'sess_vote'):
                if met in acc[name] and met in there:
                    here = [round(float(x), 2) for x in acc[name][met]]
                    repro[f'{name} | {met}'] = dict(equal=here == there[met]['per_seed'], here=here, stored=there[met]['per_seed'])
        chk['reproduces_stored_observers'] = all(v['equal'] for v in repro.values() if isinstance(v, dict))
        log(ds, 'reproduction of stored observers:', chk['reproduces_stored_observers'])
        obs = {name: {k: (summ(v) if not k.startswith('n_') else v) for k, v in r.items()} for name, r in acc.items()}
        U = {k: summ(v) for k, v in uni.items() if v and (k != 'asst_f1' or ds == 'genai')}
        for name in obs:   # seeds above the uniform guess; session-vote intervals
            obs[name]['seeds_task_f1_above_uniform'] = int(sum(a > u for a, u in zip(acc[name]['task_f1'], uni['task_f1'])))
            if ds == 'genai': obs[name]['seeds_asst_f1_above_uniform'] = int(sum(a > u for a, u in zip(acc[name]['asst_f1'], uni['asst_f1'])))
            C = np.stack(corr[name]); ci = boot_vote(C)
            Ci = np.stack([o for o, _ in keepm[name]]); Ki = np.stack([k for _, k in keepm[name]]); cii = boot_vote(Ci, Ki)
            obs[name]['sess_vote_ci95_two_level'] = ci; obs[name]['sess_vote_above_chance'] = bool(ci[0] > 100 / NA)
            obs[name]['sess_vote_inside_capture_ci95_two_level'] = cii; obs[name]['sess_vote_inside_capture_above_chance'] = bool(cii[0] > 100 / NA)
        leak = {}
        for L in Ls:
            nm = f'published L={L}: (i) W adaptive'
            leak[f'L={L}'] = {met: round(100 * (np.mean(acc['none: W'][met]) - np.mean(acc[nm][met])) / (np.mean(acc['none: W'][met]) - np.mean(uni[met])), 1)
                              for met in ('task_f1', 'asst_f1', 'sess_vote') if met in acc[nm]}
        yte, sid = y[te0], D.session_id[te0]
        cells = {'task': (boot2.Cell(yte, sid, NC), lambda p: p)}
        if ds == 'genai': cells['assistant'] = (boot2.Cell(yte // nact, sid, NA), lambda p: p // nact)
        comps = [(f'published L={L}: (i) W adaptive', 'none: W') for L in Ls]
        if len(Ls) > 1: comps += [(f'published L={Ls[-1]}: (i) W adaptive', f'published L={Ls[0]}: (i) W adaptive')]
        comps += [(f'published L={L}: (i) W adaptive', f'double-rate L={L}: (i) W adaptive') for L in Ls]
        boot = {}
        for metric, (cell, f) in cells.items():
            for name in {n for c in comps for n in c}: cell.add(name, [f(p) for p in preds[name]])
            for a_, b_ in comps:
                c = cell.compare(a_, b_)
                boot[f'{metric} | {a_} minus {b_}'] = dict(diff=round(c['diff'], 2), ci95=[round(x, 2) for x in c['ci']], verdict=c['verdict'],
                                                           per_draw=[round(x, 2) for x in c['per_draw']], n_test_sessions=c['n_test_sessions'])
        TP[ds] = dict(checks=chk, costs={f'L={L}': costs[L] for L in Ls}, uniform_guess=U,
                      majority_guess_stored=dict(task_f1=J[ds]['references']['majority_task_f1'], sess_vote=J[ds]['references']['majority_sess']),
                      double_rate_costs_stored={f'L={L}': J.get('tamaraw_L_grid', {}).get(ds, {}).get('costs', {}).get(f'L={L}') for L in Ls},
                      observers=obs, leak_removed_vs_no_defence_W_pct=leak, bootstrap_two_level=boot, reproduction=repro)
        J = read_json(); J['tamaraw_published'] = TP; write_json(J)
        log(f'{ds} done; written results/whole_connection_defences.json (key tamaraw_published)')
        for name, m in obs.items():
            log(f'  {ds} {name:40s} task {m["task_f1"]["mean"]:5.1f} [{m["task_f1"]["min"]:.1f}-{m["task_f1"]["max"]:.1f}]'
                + (f' asst {m["asst_f1"]["mean"]:5.1f} [{m["asst_f1"]["min"]:.1f}-{m["asst_f1"]["max"]:.1f}]' if 'asst_f1' in m else '')
                + f' sess {m["sess_vote"]["mean"]:5.1f} [{m["sess_vote"]["min"]:.1f}-{m["sess_vote"]["max"]:.1f}] CI {m["sess_vote_ci95_two_level"]}'
                + f' inside {m["sess_vote_inside_capture"]["mean"]:5.1f} CI {m["sess_vote_inside_capture_ci95_two_level"]}')
        for k, v in boot.items(): log(f'  {ds} boot {k}: {v["diff"]:+.2f} {v["ci95"]} {v["verdict"]}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache_dir', default=''); ap.add_argument('--datasets', default='genai,ccma')
    ap.add_argument('--tamaraw_L_grid', default='', help='e.g. 20,100,1000: run only the Tamaraw padding-multiple grid (key tamaraw_L_grid)')
    ap.add_argument('--tamaraw_published', default='', help='e.g. 100,1000: Tamaraw at its published rates and packet size (key tamaraw_published)')
    a = ap.parse_args()
    if a.tamaraw_published:
        run_tamaraw_published([int(x) for x in a.tamaraw_published.split(',')], a.cache_dir, a.datasets.split(',')); return
    if a.tamaraw_L_grid:
        run_L_grid([int(x) for x in a.tamaraw_L_grid.split(',')], a.cache_dir, a.datasets.split(',')); return
    LOCK = locked_records()
    OUT = dict(rules=__doc__.split('Analysis rules:')[1].split('Checks recorded')[0].strip(),
               fields=WW.__doc__.strip(), W_cols=WW.W_COLS, seeds=SEEDS, K=4)
    ds_stats = {}
    try:
        ds_stats = json.load(open(os.path.join(ROOT, 'results', 'defence_stats.json')))
    except Exception:
        pass
    for ds in a.datasets.split(','):
        D = GenAIData(ROOT, prefix=ds)
        y = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app
        nact = 2 if ds == 'genai' else 1; NA = NC // nact
        lab = y >= 0
        R = load_or_extract(ds, D, a.cache_dir)
        # ---------------- checks ----------------
        md_diff = np.abs(R['W_raw'][lab] - R['md_W'][lab]).max(0)
        chk = dict(n_index_rows=int(len(y)), n_labelled=int(lab.sum()), all_rows_found=bool(R['got'].all()),
                   share_first64_equal_meta=round(float(R['meta_ok'][lab].mean()), 6),
                   share_pkt_count_equal_BF_num_packets=round(float((R['real_pkts'][lab] == R['md_n'][lab]).mean()), 6),
                   max_abs_diff_vs_flow_metadata={c: float(v) for c, v in zip(WW.W_COLS, md_diff)},
                   n_flows_count_or_byte_differs_from_flow_metadata=int((np.abs(R['W_raw'][lab] - R['md_W'][lab])[:, :6] > 0.5).any(1).sum()))
        log(ds, 'checks', json.dumps(chk))
        res = dict(checks=chk, overheads_whole=overheads(R, lab, ds),
                   overheads_first64_paper=dict(tamaraw=ds_stats.get(ds, {}).get('tamaraw'), front=ds_stats.get(ds, {}).get('front')))
        log(ds, 'overheads', json.dumps(res['overheads_whole']))
        # ---------------- observers ----------------
        X64 = {'none': stat_features(D.meta, D.meta_len)}
        Wraw, Wtam, start = R['W_raw'], R['W_tam'], R['start_off'][:, None]
        split = 'fewshot_k4' if ds == 'genai' else 'ccma_fewshot_k4'
        runpre = 'fs4' if ds == 'genai' else 'ccma_fs4'
        acc = {}   # (setting, observer) -> metric -> list over seeds
        refs = dict(uniform_task_f1=[], uniform_asst_f1=[], uniform_sess=[], majority_task_f1=[], majority_asst_f1=[], majority_sess=[],
                    n_test_sessions=[], n_test_flows=[], n_train_flows=[])
        repro = []
        X64['tamaraw'] = stat_features(*apply_named('tamaraw', D.meta, D.meta_len, 0)[:2])   # deterministic
        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json'))
            tr, it = idx['train'], idx['test']; yt = y[it]; ya = yt // nact
            us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
            refs['n_test_sessions'].append(int(nS)); refs['n_test_flows'].append(int(len(it))); refs['n_train_flows'].append(int(len(tr)))
            refs['uniform_task_f1'].append(uniform_f1(yt, NC)); refs['uniform_asst_f1'].append(uniform_f1(ya, NA)); refs['uniform_sess'].append(100 / NA)
            cmaj = np.bincount(yt, minlength=NC).argmax(); refs['majority_task_f1'].append(mf1(yt, np.full_like(yt, cmaj)))
            amaj = np.bincount(ya, minlength=NA).argmax(); refs['majority_asst_f1'].append(mf1(ya, np.full_like(ya, amaj)))
            ysa = np.array([np.bincount(ya[inv == j], minlength=NA).argmax() for j in range(nS)])
            refs['majority_sess'].append(100 * float(np.bincount(ysa, minlength=NA).max() / nS))
            Xf64 = stat_features(*apply_named('front', D.meta, D.meta_len, s)[:2])
            Wfr = R['W_front'][s]
            H = lambda *b: np.concatenate(b, 1)
            OBS = [   # setting, observer, train features, test features, locked run id to reproduce (or None)
                ('none', '64 (paper LightGBM)', X64['none'], X64['none'], f'{runpre}_lgbm_meta64_s{s}'),
                ('none', 'W', Wraw, Wraw, None),
                ('none', 'W + 64', H(Wraw, X64['none']), H(Wraw, X64['none']), None),
                ('none', 'W + start', H(Wraw, start), H(Wraw, start), None),
                ('tamaraw', '64 adaptive (paper)', X64['tamaraw'], X64['tamaraw'], f'{runpre}_tamaraw_adv1_lgbm_meta64_s{s}'),
                ('tamaraw', '64 unaware (paper)', X64['none'], X64['tamaraw'], f'{runpre}_tamaraw_adv0_lgbm_meta64_s{s}'),
                ('tamaraw', '(i) W adaptive', Wtam, Wtam, None),
                ('tamaraw', '(ii) W + 64 adaptive', H(Wtam, X64['tamaraw']), H(Wtam, X64['tamaraw']), None),
                ('tamaraw', '(iii) W unaware', Wraw, Wtam, None),
                ('tamaraw', '(iv) W + start adaptive', H(Wtam, start), H(Wtam, start), None),
                ('front', '64 adaptive (paper)', Xf64, Xf64, f'{runpre}_front_adv1_lgbm_meta64_s{s}'),
                ('front', '64 unaware (paper)', X64['none'], Xf64, f'{runpre}_front_adv0_lgbm_meta64_s{s}'),
                ('front', '(i) W adaptive', Wfr, Wfr, None),
                ('front', '(ii) W + 64 adaptive', H(Wfr, Xf64), H(Wfr, Xf64), None),
                ('front', '(iii) W unaware', Wraw, Wfr, None),
                ('front', '(iv) W + start adaptive', H(Wfr, start), H(Wfr, start), None),
            ]
            for setting, obs, Xtr_all, Xte_all, rid in OBS:
                p = lgbm_predict(Xtr_all[tr], y[tr], Xte_all[it], s)
                r = acc.setdefault((setting, obs), {})
                r.setdefault('task_f1', []).append(mf1(yt, p))
                if ds == 'genai': r.setdefault('asst_f1', []).append(mf1(ya, p // nact))
                r.setdefault('sess_vote', []).append(sess_vote(ya, p // nact, inv, nS, NA))
                if rid is not None:
                    rec = LOCK.get(rid); ours = mf1(yt, p)
                    repro.append(dict(run=rid, here=round(ours, 4), locked=None if rec is None else round(100 * rec, 4),
                                      abs_diff=None if rec is None else round(abs(ours - 100 * rec), 6)))
                log(f'{ds} s{s} {setting:8s} {obs:26s} task {r["task_f1"][-1]:5.1f}' + (f' asst {r["asst_f1"][-1]:5.1f}' if ds == 'genai' else '') + f' sess {r["sess_vote"][-1]:5.1f}')
        res['references'] = {k: (summ(v) if k.startswith(('uniform', 'majority')) else v) for k, v in refs.items()}
        res['observers'] = {}
        for (setting, obs), r in acc.items():
            res['observers'].setdefault(setting, {})[obs] = {k: summ(v) for k, v in r.items()}
        res['reproduction_of_locked_runs'] = dict(max_abs_diff_pp=max((x['abs_diff'] or 0) for x in repro), runs=repro)
        OUT[ds] = res
        write_json({**OUT, **{k: v for k, v in read_json().items() if k not in OUT}})   # keep keys this mode does not produce
        log(f'{ds} done; written results/whole_connection_defences.json')
        # compact summary
        for setting, d_ in res['observers'].items():
            for obs, m in d_.items():
                log(f'  {ds} {setting:8s} {obs:26s} task {m["task_f1"]["mean"]:5.1f} [{m["task_f1"]["min"]:.1f}-{m["task_f1"]["max"]:.1f}]'
                    + (f' asst {m["asst_f1"]["mean"]:5.1f} [{m["asst_f1"]["min"]:.1f}-{m["asst_f1"]["max"]:.1f}]' if 'asst_f1' in m else '')
                    + f' sess {m["sess_vote"]["mean"]:5.1f} [{m["sess_vote"]["min"]:.1f}-{m["sess_vote"]["max"]:.1f}]')
        log('  refs', json.dumps({k: (v['mean'] if isinstance(v, dict) else v) for k, v in res['references'].items()}))


if __name__ == '__main__':
    main()
