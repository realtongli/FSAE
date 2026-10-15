"""Whole-connection FRONT at the published padding budgets (appendix on porting the baselines: with 1,700 dummies per
direction the session vote over whole connections moves by at most one point).

Why. scripts/wf_defenses_whole.front_whole simulates FRONT over a whole connection with dummy budgets N_c = 700
  (client) and N_s = 1400 (server), capped at the connection's real packet count. Gong and Wang (USENIX Security 2020,
  Table 3) evaluate FT-1 with N_s = N_c = 1700 and FT-2 with N_s = N_c = 2500, both with W_min = 1 s and W_max = 14 s.
  The cap makes the budget irrelevant for a connection of at most 700 packets, so the first-64-packet results cannot
  change; only connections longer than the budget receive more dummies under the published values.
Configurations, all in ONE pass over the raw JSON with the same code path (WW.front_whole, only N_c / N_s differ):
  paper   N_c = 700,  N_s = 1400   the configuration of wf_defenses_whole.py; must reproduce its stored numbers
  FT-1    N_c = N_s = 1700         the published configuration of the appendix
  FT-2    N_c = N_s = 2500         the heavier published configuration
  Everything else is unchanged: budget capped at the connection's real packet count, window w ~ U(1, 14) s scaled by
  the connection's duration over 14 s, Rayleigh(w) dummy times clipped to the connection, each dummy copies the IP and
  payload size of a random real packet of its direction, random stream seeded per (seed, row) exactly as WW.extract.
Observer and metrics as scripts/whole_connection_defences.py (same functions, imported): LightGBM with the fixed recipe
  on the nine whole-connection observables W, K = 4 labelled sessions per class, seeds 0-4, splits
  [ccma_]fewshot_k4_s{s}; (i) adaptive W = trained and tested on defended traffic, (iii) unaware W = trained on
  undefended W, tested on defended W; reference 'none' = undefended W. Task macro-F1, GenAI assistant macro-F1, session
  vote of the assistant (GenAI) or app (CCMA). Mean and min-max over the five seeds.
Costs over every labelled flow, per seed: dummy bytes over real IP bytes (aggregate and per-flow median) and dummies per
  real packet. Also reported: the share of labelled flows, and of their real bytes, with more than 700 / 1400 / 1700 /
  2500 real packets (the flows whose budget binds).
Consistency check: configuration 'paper' must equal results/whole_connection_defences.json (observers.front '(i) W
  adaptive' and '(iii) W unaware', observers.none 'W', overheads_whole.front_whole) to the stored rounding.
Writes results/front_whole_published.json only. No other result file is read for selection or modified."""
import json, os, sys, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from data_mfr import GenAIData
import wf_defenses_whole as WW
from whole_connection_defences import lgbm_predict, mf1, sess_vote, summ, SEEDS

CFG = {'paper_700_1400': (700, 1400), 'FT1_1700': (1700, 1700), 'FT2_2500': (2500, 2500)}
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def extract(ds):
    index = pd.read_csv(os.path.join(ROOT, 'data', 'derived', f'{ds}_index.csv'))
    sessions = pd.read_csv(os.path.join(ROOT, 'data', 'derived', f'{ds}_sessions.csv'))
    N = len(index); S = len(SEEDS)
    R = dict(W_raw=np.zeros((N, 9)), real_bytes=np.zeros(N), real_pkts=np.zeros(N), got=np.zeros(N, bool))
    for c in CFG: R[f'W_{c}'] = np.zeros((S, N, 9)); R[f'db_{c}'] = np.zeros((S, N)); R[f'dp_{c}'] = np.zeros((S, N))
    rows_by_sid = {sid: g for sid, g in index.reset_index().groupby('session_id')}
    for k, (sid, js) in enumerate(WW._iter_sessions(ROOT, ds, sessions)):
        if sid not in rows_by_sid: continue
        for row, key in zip(rows_by_sid[sid]['index'].values, rows_by_sid[sid]['biflow_key'].values):
            ts, d, ip, pay, iat = WW._flow_arrays(js[key]); t = ts - ts[0]
            R['W_raw'][row] = WW.raw_whole(t, d, ip, pay); R['real_bytes'][row] = float(ip.sum()); R['real_pkts'][row] = len(t)
            for si, s in enumerate(SEEDS):
                for c, (nc, ns) in CFG.items():
                    rng = np.random.RandomState((1_000_003 * (40_000 + s) + row) % (2 ** 32 - 1))   # as WW.extract
                    fw = WW.front_whole(t, d, ip, pay, rng, N_c=nc, N_s=ns)
                    R[f'W_{c}'][si, row] = fw['W']; R[f'db_{c}'][si, row] = fw['dummy_bytes']; R[f'dp_{c}'][si, row] = fw['dummy_pkts']
            R['got'][row] = True
        if k % 25 == 0: log(f'{ds}: session {k + 1}/{len(sessions)}, flows done {int(R["got"].sum())}/{N}')
        del js
    return R


def main():
    OUT = dict(rules=__doc__.split('Writes results')[0].strip(), configurations={c: dict(N_c=v[0], N_s=v[1], W_min=1.0, W_max=14.0) for c, v in CFG.items()},
               seeds=SEEDS, K=4)
    stored = json.load(open(os.path.join(ROOT, 'results', 'whole_connection_defences.json')))
    for ds in ('genai', 'ccma'):
        D = GenAIData(ROOT, prefix=ds); y = D.y_joint if ds == 'genai' else D.y_app; NC = D.n_joint if ds == 'genai' else D.n_app
        nact = 2 if ds == 'genai' else 1; NA = NC // nact; lab = y >= 0
        R = extract(ds); assert R['got'].all()
        real = R['real_bytes'][lab]; n = R['real_pkts'][lab]
        res = dict(n_labelled=int(lab.sum()),
                   flows_longer_than={str(b): dict(share_of_flows=round(float((n > b).mean()), 4), share_of_real_bytes=round(float(real[n > b].sum() / real.sum()), 4))
                                      for b in (700, 1400, 1700, 2500)}, costs={}, observers={})
        for c in CFG:
            fb = R[f'db_{c}'][:, lab]; fp = R[f'dp_{c}'][:, lab]
            res['costs'][c] = dict(dummy_bytes_aggregate=summ(fb.sum(1) / real.sum()), dummy_bytes_median_per_flow=summ(np.median(fb / np.maximum(real, 1)[None], axis=1)),
                                   dummies_per_real_packet=summ(fp.sum(1) / n.sum()))
            log(ds, c, 'costs', json.dumps({k: v['mean'] for k, v in res['costs'][c].items()}))
        split = 'fewshot_k4' if ds == 'genai' else 'ccma_fewshot_k4'; acc = {}
        for s in SEEDS:
            idx, _ = D.split_indices(os.path.join(ROOT, 'configs', 'splits', f'{split}_s{s}.json'))
            tr, it = idx['train'], idx['test']; yt = y[it]; ya = yt // nact
            us, inv = np.unique(D.session_id[it], return_inverse=True); nS = len(us)
            OBS = [('none', 'W', R['W_raw'], R['W_raw'])]
            for c in CFG:
                Wf = R[f'W_{c}'][s]; OBS += [(c, '(i) W adaptive', Wf, Wf), (c, '(iii) W unaware', R['W_raw'], Wf)]
            for setting, obs, Xtr, Xte in OBS:
                p = lgbm_predict(Xtr[tr], y[tr], Xte[it], s); r = acc.setdefault((setting, obs), {})
                r.setdefault('task_f1', []).append(mf1(yt, p))
                if ds == 'genai': r.setdefault('asst_f1', []).append(mf1(ya, p // nact))
                r.setdefault('sess_vote', []).append(sess_vote(ya, p // nact, inv, nS, NA))
            log(f'{ds} seed {s} done')
        for (setting, obs), r in acc.items(): res['observers'].setdefault(setting, {})[obs] = {k: summ(v) for k, v in r.items()}
        # consistency with the stored whole-connection FRONT (configuration 'paper')
        st = stored[ds]; chk = {}
        for obs in ('(i) W adaptive', '(iii) W unaware'):
            for m, v in st['observers']['front'][obs].items(): chk[f'front {obs} {m}'] = [res['observers']['paper_700_1400'][obs][m]['mean'], v['mean']]
        for m, v in st['observers']['none']['W'].items(): chk[f'none W {m}'] = [res['observers']['none']['W'][m]['mean'], v['mean']]
        for m in ('dummy_bytes_aggregate', 'dummy_bytes_median_per_flow', 'dummies_per_real_packet'):
            chk[f'cost {m}'] = [res['costs']['paper_700_1400'][m]['mean'], st['overheads_whole']['front_whole'][m]['mean']]
        res['check_paper_configuration_[here, stored]'] = chk; res['check_max_abs_diff'] = max(abs(a - b) for a, b in chk.values())
        log(ds, 'consistency max |diff| =', res['check_max_abs_diff'])
        OUT[ds] = res
        json.dump(OUT, open(os.path.join(ROOT, 'results', 'front_whole_published.json'), 'w'), indent=1)
        for setting, d_ in res['observers'].items():
            for obs, m in d_.items():
                log(f'  {ds} {setting:15s} {obs:16s} task {m["task_f1"]["mean"]:5.1f}' + (f' asst {m["asst_f1"]["mean"]:5.1f}' if 'asst_f1' in m else '')
                    + f' sess {m["sess_vote"]["mean"]:5.1f} [{m["sess_vote"]["min"]:.1f}-{m["sess_vote"]["max"]:.1f}]')
    log('written results/front_whole_published.json')


if __name__ == '__main__':
    main()
