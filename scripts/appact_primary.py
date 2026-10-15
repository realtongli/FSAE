"""Statistics of the third evaluation set (the nine MIRAGE-AppAct-2024 apps absent from CCMA).
Writes results/appact/appact_primary.json, per_draw_scores.csv and appact_summary.md.

What is computed:

R1  Metric: test macro-F1 (%) of the nine apps over all 5,044 test connections, classes present in truth or prediction
    (boot2.macro_f1_cm = sklearn f1_score(average='macro')), from <RES>/runs/<run>/test_preds.npz, asserted aligned with the
    split and equal to the macro_f1 of the run's record in <RES>/locked_test_metrics_appact.jsonl.
R2  Draws: configs/splits/appact_fewshot_k{K}_s{r}.json; K = 1, 2: r = 0..19; K = 4, 8: r = 0..4. Pre-training seed r mod 3.
R3  Observers: encT (corpus = the 28,394 GenAI and CCMA connections + AppAct training and validation connections), scratch
    (same network, no pre-training), lgbm (1,600 rounds), knn10 (1-NN, L1, first 10 packets, recomputed here), encX (the
    encoders pre-trained on GenAI and CCMA only, which never saw traffic of these apps).
R4  FAMILY (4 comparisons, claim A > B): encT - lgbm and encT - scratch at K = 1 and K = 2. Paired t over the 20 per-draw
    differences (df 19), two-sided, Holm over the four p-values at family-wise 0.05: a claim holds iff its adjusted p < 0.05
    and the mean is positive. Reported alongside: two-level paired bootstrap boot2.Cell(R=20, B=4000, RandomState(20260926)),
    95% percentile interval, two-sided bootstrap p, Holm over the four.
R5  Outside the family, descriptive, unadjusted: every other comparison at K = 1, 2 (20 draws) and all comparisons at
    K = 4, 8 (5 draws, two-level interval); session-vote accuracy (np.bincount(minlength=9).argmax(), ties to the lowest class
    index; ties counted, 'strict' counts a tied session as wrong); per-class F1; the app x phone session tables; the
    unseen-phone evaluation (K = 4, 5 draws, phones 1e, 2c, f4)."""
import collections, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
RES = os.path.join(ROOT, os.environ.get('APPACT_RES', os.path.join('results', 'appact'))); RUNS = os.path.join(RES, 'runs'); SPL = os.path.join(ROOT, 'configs', 'splits')
N_DRAWS = {1: 20, 2: 20, 4: 5, 8: 5}; NC = 9; ALPHA = 0.05
PSEED = lambda r: r % 3
RUN = {'encT': lambda K, r: f'appact_fs{K}_sslXLT_d{r}_p{PSEED(r)}', 'encX': lambda K, r: f'appact_fs{K}_sslXLX_d{r}_p{PSEED(r)}',
       'scratch': lambda K, r: f'appact_fs{K}_sslscratchXL_d{r}', 'lgbm': lambda K, r: f'appact_fs{K}_lgbm_meta64_s{r}'}
CD_RUN = {'encT': lambda d, r: f'appact_cd{d}_k4_sslXLT_d{r}_p0', 'scratch': lambda d, r: f'appact_cd{d}_k4_sslscratchXL_d{r}', 'lgbm': lambda d, r: f'appact_cd{d}_k4_lgbm_meta64_s{r}'}


def cm_of(y, p):
    cm = np.zeros((NC, NC), np.int64); np.add.at(cm, (y, p), 1); return cm


def class_f1(cm):
    tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0) * 100


def vote(y, p, inv, nS):
    cnt = np.zeros((nS, NC), np.int64); np.add.at(cnt, (inv, p), 1); tru = np.zeros((nS, NC), np.int64); np.add.at(tru, (inv, y), 1)
    v, t = cnt.argmax(1), tru.argmax(1); tie = (cnt == cnt.max(1, keepdims=True)).sum(1) > 1
    return float(100 * np.mean(v == t)), float(100 * np.mean((v == t) & ~tie)), int(tie.sum())


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from appact_wrap import AppActData, NINE, BASE_SPLIT
    from trivial_baselines import feats_knn
    import boot2
    from draws20_primary import holm, tstat, boot_compare

    D = AppActData(ROOT); y_all = D.y_app; Fk = feats_knn(D.meta, D.meta_len)
    last = {}
    for line in open(os.path.join(RES, 'locked_test_metrics_appact.jsonl'), encoding='utf-8'):
        rec = json.loads(line); last[rec['run_id']] = rec
    OUT = dict(rules=__doc__, classes=NINE, cells={}, family=[], outside_family=[], unseen_phone={}, tables={}, missing=[], checks={})
    rows = []

    def load(run, te, yte):
        z = np.load(os.path.join(RUNS, run, 'test_preds.npz'))
        assert np.array_equal(z['idx'], te) and np.array_equal(z['y'], yte), run
        f = 100 * boot2.macro_f1_cm(cm_of(yte, z['pred']))
        assert abs(f - 100 * last[run]['test']['macro_f1']) < 1e-6, (run, f, last[run]['test']['macro_f1'])
        return z['pred']

    def knn(tr, te):
        p = np.empty(len(te), int)
        for b in range(0, len(te), 512): p[b:b + 512] = y_all[tr][np.abs(Fk[te][b:b + 512, None, :] - Fk[tr][None]).sum(-1).argmin(1)]
        return p

    def have(fn, *a): return os.path.exists(os.path.join(RUNS, fn(*a), 'test_preds.npz'))

    # ---------------------------------------------------------------- tables: sessions by app x phone
    ses = D.sessions[D.sessions.app.isin(NINE) & (D.sessions.n_kept > 0)]; sp = json.load(open(os.path.join(ROOT, BASE_SPLIT)))
    tab = lambda s: s.assign(phone=s.device.str[:2]).groupby(['app', 'phone']).size().unstack(fill_value=0)
    T = {'all': tab(ses)}
    for part in ('train', 'val', 'test'): T[part] = tab(ses[ses.file.isin(sp[part])]).reindex(index=T['all'].index, columns=T['all'].columns, fill_value=0)
    OUT['tables']['sessions_app_x_phone'] = {k: {a: {c: int(v.loc[a, c]) for c in v.columns} for a in v.index} for k, v in T.items()}

    # ---------------------------------------------------------------- few-shot cells
    fam, outs = [], []
    for K, R in N_DRAWS.items():
        P = collections.defaultdict(list); runs = collections.defaultdict(list); te0 = va0 = None
        names = ['encT', 'scratch', 'lgbm'] + (['encX'] if all(have(RUN['encX'], K, r) for r in range(R)) else [])
        if any(not have(RUN[a], K, r) for a in ('encT', 'scratch', 'lgbm') for r in range(R)):
            assert K not in (1, 2), f'K={K}: runs of the primary family are missing'
            OUT['missing'].append(f'K={K}'); continue
        if 'encX' not in names: OUT['missing'].append(f'K={K} encX')
        for r in range(R):
            idx, _ = D.split_indices(os.path.join(SPL, f'appact_fewshot_k{K}_s{r}.json')); tr, te, va = idx['train'], idx['test'], idx['val']
            if te0 is None: te0, va0 = te, va
            assert np.array_equal(te0, te) and np.array_equal(va0, va), f'K={K} r={r}: val/test differ'
            for a in names: P[a].append(load(RUN[a](K, r), te, y_all[te])); runs[a].append(RUN[a](K, r))
            P['knn10'].append(knn(tr, te)); runs['knn10'].append('recomputed 1-NN (L1, first 10 packets)')
        names = names + ['knn10']; y = y_all[te0]; sid = D.session_id[te0]; us, inv = np.unique(sid, return_inverse=True); nS = len(us)
        cell = boot2.Cell(y, sid, NC, R=R); obs = {}
        for a in names:
            cell.add(a, P[a]); cms = [cm_of(y, p) for p in P[a]]; pc = np.stack([class_f1(c) for c in cms]); vs = [vote(y, p, inv, nS) for p in P[a]]
            obs[a] = dict(runs=runs[a], **cell.f1(a), per_class_f1={NINE[c]: float(pc[:, c].mean()) for c in range(NC)},
                          session_vote=dict(mean=float(np.mean([v[0] for v in vs])), sd=float(np.std([v[0] for v in vs])), per_draw=[v[0] for v in vs],
                                            strict_mean=float(np.mean([v[1] for v in vs])), ties_per_draw=[v[2] for v in vs]))
            for r in range(R):
                rows.append(dict(K=K, draw=r, pretraining_seed=PSEED(r) if a in ('encT', 'encX') else '', observer=a, run_id=runs[a][r], macro_f1=round(float(cell.F[a]['full'][r]), 4),
                                 session_vote_acc=round(vs[r][0], 4), session_vote_strict=round(vs[r][1], 4), tied_sessions=vs[r][2]))
        OUT['cells'][f'K{K}'] = dict(draws=R, n_test_sessions=int(nS), n_test_connections=int(len(y)), observers=obs)

        def comp(a, b, family, claim):
            c = boot_compare(cell, a, b); per = np.array(c['per_draw'])
            bys = {int(p): float(per[[i for i in range(R) if PSEED(i) == p]].mean()) for p in (0, 1, 2) if any(PSEED(i) == p for i in range(R))}
            return dict(key=f'K{K}|{a}-{b}', K=K, A=a, B=b, family=family, claim=claim, draws=R, diff=c['diff'], per_draw=c['per_draw'], n_pos=c['n_pos'], n_neg=c['n_neg'],
                        sd_draws=float(per.std(ddof=1)), t=tstat(per), boot=dict(ci=c['ci'], p=c['boot_p'], n_resamples=c['n_resamples']), mean_by_pretraining_seed=bys, n_test_sessions=c['n_test_sessions'])
        if K in (1, 2):
            fam.append(comp('encT', 'lgbm', f'A{K}', 'A>B')); fam.append(comp('encT', 'scratch', f'A{K + 2}', 'A>B'))
        else:
            outs.append(comp('encT', 'lgbm', 'outside', None)); outs.append(comp('encT', 'scratch', 'outside', None))
        outs.append(comp('encT', 'knn10', 'outside', None))
        if 'encX' in names:
            for b in ('lgbm', 'scratch', 'encT'): outs.append(comp('encX', b, 'outside', None))
        print(f'K={K}: {R} draws, {nS} test sessions; ' + ', '.join(f'{a} {obs[a]["mean"]:.2f}' for a in names), flush=True)

    fam.sort(key=lambda r: r['family'])
    adj_t = holm([r['t']['p'] for r in fam]); adj_b = holm([r['boot']['p'] for r in fam])
    for r, at, ab in zip(fam, adj_t, adj_b):
        r['t']['p_holm'] = float(at); r['boot']['p_holm'] = float(ab)
        r['holds_primary'] = bool(at < ALPHA and r['diff'] > 0); r['holds_bootstrap'] = bool(ab < ALPHA and r['diff'] > 0)
        r['significantly_below'] = bool(at < ALPHA and r['diff'] < 0); r['boot_ci_excludes_zero'] = bool(r['boot']['ci'][0] > 0 or r['boot']['ci'][1] < 0)
    OUT['family'] = fam; OUT['outside_family'] = outs

    # ---------------------------------------------------------------- unseen phone (K = 4, 5 draws)
    for d in ('1e', '2c', 'f4'):
        if any(not have(CD_RUN[a], d, r) for a in CD_RUN for r in range(5)): OUT['missing'].append(f'unseen phone {d}'); continue
        P = collections.defaultdict(list); te0 = None
        for r in range(5):
            idx, meta = D.split_indices(os.path.join(SPL, f'appact_cd_{d}_k4_s{r}.json')); tr, te = idx['train'], idx['test']
            if te0 is None: te0 = te
            assert np.array_equal(te0, te)
            for a in CD_RUN: P[a].append(load(CD_RUN[a](d, r), te, y_all[te]))
            P['knn10'].append(knn(tr, te))
        y = y_all[te0]; sid = D.session_id[te0]; us, inv = np.unique(sid, return_inverse=True); nS = len(us); cell = boot2.Cell(y, sid, NC, R=5); o = {}
        for a in list(CD_RUN) + ['knn10']:
            cell.add(a, P[a]); vs = [vote(y, p, inv, nS) for p in P[a]]
            o[a] = dict(**cell.f1(a), session_vote=dict(mean=float(np.mean([v[0] for v in vs])), per_draw=[v[0] for v in vs], strict_mean=float(np.mean([v[1] for v in vs])), ties_per_draw=[v[2] for v in vs]))
            for r in range(5): rows.append(dict(K=f'cd_{d}_k4', draw=r, pretraining_seed=0 if a == 'encT' else '', observer=a, run_id=CD_RUN[a](d, r) if a in CD_RUN else 'recomputed 1-NN',
                                                macro_f1=round(float(cell.F[a]['full'][r]), 4), session_vote_acc=round(vs[r][0], 4), session_vote_strict=round(vs[r][1], 4), tied_sessions=vs[r][2]))
        c = cell.compare('encT', 'lgbm'); c2 = cell.compare('encT', 'scratch')
        OUT['unseen_phone'][d] = dict(classes=meta['classes'], n_test_sessions=int(nS), n_test_connections=int(len(y)), observers=o,
                                      encT_minus_lgbm=dict(diff=c['diff'], ci=c['ci'], per_draw=c['per_draw'], verdict=c['verdict']), encT_minus_scratch=dict(diff=c2['diff'], ci=c2['ci'], per_draw=c2['per_draw'], verdict=c2['verdict']))
        print(f'unseen phone {d}: {nS} test sessions; ' + ', '.join(f'{a} {o[a]["mean"]:.2f} (vote {o[a]["session_vote"]["mean"]:.1f})' for a in o), flush=True)

    json.dump(OUT, open(os.path.join(RES, 'appact_primary.json'), 'w'), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'per_draw_scores.csv'), index=False)

    # ---------------------------------------------------------------- summary (markdown) and console
    L = ['# Third evaluation set: summary', '', 'Source: `results/appact/appact_primary.json`.', '']
    L += ['## Family (4 comparisons, 20 draws, paired t df 19, Holm over 4; two-level bootstrap alongside)', '',
          '| id | comparison | K | diff | positive draws | t 95% | p | Holm p | bootstrap 95% | boot Holm p | claim A>B |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in fam:
        L.append(f"| {r['family']} | {r['A']} - {r['B']} | {r['K']} | {r['diff']:+.2f} | {r['n_pos']}/{r['draws']} | [{r['t']['ci'][0]:+.2f}, {r['t']['ci'][1]:+.2f}] | {r['t']['p']:.4f} | {r['t']['p_holm']:.4f} | "
                 f"[{r['boot']['ci'][0]:+.2f}, {r['boot']['ci'][1]:+.2f}] | {r['boot']['p_holm']:.4f} | {'holds' if r['holds_primary'] else ('BELOW' if r['significantly_below'] else 'does not hold')} (bootstrap: {'holds' if r['holds_bootstrap'] else 'does not hold'}) |")
    L += ['', '## Macro-F1 (%) and session-vote accuracy (%), mean over draws (SD)', '', '| K | draws | observer | macro-F1 | min-max | session vote | strict |', '|---|---|---|---|---|---|---|']
    for k, c in OUT['cells'].items():
        for a, o in c['observers'].items():
            L.append(f"| {k[1:]} | {c['draws']} | {a} | {o['mean']:.2f} ({o['sd']:.2f}) | {o['min']:.1f}-{o['max']:.1f} | {o['session_vote']['mean']:.1f} ({o['session_vote']['sd']:.1f}) | {o['session_vote']['strict_mean']:.1f} |")
    L += ['', '## Outside the family (descriptive, unadjusted)', '', '| comparison | K | draws | diff | positive | t 95% | t p | two-level 95% |', '|---|---|---|---|---|---|---|---|']
    for r in outs:
        L.append(f"| {r['A']} - {r['B']} | {r['K']} | {r['draws']} | {r['diff']:+.2f} | {r['n_pos']}/{r['draws']} | [{r['t']['ci'][0]:+.2f}, {r['t']['ci'][1]:+.2f}] | {r['t']['p']:.4f} | [{r['boot']['ci'][0]:+.2f}, {r['boot']['ci'][1]:+.2f}] |")
    L += ['', '## Per-class F1 (%), mean over draws', '', '| K | observer | ' + ' | '.join(NINE) + ' |', '|---|---|' + '---|' * NC]
    for k, c in OUT['cells'].items():
        for a, o in c['observers'].items(): L.append(f"| {k[1:]} | {a} | " + ' | '.join(f"{o['per_class_f1'][n]:.1f}" for n in NINE) + ' |')
    L += ['', '## Unseen phone (K = 4, 5 draws)', '', '| phone | classes | test sessions | observer | macro-F1 | session vote | encT - lgbm (two-level 95%) |', '|---|---|---|---|---|---|---|']
    for d, u in OUT['unseen_phone'].items():
        for a, o in u['observers'].items():
            L.append(f"| {d} | {len(u['classes'])} | {u['n_test_sessions']} | {a} | {o['mean']:.2f} ({o['sd']:.2f}) | {o['session_vote']['mean']:.1f} | " + (f"{u['encT_minus_lgbm']['diff']:+.2f} [{u['encT_minus_lgbm']['ci'][0]:+.2f}, {u['encT_minus_lgbm']['ci'][1]:+.2f}]" if a == 'encT' else '') + ' |')
    for k, v in T.items():
        L += ['', f'## Sessions by app x phone: {k}', '', '| app | ' + ' | '.join(v.columns) + ' | total |', '|---|' + '---|' * (len(v.columns) + 1)]
        L += [f'| {a} | ' + ' | '.join(str(int(v.loc[a, c])) for c in v.columns) + f' | {int(v.loc[a].sum())} |' for a in v.index]
        L.append('| total | ' + ' | '.join(str(int(v[c].sum())) for c in v.columns) + f' | {int(v.values.sum())} |')
    if OUT['missing']: L += ['', '## Missing runs', ''] + [f'- {m}' for m in OUT['missing']]
    open(os.path.join(RES, 'appact_summary.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwritten', os.path.join(RES, 'appact_primary.json'), 'per_draw_scores.csv', 'appact_summary.md')


if __name__ == '__main__':
    main()
