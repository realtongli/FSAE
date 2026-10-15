"""Leak checks of the third evaluation set, to run before its pre-training (scripts/run_paper.sh, block 8).
No model is trained and no metric is computed. Output: logs/appact_leakcheck.log. Exit code 1 on any hard failure.

  1. the AppAct corpus sessions (training + validation of the base split) and the 163 test sessions are disjoint   [hard]
  2. no capture timestamp of any corpus session (GenAI, CCMA, AppAct) equals that of an AppAct test session          [hard]
  3. every K-session draw labels training sessions only; validation and test are the same in every draw             [hard]
  4. test connections whose first-64-packet metadata equals a corpus connection byte for byte, by length;
     with five or more packets there must be none                                                                    [hard for >= 5]
  5. the AppAct captures that start within 15 minutes of a CCMA capture, with the split part each falls into         [report]
  6. unseen-phone corpora: the exclusion lists hold every session of the held-out phone (AppAct and CCMA)            [hard]"""
import collections, hashlib, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
from appact_wrap import AppActData, Data, NINE, BASE_SPLIT

SPL = os.path.join(ROOT, 'configs', 'splits'); LOG = os.path.join(ROOT, 'logs', 'appact_leakcheck.log')
FAIL = []
_lines = []


def say(s=''):
    print(s, flush=True); _lines.append(s)


def check(ok, text):
    say(('PASS  ' if ok else 'FAIL  ') + text)
    if not ok: FAIL.append(text)


def hashes(meta, n, idx):
    out = []
    for i in idx:
        k = int(min(n[i], 64)); out.append(hashlib.md5(np.ascontiguousarray(meta[i, :k]).tobytes()).hexdigest() + f'_{k}')
    return np.array(out)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    D = AppActData(ROOT); sp = json.load(open(os.path.join(ROOT, BASE_SPLIT))); ses = D.sessions
    f2ts = dict(zip(ses.file, ses.session_ts.astype(int))); f2dev = dict(zip(ses.file, ses.device)); f2app = dict(zip(ses.file, ses.app))
    corpus_f, test_f = set(sp['train'] + sp['val']), set(sp['test'])
    say(f'AppAct base split {sp["meta"]["hash"]} {sp["meta"]["counts"]}')
    # 1
    idx, _ = D.split_indices(os.path.join(ROOT, BASE_SPLIT)); cidx = np.concatenate([idx['train'], idx['val']]); tidx = idx['test']
    check(not (corpus_f & test_f) and not (set(D.session_id[cidx]) & set(D.session_id[tidx])), f'1. corpus sessions ({len(corpus_f)}) and test sessions ({len(test_f)}) are disjoint; corpus {len(cidx)} connections, test {len(tidx)}')
    check(len(cidx) == 19614 and len(tidx) == 5044 and set(f2app[f] for f in corpus_f | test_f) == set(NINE), '1. corpus 19,614 and test 5,044 connections, nine apps only')
    # 2
    test_ts = set(f2ts[f] for f in test_f); corp_ts = {'appact': set(f2ts[f] for f in corpus_f)}; other = {}
    for ds, split in (('genai', 'session_split_s2026.json'), ('ccma', 'ccma_session_split_s2026.json')):
        G = Data(ROOT, ds); s2 = json.load(open(os.path.join(SPL, split))); keep = set(s2['train'] + s2['val'])
        corp_ts[ds] = set(G.sessions[G.sessions.file.isin(keep)].session_ts.astype(int))
        sids = np.array([G.file2sid[f] for f in sorted(keep)]); other[ds] = (G, np.where(np.isin(G.session_id, sids))[0])
    for ds, ts in corp_ts.items():
        check(not (ts & test_ts), f'2. capture timestamps: {ds} corpus sessions ({len(ts)}) vs AppAct test sessions ({len(test_ts)}): {len(ts & test_ts)} shared')
    say(f'   existing corpus connections: genai {len(other["genai"][1])} + ccma {len(other["ccma"][1])} = {len(other["genai"][1]) + len(other["ccma"][1])}')
    check(len(other['genai'][1]) + len(other['ccma'][1]) == 28394, '2. existing corpus has 28,394 connections')
    # 3
    ok = True; n = 0
    for K, R in ((1, 20), (2, 20), (4, 5), (8, 5)):
        for r in range(R):
            d = json.load(open(os.path.join(SPL, f'appact_fewshot_k{K}_s{r}.json'))); n += 1
            ok &= set(d['train']) <= set(sp['train']) and d['val'] == sp['val'] and d['test'] == sp['test'] and len(d['train']) == 9 * K
    check(ok, f'3. all {n} K-session draws label training sessions only (9K each); validation and test identical to the base split')
    # 4
    ht = hashes(D.meta, D.meta_len, tidx); lt = np.minimum(D.meta_len[tidx], 64)
    pools = {'appact corpus': set(hashes(D.meta, D.meta_len, cidx))}
    for ds, (G, gi) in other.items(): pools[f'{ds} corpus'] = set(hashes(G.meta, G.meta_len, gi))
    for name, pool in pools.items():
        dup = np.array([h in pool for h in ht]); by = collections.Counter('1' if l == 1 else ('2-4' if l < 5 else '>=5') for l in lt[dup])
        say(f'   test connections identical to a connection of the {name}: {int(dup.sum())} of {len(ht)}; by packets {dict(by)}')
        check(by.get('>=5', 0) == 0, f'4. no test connection with >= 5 packets is identical to a connection of the {name}')
    # 5
    cs = pd.read_csv(os.path.join(ROOT, 'data', 'derived', 'ccma_sessions.csv')); cts = np.sort(cs.session_ts.values.astype(np.int64))
    part = {f: p for p in ('train', 'val', 'test') for f in sp[p]}; near = collections.Counter()
    for f, ts in f2ts.items():
        if np.abs(cts - ts).min() < 900: near[(f2app[f], f2dev[f][:2], part.get(f, 'not in the nine-app split'))] += 1
    say(f'5. captures starting within 15 minutes of a CCMA capture: {sum(near.values())}; (app, phone, part): {dict(near)}')
    # 6
    s9 = ses[ses.app.isin(NINE) & (ses.n_kept > 0)]; cdev = dict(zip(cs.file, cs.device.astype(str)))
    for tag, ccma_ex in (('1e', None), ('2c', 'ccma_cd_2c_k4_s0.json'), ('f4', 'ccma_cd_f4_k4_s0.json')):
        ex = set(json.load(open(os.path.join(SPL, f'appact_cdexcl_{tag}.json')))['test']); on = set(s9[s9.device.str.startswith(tag)].file)
        tests = [set(json.load(open(os.path.join(SPL, f'appact_cd_{tag}_k4_s{r}.json')))['test']) for r in range(5)]
        trains = [set(json.load(open(os.path.join(SPL, f'appact_cd_{tag}_k4_s{r}.json')))['train']) for r in range(5)]
        ok = ex == on and all(t == tests[0] and t <= ex for t in tests) and all(not (t & ex) for t in trains)
        left = (corpus_f - ex); ok &= not any(f2dev[f].startswith(tag) for f in left)
        msg = f'6. phone {tag}: exclusion list = all {len(on)} nine-app sessions on the phone; test {len(tests[0])} sessions; labelled sessions on other phones; AppAct corpus keeps {len(left)} sessions, none on the phone'
        if ccma_ex:
            cex = set(json.load(open(os.path.join(SPL, ccma_ex)))['test']); s2 = json.load(open(os.path.join(SPL, 'ccma_session_split_s2026.json'))); ckeep = set(s2['train'] + s2['val']) - cex
            ok &= not any(cdev[f].startswith(tag) for f in ckeep); msg += f'; CCMA corpus keeps {len(ckeep)} sessions, none on the phone'
        else:
            ok &= not any(d.startswith(tag) for d in cdev.values()); msg += '; the phone does not occur in CCMA'
        check(ok, msg)
    say(); say('LEAK CHECK ' + ('FAILED: ' + '; '.join(FAIL) if FAIL else 'PASSED (all hard checks)'))
    os.makedirs(os.path.dirname(LOG), exist_ok=True); open(LOG, 'w', encoding='utf-8').write('\n'.join(_lines) + '\n')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
