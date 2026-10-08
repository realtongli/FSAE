"""Main-text result figures (Linux Libertine, ACM print sizes, Okabe-Ito colours).

  fig_fewshot_main.pdf     (column width) few-shot curves, test macro-F1 vs K = 1, 2, 4, 8 labelled capture sessions per
                           class, GenAI (6 app x modality classes) and CCMA (9 apps); mean over the five K-session draws
                           (seeds 0-4) with +-1 population s.d. over the draws as error bars.
  fig_defence_tradeoff.pdf (column width) defence cost vs remaining leak: x = bytes a defence adds relative to the bytes of
                           what the observer sees (log scale), y = share of test sessions whose assistant (GenAI) / app
                           (CCMA) wins the session vote; chance = uniform guess.

  fig_defence_tradeoff_genai.pdf  the same, panel (a) only (one-panel variant).
Writes the PDFs to paper/figs/ and a PNG preview of each to paper/figs/preview/ (PREVIEW). The Linux Libertine OpenType
fonts are loaded from the directory named by the environment variable LIBERTINE_DIR; without it matplotlib's default serif
font is used. Nothing is computed from raw traffic here: every plotted number is read from a result file, and the script
asserts the consistency checks listed below before drawing. No value is estimated; a point whose value is missing for an
observer is left out.

------------------------------------------------------------------------------------------------------------------------
F1 sources. results/locked_test_metrics.jsonl, test macro-F1, last record per run_id, smoke tags skipped (the rule of
scripts/summarize_sota_fewshot.py); runs <prefix>_s{0..4} with the prefixes of paper/tables/tab_sota_fewshot.tex:
  pre-trained encoder      fs{K}_sslXL            ccma_fs{K}_sslXL
  encoder, no pre-training fs{K}_sslscratchXL     ccma_fs{K}_sslscratchXL
  LightGBM                 fs{K}_lgbm_meta64      ccma_fs{K}_lgbm_meta64
  1-NN, first 10 packets   fs{K}_knn10            ccma_fs{K}_knn10
  NetCLR (DF backbone)     fs{K}_netclr           ccma_fs{K}_netclr
  YaTC (payload bytes)     fs{K}_yatc_pre         ccma_fs{K}_yatc_pre
All five seeds must exist; each plotted mean is asserted equal to the printed cell of tab_sota_fewshot.tex.

------------------------------------------------------------------------------------------------------------------------
F2 sources (per campaign ds in {genai, ccma}; K = 4, seeds 0-4; every y is a seed mean in %).
 y, session vote, 64-packet observers (rule of scripts/robust_assistant.py: majority of the per-connection assistant/app
    decision over a test session's connections, ties to the lowest index):
      LightGBM, encoder  results/robust_assistant.json      [ds][cond]['lgbm'|'ours']['sess_asst_acc'][0]
      1-NN (10 packets)  results/robust_assistant_knn.json  [ds][cond]['knn']['sess_asst_acc']['mean']
    cond: none; ech (ClientHello padded); echfull (server flight padded too); pj_unaware / pj_adaptive (padding to 256 B
    + Exp(20 ms) jitter); front_unaware / front_adaptive; tamaraw_adaptive (published rates 40/12 ms, L = 100).
 y, flow-record LightGBM over whole connections (observer (i): W features, trained on defended traffic):
      Tamaraw  results/whole_connection_defences.json ['tamaraw_published'][ds]['observers']
               ['published L=100: (i) W adaptive']['sess_vote']['mean']   (the published configuration; the Tamaraw
               entries of the default mode and of 'tamaraw_L_grid' in that file use other rates and are not read)
      FRONT    results/whole_connection_defences.json [ds]['observers']['front']['(i) W adaptive']['sess_vote']['mean']
 y, white-box adversarial shaping against the pre-trained encoder (PGD of scripts/adv_eval.py, budgets 64 B/10 ms,
    256 B/50 ms, 1460 B/200 ms per packet), encoder and 1-NN scored on the SAME shaped test connections, both trained on
    unshaped sessions: results/adv_transfer.json ['results'][ds]['wb'][budget]['encoder'|'knn10']['sess_vote']['mean'].
 x, added bytes / observed bytes (aggregate: sum over labelled connections of added bytes / sum of observed IP bytes):
      ECH, ClientHello    results/session_extras.json ['f_overheads'][ds]['ech']['window_clienthello_only']['aggregate']
      ECH, both flights   results/session_extras.json ['f_overheads'][ds]['ech']['window_clienthello_plus_server']['aggregate']
      padding + jitter    results/session_extras.json ['f_overheads'][ds]['pad256_bytes']['aggregate'] (jitter adds no bytes)
                          (only these three costs, the ones quoted in the text, are read from session_extras.json; its
                          Tamaraw session rows are not used here)
      FRONT, 64 packets   results/defence_stats.json [ds]['front']['dummy_bytes_over_real']
      Tamaraw, 64 packets results/defence_stats.json [ds]['tamaraw']['extra_bytes_aggregate'] (asserted: 1500 B, 40/12 ms, L=100)
      Tamaraw, whole      results/whole_connection_defences.json ['tamaraw_published'][ds]['costs']['L=100']['extra_bytes_aggregate']
      FRONT, whole        results/whole_connection_defences.json [ds]['overheads_whole']['front_whole']['dummy_bytes_aggregate']['mean']
      white-box shaping   results/adv_transfer.json ['overheads'][ds]['wb'][budget]['bytes_ratio']['mean']
                          (added bytes / IP bytes of the observed 64-packet prefix)
      no defence          0, drawn in a separate slot left of an axis break.
 Consistency checks asserted before drawing: the LightGBM/encoder votes stored in robust_assistant_knn.json equal those of
 robust_assistant.json; the checks R1_pass, R2_pass and R3_pass stored in adv_transfer.json hold, its unshaped encoder vote
 equals robust_assistant.json's, and its white-box encoder task macro-F1 equals the mean test_f1_adv of the tag-adv_shaping
 white-box records of results/adv_eval.jsonl (standard, not adversarially trained, encoder).
 Markers of one defence are offset horizontally by at most a factor 1.23 (0.09 decade) so that coinciding markers do not
 hide one another completely; the white-box points are offset by at most a factor 1.07. On CCMA, ECH on both flights (cost 0.231) and
 padding + jitter (0.219) would overlap, so their groups are offset in opposite directions, keeping their order (within
 the same factor 1.23). At print size one offset step is about 1.2 pt against 4-pt markers: markers of one defence touch,
 so an individual observer's value is read from the result tables, not from the figure.
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'paper', 'figs')
PREVIEW = os.path.join(ROOT, 'paper', 'figs', 'preview')
os.makedirs(PREVIEW, exist_ok=True)

# ---------- style ----------
LIB = os.environ.get('LIBERTINE_DIR', '')  # directory of the Linux Libertine OpenType fonts (LinLibertine_*.otf)
FAM = 'serif'  # matplotlib's default serif font, kept when LIB is empty, does not exist or holds no LinLibertine_R.otf
if LIB and os.path.isfile(os.path.join(LIB, 'LinLibertine_R.otf')):
    for f in glob.glob(os.path.join(LIB, 'LinLibertine_R*.otf')) + glob.glob(os.path.join(LIB, 'LinLibertine_I.otf')):
        font_manager.fontManager.addfont(f)
    FAM = font_manager.FontProperties(fname=os.path.join(LIB, 'LinLibertine_R.otf')).get_name()
plt.rcParams.update({
    'font.family': FAM, 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'mathtext.fontset': 'custom', 'mathtext.rm': FAM, 'mathtext.it': f'{FAM}:italic', 'mathtext.bf': f'{FAM}:bold',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 600,
    'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.labelpad': 1.0, 'axes.titlepad': 1.0,
})
COLW, TEXTW = 3.33, 7.0
OURS_C, BLUE, LBLUE = '#D55E00', '#0072B2', '#56B4E9'
INK = '#222222'


def save(fig, name):
    # fixed figure size (no bbox cropping), so LaTeX includes it at 100% and fonts print at their nominal size
    fig.savefig(os.path.join(FIGS, name + '.pdf'))
    fig.savefig(os.path.join(PREVIEW, name + '.png'), dpi=300)
    plt.close(fig)


def jload(name): return json.load(open(os.path.join(RES, name), encoding='utf-8'))


# =====================================================================================================================
# F1: few-shot curves
# =====================================================================================================================
T = {}
for l in open(os.path.join(RES, 'locked_test_metrics.jsonl'), encoding='utf-8'):
    j = json.loads(l)
    if str(j.get('tag', '')).startswith('smoke'): continue
    T[j['run_id']] = j['test']['macro_f1']  # last record wins for a run_id


def seeds(prefix):
    v = [T.get(f'{prefix}_s{s}') for s in range(5)]
    assert all(x is not None for x in v), f'missing seed for {prefix}'
    return 100 * np.array(v)


# (label, GenAI prefix, CCMA prefix, colour, marker, linestyle, filled, line width, marker size, z, row label in tab:sota)
SERIES = [
    ('Pre-trained encoder (ours)', 'fs{K}_sslXL', 'ccma_fs{K}_sslXL', OURS_C, 'o', '-', True, 1.6, 4.2, 6, 'Pre-trained metadata encoder (ours)'),
    ('Encoder, no pre-training', 'fs{K}_sslscratchXL', 'ccma_fs{K}_sslscratchXL', OURS_C, 'o', (0, (3, 1.5)), False, 0.9, 3.4, 5, 'Our encoder, no pre-training'),
    ('LightGBM', 'fs{K}_lgbm_meta64', 'ccma_fs{K}_lgbm_meta64', '#1a1a1a', 's', '-', True, 0.9, 3.3, 4, 'LightGBM, packet statistics'),
    ('1-NN, first 10 packets', 'fs{K}_knn10', 'ccma_fs{K}_knn10', '#6a6a6a', '^', (0, (1, 1)), True, 0.9, 3.6, 3, '1-NN on the first 10 packets'),
    ('NetCLR', 'fs{K}_netclr', 'ccma_fs{K}_netclr', '#9a9a9a', 'D', '-.', False, 0.9, 3.0, 2, 'NetCLR~'),
    ('YaTC, payload bytes', 'fs{K}_yatc_pre', 'ccma_fs{K}_yatc_pre', BLUE, 'P', (0, (4, 1.5)), True, 0.9, 3.6, 3, '$\\bullet$ YaTC'),
]
KS = np.array([1, 2, 4, 8])

# the table's printed cells, to assert that the figure plots the same numbers
tab = open(os.path.join(ROOT, 'paper', 'tables', 'tab_sota_fewshot.tex'), encoding='utf-8').read().splitlines()


def table_cells(row_start):
    ln = [x for x in tab if x.startswith(row_start)]
    assert len(ln) == 1, row_start
    cells = ln[0].rstrip('\\').split('&')[1:]
    return [float(re.search(r'(\d+\.\d)', c).group(1)) for c in cells]


F1 = {}
for lab, g, c, *_rest in SERIES:
    row = _rest[-1]
    F1[lab] = {}
    for ds, pre in (('genai', g), ('ccma', c)):
        v = [seeds(pre.format(K=K)) for K in KS]
        F1[lab][ds] = (np.array([x.mean() for x in v]), np.array([x.std() for x in v]))  # population s.d., as the table
    printed = table_cells(row)
    mine = [round(float(m), 1) for m in np.concatenate([F1[lab]['genai'][0], F1[lab]['ccma'][0]])]
    assert mine == printed, (lab, mine, printed)
    print(f'F1 {lab:28s} GenAI', ' '.join(f'{m:5.2f}±{s:4.2f}' for m, s in zip(*F1[lab]['genai'])),
          '| CCMA', ' '.join(f'{m:5.2f}±{s:4.2f}' for m, s in zip(*F1[lab]['ccma'])))
for ds in ('genai', 'ccma'):
    o = F1['Pre-trained encoder (ours)'][ds][0]
    for lab in ('LightGBM', '1-NN, first 10 packets', 'Encoder, no pre-training'):
        print(f'   {ds} ours - {lab:26s}', ' '.join(f'{x:+5.2f}' for x in o - F1[lab][ds][0]))

fig, axes = plt.subplots(1, 2, figsize=(COLW, 2.2))
dodge = np.linspace(-0.11, 0.11, len(SERIES))  # log2 units, so the error bars of one K do not overlap
panels = [('genai', '(a) GenAI, 6 classes', (30, 62), [30, 40, 50, 60]),
          ('ccma', '(b) CCMA, 9 apps', (45, 90), [50, 60, 70, 80, 90])]
for ax, (ds, title, ylim, yt) in zip(axes, panels):
    for i, (lab, g, c, col, mk, ls, filled, lw, ms, z, _row) in enumerate(SERIES):
        m, sd = F1[lab][ds]
        ax.errorbar(KS * 2 ** dodge[i], m, yerr=sd, color=col, marker=mk, ms=ms, ls=ls, lw=lw,
                    mfc=col if filled else 'white', mec=col, mew=0.8, capsize=1.2, elinewidth=0.5, capthick=0.5,
                    zorder=z, label=lab)
    ax.set_xscale('log', base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
    ax.set_xlim(0.78, 10.2); ax.set_ylim(*ylim); ax.set_yticks(yt)
    ax.set_xlabel('labelled sessions per class, $K$'); ax.set_title(title, loc='left')
axes[0].set_ylabel('test macro-F1 (%)')
fig.tight_layout(pad=0.15, w_pad=0.9, rect=(0, 0, 1, 0.80))  # the top strip holds the shared legend
h = [Line2D([], [], color=col, marker=mk, ms=ms, ls=ls, lw=lw, mfc=col if filled else 'white', mec=col, mew=0.8, label=lab)
     for lab, g, c, col, mk, ls, filled, lw, ms, z, _row in SERIES]  # legend keys without the error bars
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.6, columnspacing=1.2,
           handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
save(fig, 'fig_fewshot_main')

# =====================================================================================================================
# F2: defence cost vs remaining leak
# =====================================================================================================================
RA, RK, DS = jload('robust_assistant.json'), jload('robust_assistant_knn.json'), jload('defence_stats.json')
SE, WC, AT = jload('session_extras.json'), jload('whole_connection_defences.json'), jload('adv_transfer.json')
AE = [json.loads(l) for l in open(os.path.join(RES, 'adv_eval.jsonl'), encoding='utf-8')]
BUDGETS = ['64_10', '256_50', '1460_200']

# ---- consistency checks ----
for ds in ('genai', 'ccma'):
    for cond in RA[ds]:
        for o in ('lgbm', 'ours'):
            assert abs(RK[ds][cond][o]['sess_asst_acc']['mean'] - RA[ds][cond][o]['sess_asst_acc'][0]) < 1e-6, (ds, cond, o)
    t = DS[ds]['tamaraw']['params']
    assert (t['packet_bytes'], t['rho_up_ms'], t['rho_down_ms'], t['pad_multiple'], t['causal']) == (1500.0, 40.0, 12.0, 100, True), t
    assert abs(AT['results'][ds]['none']['none']['encoder']['sess_vote']['mean'] - RA[ds]['none']['ours']['sess_asst_acc'][0]) < 0.01
    for b in BUDGETS:
        bb = float(b.split('_')[0])
        rec = [r for r in AE if r.get('tag') == 'adv_shaping' and r['dataset'] == ds and r['surrogate'] == 'whitebox'
               and 'advtrain' not in r['run'] and r['budget_bytes'] == bb]
        assert len(rec) == 5, (ds, b, len(rec))
        assert abs(100 * np.mean([r['test_f1_adv'] for r in rec]) - AT['results'][ds]['wb'][b]['encoder']['task_f1']['mean']) < 1e-3
assert AT['checks']['R1_pass'] and AT['checks']['R2_pass'] and AT['checks']['R3_pass']


def vote(ds, cond, obs):
    if obs == 'knn': return RK[ds][cond]['knn']['sess_asst_acc']['mean']
    return RA[ds][cond]['lgbm' if obs == 'lgbm' else 'ours']['sess_asst_acc'][0]


FAMC = {'none': '#555555', 'ech': LBLUE, 'pj': '#009E73', 'front': BLUE, 'tamaraw': '#E69F00', 'adv': '#CC79A7'}
OBSM = {'lgbm': 's', 'enc': 'o', 'knn': '^', 'flow': 'D'}
OBS_OFF = {'lgbm': -1, 'enc': 0, 'knn': 1, 'flow': 0}  # horizontal offsets within a defence, in steps of STEP decades
STEP = 0.028
X0 = 0.0025  # slot of 'no defence' on the log axis, left of the break


def points(ds):
    """One dict per plotted point: family, observer, trained on defended traffic?, x (cost), y (vote), group offset."""
    ext = SE['f_overheads'][ds]
    cost = {'ech': ext['ech']['window_clienthello_only']['aggregate'],
            'echfull': ext['ech']['window_clienthello_plus_server']['aggregate'],
            'pj': ext['pad256_bytes']['aggregate'],
            'front': DS[ds]['front']['dummy_bytes_over_real'],
            'tamaraw': DS[ds]['tamaraw']['extra_bytes_aggregate']}
    P = []
    # on CCMA, ECH on both flights (0.231) and padding + jitter (0.219) cost nearly the same: offset the two groups in
    # opposite directions, keeping their order on the axis, so that neither hides the other (still within the factor 1.23
    # stated in the docstring)
    g_echfull, g_pjad = (1, -1) if ds == 'ccma' else (0, 0)
    for fam, cond, x, adapt, g in [('none', 'none', X0, True, 0), ('ech', 'ech', cost['ech'], True, 0),
                                   ('ech', 'echfull', cost['echfull'], True, g_echfull),
                                   ('pj', 'pj_unaware', cost['pj'], False, 1), ('pj', 'pj_adaptive', cost['pj'], True, g_pjad),
                                   ('front', 'front_unaware', cost['front'], False, -1), ('front', 'front_adaptive', cost['front'], True, 1),
                                   ('tamaraw', 'tamaraw_adaptive', cost['tamaraw'], True, 0)]:
        for obs in ('lgbm', 'enc', 'knn'):
            P.append(dict(fam=fam, cond=cond, obs=obs, adapt=adapt, x=x, y=vote(ds, cond, obs), g=g))
    tp = WC['tamaraw_published'][ds]
    P.append(dict(fam='tamaraw', cond='tamaraw_whole', obs='flow', adapt=True, x=tp['costs']['L=100']['extra_bytes_aggregate'],
                  y=tp['observers']['published L=100: (i) W adaptive']['sess_vote']['mean'], g=0))
    P.append(dict(fam='front', cond='front_whole', obs='flow', adapt=True,
                  x=WC[ds]['overheads_whole']['front_whole']['dummy_bytes_aggregate']['mean'],
                  y=WC[ds]['observers']['front']['(i) W adaptive']['sess_vote']['mean'], g=0))
    for b in BUDGETS:
        for obs, key in (('enc', 'encoder'), ('knn', 'knn10')):
            P.append(dict(fam='adv', cond='wb_' + b, obs=obs, adapt=False, x=AT['overheads'][ds]['wb'][b]['bytes_ratio']['mean'],
                          y=AT['results'][ds]['wb'][b][key]['sess_vote']['mean'], g=0))
    return P


PTS = {ds: points(ds) for ds in ('genai', 'ccma')}
for ds in PTS:
    for p in PTS[ds]:
        print(f"F2 {ds:5s} {p['cond']:17s} {p['obs']:4s} x={p['x']:.4g} y={p['y']:.2f}")

# direct labels of the defences, panel (a) only (panel (b) uses the same colours and markers):
# (text, x, y, ha, va) in data coordinates; the x of a label is the cost of the defence it names
LABELS = {
    'genai': [('none', X0, 94, 'center', 'top'),
              ('ECH,\nClientHello', 0.008, 102.5, 'center', 'bottom'),
              ('pad.+\njitter', 0.13, 102.5, 'center', 'bottom'),
              ('ECH,\nboth', 0.34, 102.5, 'center', 'bottom'),
              ('FRONT', 1.1, 102.5, 'center', 'bottom'),
              ('FRONT, whole\nconnections', 0.55, 78.5, 'left', 'center'),
              ('Tamaraw,\n64 packets', 6.75, 37, 'center', 'bottom'),
              ('Tamaraw, whole\nconnections', 45.6, 59, 'center', 'bottom')],
    'ccma': [],
}
LEADERS = {'genai': [((0.53, 80.5), (0.15, 92.8))], 'ccma': []}  # (text end, point) for labels set apart from their point
CHANCE = {'genai': 100 / 3, 'ccma': 100 / 9}
YTOP = {'genai': 119, 'ccma': 104}


def draw(ax, ds):
    P = PTS[ds]
    ax.axhline(CHANCE[ds], color='#606060', ls='--', lw=0.7, zorder=1)
    ax.text(0.0017, CHANCE[ds] + 1.5, f'chance {CHANCE[ds]:.0f}%', fontsize=7, color='#404040', ha='left', va='bottom')
    # white-box shaping: connect the budgets of each observer, so the divergence of the two votes reads as two curves
    for obs in ('enc', 'knn'):
        q = sorted([p for p in P if p['fam'] == 'adv' and p['obs'] == obs], key=lambda p: p['x'])
        ax.plot([p['x'] * 10 ** (OBS_OFF[obs] * STEP) for p in q], [p['y'] for p in q], color=FAMC['adv'], lw=0.8,
                ls='-' if obs == 'enc' else (0, (1, 1)), zorder=2)
    for p in P:
        x = p['x'] * 10 ** ((OBS_OFF[p['obs']] + 2.2 * p['g']) * STEP) if p['fam'] != 'adv' else p['x'] * 10 ** (OBS_OFF[p['obs']] * STEP)
        col = FAMC[p['fam']]
        ax.plot(x, p['y'], marker=OBSM[p['obs']], ms=4.0 if p['obs'] != 'flow' else 3.6, ls='none', mec=col, mew=0.9,
                mfc=col if p['adapt'] else 'white', zorder=6 if p['obs'] == 'flow' else (4 if p['adapt'] else 5))
    for txt, x, y, ha, va in LABELS[ds]:
        ax.text(x, y, txt, fontsize=7, color=INK, ha=ha, va=va, linespacing=0.9, zorder=6)
    for (x0, y0), (x1, y1) in LEADERS[ds]:
        ax.plot([x0, x1], [y0, y1], color='#808080', lw=0.5, zorder=3)
    if ds == 'genai':  # key of the two white-box curves, in the empty lower-left corner
        kh = [Line2D([], [], color=FAMC['adv'], lw=0.8, ls='-', marker='o', ms=4, mfc='white', mec=FAMC['adv'], mew=0.9,
                     label="white-box shaping: encoder's vote"),
              Line2D([], [], color=FAMC['adv'], lw=0.8, ls=(0, (1, 1)), marker='^', ms=4, mfc='white', mec=FAMC['adv'], mew=0.9,
                     label="white-box shaping: 1-NN's vote")]
        ax.legend(handles=kh, loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=7, handlelength=2.4, handletextpad=0.4,
                  borderaxespad=0.2, labelspacing=0.2)
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xlim(0.0016, 130)
    ax.set_xticks([X0, 0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(['0', '0.01', '0.1', '1', '10', '100'])
    ax.set_ylim(0, YTOP[ds]); ax.set_yticks([0, 20, 40, 60, 80, 100])
    # axis break between the 'no defence' slot and the log scale
    xb = 0.0046
    ax.plot([xb / 1.12, xb * 1.12], [0, 0], color='white', lw=2.5, transform=ax.get_xaxis_transform(), clip_on=False, zorder=7)
    for dx in (1 / 1.12, 1.12):
        ax.plot([xb * dx / 1.06, xb * dx * 1.06], [-0.025, 0.025], color='#222222', lw=0.6, transform=ax.get_xaxis_transform(),
                clip_on=False, zorder=8)
    ax.axvline(xb, color='#e3e3e3', lw=0.5, zorder=0)


def defence_figure(dss, name, height, ratios):
    fig, axes = plt.subplots(len(dss), 1, figsize=(COLW, height), sharex=True, gridspec_kw=dict(height_ratios=ratios), squeeze=False)
    axes = axes[:, 0]
    titles = {'genai': 'GenAI: sessions whose assistant is named (%)', 'ccma': 'CCMA: sessions whose app is named (%)'}
    for i, (ax, ds) in enumerate(zip(axes, dss)):
        draw(ax, ds); ax.set_title((f'({"ab"[i]}) ' if len(dss) > 1 else '') + titles[ds], loc='left')
    axes[-1].set_xlabel('bytes added by the defence / bytes observed (log scale)')
    fig.tight_layout(pad=0.15, h_pad=0.5, rect=(0, 0, 1, 1 - 0.45 / height))  # the top strip holds the legend
    handles = [Line2D([], [], marker='s', ls='none', ms=4, mfc=INK, mec=INK, label='LightGBM, first 64 packets'),
               Line2D([], [], marker='o', ls='none', ms=4, mfc=INK, mec=INK, label='pre-trained encoder'),
               Line2D([], [], marker='^', ls='none', ms=4, mfc=INK, mec=INK, label='1-NN, first 10 packets'),
               Line2D([], [], marker='D', ls='none', ms=3.6, mfc=INK, mec=INK, label='LightGBM, flow records'),
               Line2D([], [], marker='o', ls='none', ms=4, mfc='white', mec=INK, mew=0.9, label='open: trained on undefended traffic'),
               Line2D([], [], color='#606060', ls='--', lw=0.7, label='uniform guess')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
    save(fig, name)


defence_figure(('genai', 'ccma'), 'fig_defence_tradeoff', 3.6, [1.3, 1.0])
defence_figure(('genai',), 'fig_defence_tradeoff_genai', 2.45, [1.0])  # one-panel variant
print('done')
