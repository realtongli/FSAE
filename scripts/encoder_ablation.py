"""Encoder size and use of pre-training on CCMA with K = 4 (appendix, ablation): test macro-F1 (%) over the five K-session
draws of the pre-trained encoders of 0.14M, 0.8M and 2.4M parameters (pre-trained for 200, 300 and 400 epochs, fine-tuned
with the same recipe) and of a linear probe on the frozen 2.4M encoder. Runs: block 6 of scripts/run_paper.sh.
Reads results/locked_test_metrics.jsonl (the last record of a run id counts); writes results/encoder_ablation.json."""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
T = {}
for l in open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), encoding='utf-8'):
    r = json.loads(l)
    if r['tag'].startswith('smoke'): continue
    T[r['run_id']] = r['test']['macro_f1']
ROWS = [('0.14M encoder, pre-trained for 200 epochs', 'ccma_fs4_ssl_ft_lr3e4'), ('0.8M encoder, pre-trained for 300 epochs', 'ccma_fs4_sslL_ft_lr3e4'),
        ('2.4M encoder, pre-trained for 400 epochs (ours)', 'ccma_fs4_sslXL'), ('linear probe on the frozen 2.4M encoder', 'ccma_fs4_sslXL_probe')]
OUT = {}
for lab, p in ROWS:
    v = 100 * np.array([T[f'{p}_s{s}'] for s in range(5)])
    OUT[p] = dict(label=lab, mean=float(v.mean()), sd=float(v.std()), per_draw=[float(x) for x in v])
    print(f'{lab:48s} {v.mean():5.1f} +- {v.std():.1f}')
json.dump(OUT, open(os.path.join(ROOT, 'results', 'encoder_ablation.json'), 'w'), indent=1)
print('written results/encoder_ablation.json')
