"""Fine-tune ET-BERT, a byte-level reference observer, on bigram tokens of the biflow payload
(scripts/build_etbert_tokens.py). Fine-tuning defaults of the authors: AdamW, linear warm-up over 10% of the steps
then linear decay, batch 32, 10 epochs, dropout 0.1. Checkpoint rule and record format as in src/train_yatc.py.
"""
import argparse, hashlib, json, math, os, random, sys, time, copy
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData
from train_yatc import metrics_from_preds, set_seed
from carriers.etbert_compat import ETBERTClassifier


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default=''); p.add_argument('--tag', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', default='configs/splits/session_split_s2026.json')
    p.add_argument('--epochs', type=int, default=10); p.add_argument('--bs', type=int, default=32)  # ET-BERT fine-tuning: 10 epochs, batch 32
    p.add_argument('--lr', type=float, default=6e-5); p.add_argument('--warmup', type=float, default=0.1)  # ET-BERT paper: 6e-5 for flow-level fine-tuning
    p.add_argument('--wd', type=float, default=0.01); p.add_argument('--smoothing', type=float, default=0.0)
    p.add_argument('--pretrained', default='weights/etbert_pretrained_model.bin')
    p.add_argument('--mask_sni', type=int, default=0); p.add_argument('--subset', type=int, default=0)
    p.add_argument('--amp', default='bf16', choices=['bf16', 'fp32'])
    p.add_argument('--out_dir', default='results/runs')
    return p.parse_args()


def build_model(args, log):
    model = ETBERTClassifier(6)
    load_msg = None
    if args.pretrained:
        load_msg = model.load_pretrained(os.path.join(ROOT, args.pretrained)); log(f'loaded {args.pretrained}: {load_msg}')
    return model, load_msg


@torch.no_grad()
def predict(model, src, seg, y, device, amp_dtype, bs=256):
    model.eval(); preds = []
    for b in range(0, len(src), bs):
        s, g = src[b:b + bs].to(device), seg[b:b + bs].to(device)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            preds.append(model(s, g).argmax(1).cpu().numpy())
    return y, np.concatenate(preds)


def main():
    args = get_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ('run_name', 'tag', 'out_dir')}
    cfg['carrier'] = 'etbert'
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    run_id = args.run_name or f'etbert_{cfg_hash}_s{args.seed}'
    run_dir = os.path.join(ROOT, args.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== run {run_id} cfg_hash={cfg_hash} {json.dumps(cfg)}')
    set_seed(args.seed); torch.backends.cudnn.benchmark = True
    device = torch.device('cuda'); amp_dtype = torch.bfloat16 if args.amp == 'bf16' else None
    data = GenAIData(ROOT); idx, split_meta = data.split_indices(os.path.join(ROOT, args.split))
    tok = np.load(os.path.join(ROOT, 'data', 'derived', 'etbert_tokens.npz'))
    src_all = torch.from_numpy(tok['src_masksni' if args.mask_sni else 'src'].astype(np.int64))
    seg_all = torch.from_numpy(tok['seg_masksni' if args.mask_sni else 'seg'].astype(np.int64))
    y_all = data.y_joint
    tr_idx = idx['train']
    if args.subset:
        rng = np.random.RandomState(args.seed); keep = []
        for c in range(6):
            ci = tr_idx[y_all[tr_idx] == c]; keep.append(rng.choice(ci, size=min(len(ci), max(1, args.subset // 6)), replace=False))
        tr_idx = np.concatenate(keep)
    log(f'split hash={split_meta["hash"]} n_train={len(tr_idx)} n_val={len(idx["val"])} n_test={len(idx["test"])}')

    model, load_msg = build_model(args, log); model.to(device)
    n_params = sum(p.numel() for p in model.parameters()); n_tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'params total={n_params} trainable={n_tr_params}')
    no_decay = ['bias', 'gamma', 'beta']
    groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
              {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = int(math.ceil(len(tr_idx) / args.bs)); total = steps_per_epoch * args.epochs; warm = int(total * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    crit = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    yv = y_all[idx['val']]; best = dict(macro_f1=-1, epoch=-1); best_state = None; hist = []; nonfinite = 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); rng = np.random.RandomState(args.seed)
    for ep in range(args.epochs):
        model.train(); perm = rng.permutation(tr_idx); tl = 0.0; tn = 0
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            s, g, yb = src_all[bi].to(device), seg_all[bi].to(device), torch.from_numpy(y_all[bi]).to(device)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = crit(model(s, g), yb)
            if not torch.isfinite(loss):
                nonfinite += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(bi); tn += len(bi)
        _, pv = predict(model, src_all[idx['val']], seg_all[idx['val']], yv, device, amp_dtype)
        f1 = float(f1_score(yv, pv, average='macro')); acc = float(accuracy_score(yv, pv))
        hist.append(dict(epoch=ep, train_loss=tl / max(tn, 1), val_macro_f1=f1, val_acc=acc))
        log(f'ep {ep:3d} loss {tl / max(tn, 1):.4f} val_f1 {f1:.4f} val_acc {acc:.4f}' + (f' nonfinite={nonfinite}' if nonfinite else ''))
        if (ep == args.epochs - 1) if os.environ.get('CKPT_SELECT', 'val') == 'last' else (f1 > best['macro_f1']):  # CKPT_SELECT=last: final epoch of the fixed schedule
            best = dict(macro_f1=f1, epoch=ep); best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t0; peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20
    model.load_state_dict(best_state)
    torch.save({'model': best_state, 'args': vars(args), 'best': best}, os.path.join(run_dir, 'best.pt'))
    m2, _ = build_model(args, lambda s: None); m2.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), weights_only=False)['model']); m2.to(device)
    _, pv = predict(model, src_all[idx['val']], seg_all[idx['val']], yv, device, amp_dtype)
    _, pv2 = predict(m2, src_all[idx['val']], seg_all[idx['val']], yv, device, amp_dtype)
    assert (pv == pv2).mean() > 0.99, 'checkpoint reload mismatch'
    val_m = metrics_from_preds(yv, pv, sessions=data.session_id[idx['val']], seed=args.seed)
    np.savez(os.path.join(run_dir, 'val_preds.npz'), y=yv, pred=pv, idx=idx['val'])
    # latency
    model.eval(); xs, xg = src_all[:64].to(device), seg_all[:64].to(device)
    with torch.no_grad(), torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
        for _ in range(5): model(xs, xg)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): model(xs, xg)
        torch.cuda.synchronize(); lat64 = (time.perf_counter() - t) / 20 * 1000 / 64
    rec = dict(run_id=run_id, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=args.tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=split_meta['hash'], seed=args.seed,
               best_epoch=best['epoch'], epochs=args.epochs, n_train=int(len(tr_idx)), n_val=int(len(idx['val'])), pretrained_load=load_msg,
               params=n_params, trainable_params=n_tr_params, train_time_s=round(train_time, 1), peak_mem_mb=round(peak_mem, 1),
               latency_ms_per_sample_b64=round(lat64, 4), nonfinite_steps=nonfinite, val=val_m, hist=hist)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
    log(f'BEST epoch {best["epoch"]} val macro_f1 {val_m["macro_f1"]:.4f} ci95 {val_m["macro_f1_ci95_sessions"]} app_f1 {val_m["app_macro_f1"]:.4f} act_f1 {val_m["act_macro_f1"]:.4f} | params {n_params} time {train_time:.0f}s mem {peak_mem:.0f}MB lat {lat64:.3f}ms')
    yt = y_all[idx['test']]; _, pt = predict(model, src_all[idx['test']], seg_all[idx['test']], yt, device, amp_dtype)
    test_m = metrics_from_preds(yt, pt, sessions=data.session_id[idx['test']], seed=args.seed)
    np.savez(os.path.join(run_dir, 'test_preds.npz'), y=yt, pred=pt, idx=idx['test'])
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=run_id, cfg_hash=cfg_hash, seed=args.seed, tag=args.tag, split_hash=split_meta['hash'], best_epoch=best['epoch'], test=test_m)) + '\n')
    json.dump(rec, open(os.path.join(run_dir, 'record.json'), 'w'), indent=1); logf.close()


if __name__ == '__main__':
    main()
