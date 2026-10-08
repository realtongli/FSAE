"""Fine-tune and evaluate YaTC, the byte-level reference observer, on the derived biflow arrays.

With CKPT_SELECT=last the final epoch of the fixed schedule is kept, so no labelled session beyond the K training
sessions is used (the protocol of the paper); otherwise the epoch with the best validation macro-F1 is kept. Validation
metrics are appended to results/metrics.jsonl and test metrics to results/locked_test_metrics.jsonl. The module also
holds the metric and seeding helpers shared by every trainer (metrics_from_preds, set_seed).
"""
import argparse, hashlib, json, math, os, random, sys, time, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
from data_mfr import GenAIData, MFRDataset, CLASS_NAMES
from carriers import yatc_compat


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default='')
    p.add_argument('--tag', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', default='configs/splits/session_split_s2026.json')
    p.add_argument('--dataset', default='genai', choices=['genai', 'ccma'])
    p.add_argument('--target', default='joint', choices=['joint', 'app'])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--blr', type=float, default=2e-3)  # YaTC default; lr = blr * bs / 256
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--drop_path', type=float, default=0.1)
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--clip_grad', type=float, default=0.0)
    p.add_argument('--pretrained', default='')  # weights/yatc_pretrained-model.pth
    p.add_argument('--layer_decay', type=float, default=0.0, help='official YaTC fine-tuning uses 0.75; 0 = off')
    p.add_argument('--mfr', default='payload5', choices=['payload5', 'official'],
                   help='official: first 5 packets incl. payload-free ones, each 80 B header (zeros: MIRAGE has no header bytes) + 240 B payload')
    p.add_argument('--mask_sni', type=int, default=0)
    p.add_argument('--mask_mode', default='', choices=['', 'none', 'sni', 'clienthello'])
    p.add_argument('--subset', type=int, default=0, help='use only N training samples (smoke test)')
    p.add_argument('--amp', default='bf16', choices=['bf16', 'fp32'])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--out_dir', default='results/runs')
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def adjust_lr(opt, ep_frac, args, base_lr):
    if ep_frac < args.warmup:
        lr = base_lr * ep_frac / args.warmup
    else:
        lr = args.min_lr + (base_lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * (ep_frac - args.warmup) / (args.epochs - args.warmup)))
    for g in opt.param_groups:
        g['lr'] = lr * g.get('lr_scale', 1.0)
    return lr


def metrics_from_preds(y, pred, sessions=None, n_boot=500, seed=0, n_classes=6, n_act=2):
    L = list(range(n_classes))
    m = dict(acc=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average='macro')),
             weighted_f1=float(f1_score(y, pred, average='weighted')),
             per_class_recall=[float(v) for v in recall_score(y, pred, average=None, labels=L)],
             per_class_f1=[float(v) for v in f1_score(y, pred, average=None, labels=L)],
             app_macro_f1=float(f1_score(y // n_act, pred // n_act, average='macro')) if n_act > 1 else float('nan'),
             act_macro_f1=float(f1_score(y % n_act, pred % n_act, average='macro')) if n_act > 1 else float('nan'),
             cm=confusion_matrix(y, pred, labels=L).tolist(), n=int(len(y)))
    if sessions is not None:  # session-level bootstrap CI for macro-F1 (sessions are the independent units)
        rng = np.random.RandomState(seed); us = np.unique(sessions); vals = []
        groups = {s: np.where(sessions == s)[0] for s in us}
        for _ in range(n_boot):
            pick = rng.choice(us, size=len(us), replace=True)
            idx = np.concatenate([groups[s] for s in pick])
            vals.append(f1_score(y[idx], pred[idx], average='macro'))
        m['macro_f1_ci95_sessions'] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        m['n_sessions'] = int(len(us))
    return m


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval(); preds, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
        preds.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


@torch.no_grad()
def latency(model, device, amp_dtype, bs, n_iter=30):
    model.eval(); x = torch.randn(bs, 1, 40, 40, device=device)
    for _ in range(5):
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            model(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n_iter):
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n_iter * 1000.0 / bs  # ms per sample


def build_model(args, log):
    model = yatc_compat.TraFormer_YaTC(num_classes=args.n_classes, drop_path_rate=args.drop_path)
    load_msg = None
    if args.pretrained:
        ck = torch.load(os.path.join(ROOT, args.pretrained), map_location='cpu', weights_only=False)
        sd = ck['model'] if 'model' in ck else ck
        for k in ['head.weight', 'head.bias']:
            if k in sd and sd[k].shape != model.state_dict()[k].shape:
                del sd[k]
        msg = model.load_state_dict(sd, strict=False)
        load_msg = dict(missing=list(msg.missing_keys), unexpected=list(msg.unexpected_keys))
        from timm.layers import trunc_normal_
        trunc_normal_(model.head.weight, std=2e-5)
        log(f'loaded pretrained {args.pretrained}: missing={msg.missing_keys} unexpected={msg.unexpected_keys}')
    return model, load_msg


def main():
    args = get_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ('run_name', 'tag', 'out_dir', 'num_workers', 'n_classes')}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    run_id = args.run_name or f'{cfg_hash}_s{args.seed}'
    run_dir = os.path.join(ROOT, args.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== run {run_id} cfg_hash={cfg_hash} {json.dumps(cfg)}')
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' else None

    data = GenAIData(ROOT, prefix=args.dataset)
    n_act = data.n_act if args.target == 'joint' else 1
    args.n_classes = data.n_joint if args.target == 'joint' else data.n_app
    NC = args.n_classes
    idx, split_meta = data.split_indices(os.path.join(ROOT, args.split))
    tr_idx = idx['train']
    if args.subset:
        rng = np.random.RandomState(args.seed)
        # stratified subset covering all classes
        keep = []
        for c in range(NC):
            ci = tr_idx[(data.y_joint if args.target == 'joint' else data.y_app)[tr_idx] == c]; keep.append(rng.choice(ci, size=min(len(ci), max(1, args.subset // 6)), replace=False))
        tr_idx = np.concatenate(keep)
    log(f'split={args.split} hash={split_meta["hash"]} n_train={len(tr_idx)} n_val={len(idx["val"])} n_test={len(idx["test"])}')
    log('train class counts: ' + str(np.bincount((data.y_joint if args.target == 'joint' else data.y_app)[tr_idx], minlength=NC).tolist()))
    if args.mfr == 'official':
        data.use_official_mfr()
    ds_tr = MFRDataset(data, tr_idx, mask_sni=bool(args.mask_sni), target=args.target, mask_mode=(args.mask_mode or None))
    ds_va = MFRDataset(data, idx['val'], mask_sni=bool(args.mask_sni), target=args.target, mask_mode=(args.mask_mode or None))
    ds_te = MFRDataset(data, idx['test'], mask_sni=bool(args.mask_sni), target=args.target, mask_mode=(args.mask_mode or None))
    g = torch.Generator(); g.manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.num_workers, generator=g)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False, num_workers=args.num_workers)

    model, load_msg = build_model(args, log)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()); n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'params total={n_params} trainable={n_train_params}')

    base_lr = args.blr * args.bs / 256
    if args.layer_decay > 0:  # official fine-tune.py: BEiT-style layer-wise lr decay (util/lr_decay.py)
        import importlib.util
        spec = importlib.util.spec_from_file_location('yatc_lrd', os.path.join(ROOT, 'third_party', 'YaTC', 'util', 'lr_decay.py'))
        lrd = importlib.util.module_from_spec(spec); spec.loader.exec_module(lrd); param_groups_lrd = lrd.param_groups_lrd
        groups = param_groups_lrd(model, args.wd, no_weight_decay_list=model.no_weight_decay(), layer_decay=args.layer_decay)
        opt = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999))
    else:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim < 2 or 'pos_embed' in n or 'cls_token' in n or 'positional_encoding' in n) else decay).append(p)
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd}, {'params': no_decay, 'weight_decay': 0.0}], lr=base_lr, betas=(0.9, 0.999))
    crit = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    best = dict(macro_f1=-1, epoch=-1); best_state = None; hist = []
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); nonfinite = 0
    steps_per_epoch = len(dl_tr)
    for ep in range(args.epochs):
        model.train(); tl, tn = 0.0, 0
        for it, (x, y) in enumerate(dl_tr):
            lr = adjust_lr(opt, ep + it / steps_per_epoch, args, base_lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x); loss = crit(out, y)
            if not torch.isfinite(loss):
                nonfinite += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step(); tl += loss.item() * len(y); tn += len(y)
        yv, pv = predict(model, dl_va, device, amp_dtype)
        mv = dict(macro_f1=float(f1_score(yv, pv, average='macro')), acc=float(accuracy_score(yv, pv)))
        hist.append(dict(epoch=ep, train_loss=tl / max(tn, 1), lr=lr, val_macro_f1=mv['macro_f1'], val_acc=mv['acc']))
        log(f'ep {ep:3d} loss {tl / max(tn, 1):.4f} lr {lr:.2e} val_f1 {mv["macro_f1"]:.4f} val_acc {mv["acc"]:.4f}' + (f' nonfinite={nonfinite}' if nonfinite else ''))
        if (ep == args.epochs - 1) if os.environ.get('CKPT_SELECT', 'val') == 'last' else (mv['macro_f1'] > best['macro_f1']):
            best = dict(macro_f1=mv['macro_f1'], epoch=ep); best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20

    model.load_state_dict(best_state)
    torch.save({'model': best_state, 'args': vars(args), 'best': best}, os.path.join(run_dir, 'best.pt'))
    # reload check
    m2, _ = build_model(args, lambda s: None); m2.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), weights_only=False)['model']); m2.to(device)
    yv, pv = predict(model, dl_va, device, amp_dtype); yv2, pv2 = predict(m2, dl_va, device, amp_dtype)
    assert (pv == pv2).mean() > 0.99, 'checkpoint reload mismatch'
    val_m = metrics_from_preds(yv, pv, sessions=data.session_id[idx['val']], seed=args.seed, n_classes=NC, n_act=n_act)
    np.savez(os.path.join(run_dir, 'val_preds.npz'), y=yv, pred=pv, idx=idx['val'])
    lat64 = latency(model, device, amp_dtype, 64); lat1 = latency(model, device, amp_dtype, 1)
    rec = dict(run_id=run_id, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), tag=args.tag, cfg_hash=cfg_hash, cfg=cfg, split_hash=split_meta['hash'],
               seed=args.seed, best_epoch=best['epoch'], epochs=args.epochs, n_train=int(len(tr_idx)), n_val=int(len(idx['val'])),
               pretrained_load=load_msg, params=n_params, trainable_params=n_train_params,
               train_time_s=round(train_time, 1), peak_mem_mb=round(peak_mem, 1), latency_ms_per_sample_b64=round(lat64, 4),
               latency_ms_per_sample_b1=round(lat1, 3), nonfinite_steps=nonfinite, val=val_m, hist=hist)
    with open(os.path.join(ROOT, 'results', 'metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
    log(f'BEST epoch {best["epoch"]} val macro_f1 {val_m["macro_f1"]:.4f} ci95 {val_m["macro_f1_ci95_sessions"]} app_f1 {val_m["app_macro_f1"]:.4f} act_f1 {val_m["act_macro_f1"]:.4f} | params {n_params} time {train_time:.0f}s mem {peak_mem:.0f}MB lat {lat64:.3f}ms')
    # locked test metrics (never used for selection)
    yt, pt = predict(model, dl_te, device, amp_dtype)
    test_m = metrics_from_preds(yt, pt, sessions=data.session_id[idx['test']], seed=args.seed, n_classes=NC, n_act=n_act)
    np.savez(os.path.join(run_dir, 'test_preds.npz'), y=yt, pred=pt, idx=idx['test'])
    with open(os.path.join(ROOT, 'results', 'locked_test_metrics.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(run_id=run_id, cfg_hash=cfg_hash, seed=args.seed, tag=args.tag, split_hash=split_meta['hash'], best_epoch=best['epoch'], test=test_m)) + '\n')
    json.dump(rec, open(os.path.join(run_dir, 'record.json'), 'w'), indent=1)
    logf.close()


if __name__ == '__main__':
    main()
