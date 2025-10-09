# ncars_linear_baseline.py
import os
import sys
import json
import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from tqdm import tqdm
from time import perf_counter

# helpers from your repo
from utils import (
    device,
    infer_sensor_size_from_cache,
    window_to_pol_hist,      # (H,W,2) histogram for a single window
    decode_addr_to_xy_p,     # decodes addr -> (x, y, p in {-1,+1})
)

# ------------------------
# Tiny stdout tee → file + console
# ------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def _start_logging(out_dir: str, run_name: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(out_dir, f"{ts}_{run_name}.log")
    f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    return log_path

# ------------------------
# Reproducibility helpers
# ------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # lite determinism (no CuBLAS strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------
# Cache utilities
# ------------------------
def _load_cache(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache: {path}")
    return torch.load(path, weights_only=False)

def _sample_indices(raw, n: int):
    n = min(n, len(raw["index"]))
    idx = np.random.choice(len(raw["index"]), size=n, replace=False)
    return idx.tolist()

# ------------------------
# Feature builders
# ------------------------
# ...existing code...

# --- Replace this function ---
def window_to_pol_voxel(raw, idx: int, W: int, H: int, time_bins: int) -> np.ndarray:
    # Vectorized version from snn.py
    file_idx, t_start, t_end, _ = raw["index"][idx]
    window_us = max(1, int(t_end - t_start))
    ts = raw["data"][file_idx]["ts"]
    addr = raw["data"][file_idx]["addr"]

    m = (ts >= t_start) & (ts < t_end)
    if not np.any(m):
        return np.zeros((H, W, 2 * time_bins), dtype=np.float32)

    ts_w = ts[m]
    addr_w = addr[m]

    x, y, p = decode_addr_to_xy_p(addr_w)
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    pol = (p > 0).astype(np.int64)

    rel = ts_w - t_start
    bin_idx = np.floor(rel * time_bins / window_us).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, time_bins - 1)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not np.any(valid):
        return np.zeros((H, W, 2 * time_bins), dtype=np.float32)
    x = x[valid]
    y = y[valid]
    pol = pol[valid]
    bin_idx = bin_idx[valid]

    ch = pol + 2 * bin_idx
    flat_idx = ((y * W + x) * (2 * time_bins) + ch).astype(np.int64)
    vox_flat = np.zeros(H * W * 2 * time_bins, dtype=np.uint32)
    np.add.at(vox_flat, flat_idx, 1)
    return vox_flat.reshape(H, W, 2 * time_bins).astype(np.float32)
# --- End replacement ---

# ...rest of your code...

def _build_split(
    raw_pos,
    raw_neg,
    n_pos: int,
    n_neg: int,
    W: int,
    H: int,
    time_bins: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_list, y_list = [], []

    pos_idx = _sample_indices(raw_pos, n_pos)
    neg_idx = _sample_indices(raw_neg, n_neg)

    for i in tqdm(pos_idx, desc=f"Building + class (vox T={time_bins})", leave=False):
        vox = window_to_pol_voxel(raw_pos, i, W, H, time_bins)  # always use vectorized
        X_list.append(vox)
        y_list.append(1)
    for i in tqdm(neg_idx, desc=f"Building - class (vox T={time_bins})", leave=False):
        vox = window_to_pol_voxel(raw_neg, i, W, H, time_bins)
        X_list.append(vox)
        y_list.append(0)
    C_out = 2 * time_bins

    X = np.stack(X_list, axis=0)  # (N,H,W,C)
    sums = X.sum(axis=(1, 2, 3), keepdims=True)
    sums[sums == 0] = 1.0
    X = X / sums

    X = torch.from_numpy(X).float().permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
    y = torch.tensor(y_list, dtype=torch.long)
    assert X.shape[1] == C_out
    return X, y

# ------------------------
# Norm helper
# ------------------------
def _make_norm(kind: str, num_channels: int):
    kind = (kind or "bn").lower()
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    if kind == "gn":
        groups = 8 if num_channels % 8 == 0 else 4
        return nn.GroupNorm(groups, num_channels)
    if kind == "ln":
        # LayerNorm over channels: GroupNorm with 1 group is equivalent & fast
        return nn.GroupNorm(1, num_channels)
    return nn.Identity()

# ------------------------
# Models
# ------------------------
class LinearHead(nn.Sequential):
    def __init__(self, in_ch, H, W):
        super().__init__(nn.Flatten(), nn.Linear(in_ch * H * W, 2))

class TinyCNN(nn.Module):
    """
    Very small CNN with selectable normalization:
      Conv → Norm → ReLU → Conv → Norm → ReLU → GAP → Linear
    """
    def __init__(self, in_ch: int, norm: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_norm(norm, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm(norm, 32)
        self.relu  = nn.ReLU(inplace=True)
        self.head  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return self.head(x)

def build_model(model_name: str, C: int, H: int, W: int, norm: str = "bn") -> nn.Module:
    if model_name == "linear":
        return LinearHead(C, H, W)
    elif model_name == "cnn":
        return TinyCNN(C, norm=norm)
    else:
        raise ValueError(f"Unknown --model {model_name!r} (use 'linear' or 'cnn')")

# ------------------------
# Train / eval
# ------------------------
def train_and_eval(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    model_name: str = "linear",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    num_workers: int = 8,
    norm: str = "bn",
):
    Ntr, C, H, W = Xtr.shape
    model = build_model(model_name, C, H, W, norm=norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    print(f"[BASELINE-{model_name}] training: Ntr={Ntr}, Nte={Xte.size(0)}, Din={C*H*W}, epochs={epochs}")
    model.train()
    for ep in range(1, epochs + 1):
        total, correct, total_loss = 0, 0, 0.0
        for xb, yb in tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total += yb.numel()
            total_loss += loss.item() * yb.numel()
            correct += (logits.argmax(dim=1) == yb).sum().item()
        print(f"[epoch {ep:2d}] train acc={correct/total:.3f}  loss={total_loss/total:.4f}")

    # -------- Batched EVAL to avoid OOM --------
    model.eval()
    te_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    all_preds = []
    all_t = []
    with torch.no_grad():
        for xb, yb in tqdm(te_loader, desc="eval", leave=False):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_t.append(yb.cpu())
    preds = torch.cat(all_preds, dim=0)
    y_cpu = torch.cat(all_t, dim=0)
    acc = (preds == y_cpu).float().mean().item()
    print(f"[BASELINE-{model_name}] test accuracy: {acc:.3f}")

    # Confusion matrix
    cm = torch.zeros(2, 2, dtype=torch.int64)
    for t, p in zip(y_cpu, preds):
        cm[t, p] += 1
    tn, fp, fn, tp = cm[0,0].item(), cm[0,1].item(), cm[1,0].item(), cm[1,1].item()
    print(f"[BASELINE-{model_name}] confusion matrix:\n"
          f"            pred 0   pred 1\n"
          f" true 0     {tn:6d}   {fp:6d}\n"
          f" true 1     {fn:6d}   {tp:6d}")

    return {"test_acc": acc, "cm": cm.tolist()}

# ------------------------
# Main
# ------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="NCARS baseline (linear or tiny CNN; hist or voxel).")
    parser.add_argument("--train_cars", default="caches/ncars_train_cars_win50ms.pt")
    parser.add_argument("--train_bg",   default="caches/ncars_train_background_win50ms.pt")
    parser.add_argument("--test_cars",  default="caches/ncars_test_cars_win50ms.pt")
    parser.add_argument("--test_bg",    default="caches/ncars_test_background_win50ms.pt")
    parser.add_argument("--per_class_train", type=int, default=1000)
    parser.add_argument("--per_class_test",  type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--time_bins", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model", choices=["linear", "cnn"], default="linear")
    parser.add_argument("--norm", choices=["bn", "gn", "ln", "none"], default="bn",
                        help="Normalization for CNN: batch/group/layer/none")
    parser.add_argument("--profile_build_only", action="store_true",
                    help="Build X/y for train+test, report timing, then exit (no training).")

    # logging
    parser.add_argument("--out_dir", default="/workspace/results")
    parser.add_argument("--run_name", default=None, help="optional tag for filenames")
    args = parser.parse_args()

    # compose run name
    tb_tag = 'hist' if args.time_bins <= 1 else f'voxT{args.time_bins}'
    default_name = (f"{args.model}_{tb_tag}_norm{args.norm}_W{args.W or 'auto'}H{args.H or 'auto'}_"
                    f"train{args.per_class_train}_test{args.per_class_test}_seed{args.seed}")
    run_name = args.run_name or default_name

    # start tee logging
    log_path = _start_logging(args.out_dir, run_name)
    print(f"[LOG] writing to: {log_path}")

    set_global_seed(args.seed)

    print("[BASELINE] loading caches…")
    train_cars = _load_cache(args.train_cars)
    train_bg   = _load_cache(args.train_bg)
    test_cars  = _load_cache(args.test_cars)
    test_bg    = _load_cache(args.test_bg)

    if args.W is not None and args.H is not None:
        W, H = args.W, args.H
        print(f"[BASELINE] using manual sensor: W={W}, H={H}")
    else:
        W, H = infer_sensor_size_from_cache(train_cars)
        print(f"[BASELINE] inferred sensor: W={W}, H={H}")

    mode_str = "hist (2ch)" if args.time_bins <= 1 else f"voxel (2*{args.time_bins} ch)"
    
    # --- Time train split creation ---
    print(f"[BASELINE] building train split…  [{mode_str}]")
    t0 = time.perf_counter()
    Xtr, ytr = _build_split(train_cars, train_bg, args.per_class_train, args.per_class_train, W, H, args.time_bins)
    t1 = perf_counter()
    print(f"[TIME] build_train: {(t1 - t0):.2f}s  (N={Xtr.size(0)}, C={Xtr.size(1)}, H={H}, W={W})")
    
    # --- Time test split creation ---
    t2 = perf_counter()
    print(f"[BASELINE] building test split…   [{mode_str}]")
    Xte, yte = _build_split(test_cars, test_bg, args.per_class_test, args.per_class_test, W, H, args.time_bins)
    t3 = perf_counter()
    print(f"[TIME] build_test : {(t3 - t2):.2f}s  (N={Xte.size(0)}, C={Xte.size(1)}, H={H}, W={W})")
    print(f"[TIME] build_total: {(t3 - t0):.2f}s")

    if args.profile_build_only:
        print("[MODE] profile_build_only → skipping training/eval.")
        return


    # quick sanity log
    print(f"[DEBUG] Xtr shape = {tuple(Xtr.shape)}  (N,C,H,W) | Xte shape = {tuple(Xte.shape)}")
    if args.time_bins > 1:
        with torch.no_grad():
            ch_means = Xtr.mean(dim=(0, 2, 3))
            show = min(12, ch_means.numel())
            print("[DEBUG] per-channel mean (first {}): {}".format(
                show, ", ".join(f"{v:.6f}" for v in ch_means[:show])))

    metrics = train_and_eval(
        Xtr, ytr, Xte, yte,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        norm=args.norm,
    )

    # save a compact metrics json + append to summary tsv
    meta = {
        "run_name": run_name,
        "log_path": log_path,
        "model": args.model,
        "norm": args.norm,
        "W": W, "H": H, "time_bins": args.time_bins,
        "per_class_train": args.per_class_train,
        "per_class_test": args.per_class_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device.type,
    }
    out_json = os.path.join(args.out_dir, os.path.basename(log_path).replace(".log", "_metrics.json"))
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "metrics": metrics}, f, indent=2)
    print(f"[SAVE] metrics → {out_json}")

    summary_path = os.path.join(args.out_dir, "summary.tsv")
    header = ("timestamp\trun_name\tmodel\tW\tH\tnorm\ttime_bins\tper_class_train\tper_class_test\t"
              "epochs\tbatch\tlr\twd\tseed\tdevice\ttest_acc\tcm")
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{run_name}\t{args.model}\t{W}\t{H}\t{args.norm}\t{args.time_bins}\t"
        f"{args.per_class_train}\t{args.per_class_test}\t{args.epochs}\t{args.batch_size}\t"
        f"{args.lr}\t{args.weight_decay}\t{args.seed}\t{device.type}\t{metrics['test_acc']:.6f}\t{metrics['cm']}"
    )
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header + "\n")
    with open(summary_path, "a") as f:
        f.write(line + "\n")
    print(f"[APPEND] {summary_path}")

if __name__ == "__main__":
    main()