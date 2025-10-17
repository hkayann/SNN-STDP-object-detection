# snn_linear.py
import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Tuple, List

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from spikingjelly.activation_based import neuron

from utils import (
    device,
    infer_sensor_size_from_cache,
    decode_addr_to_xy_p,
)


# ----------------------------------------------------------------------
# Tiny stdout tee → file + console (mirrors ncars_linear_baseline.py)
# ----------------------------------------------------------------------
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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(out_dir, f"{ts}_{run_name}.log")
    f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    return log_path


# ----------------------------------------------------------------------
# Reproducibility helpers (copied from baseline for parity)
# ----------------------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------
# Cache utilities
# ----------------------------------------------------------------------
def _load_cache(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache: {path}")
    return torch.load(path, weights_only=False)


def _sample_indices(raw, n: int | None) -> List[int]:
    total = len(raw["index"])
    if n is None or n <= 0:
        n = total
    else:
        n = min(n, total)
    idx = np.random.choice(total, size=n, replace=False)
    return idx.tolist()


# ----------------------------------------------------------------------
# Event window → voxel sequence (T, 2, H, W)
# ----------------------------------------------------------------------
def window_to_voxel_sequence(raw, idx: int, W: int, H: int, time_bins: int) -> torch.Tensor:
    file_idx, t_start, t_end, _ = raw["index"][idx]
    ts = raw["data"][file_idx]["ts"]
    addr = raw["data"][file_idx]["addr"]
    mask = (ts >= t_start) & (ts < t_end)
    if not np.any(mask):
        return torch.zeros(time_bins, 2, H, W, dtype=torch.float32)

    ts_win = ts[mask]
    addr_win = addr[mask]
    x, y, p = decode_addr_to_xy_p(addr_win)
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    pol = (p > 0).astype(np.int64)

    rel = ts_win - t_start
    win_us = max(1, int(t_end - t_start))
    bin_idx = np.floor(rel * time_bins / win_us).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, time_bins - 1)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not np.any(valid):
        return torch.zeros(time_bins, 2, H, W, dtype=torch.float32)

    x = x[valid]
    y = y[valid]
    pol = pol[valid]
    bin_idx = bin_idx[valid]

    ch = pol + 2 * bin_idx
    flat_idx = ((ch * H + y) * W + x).astype(np.int64)
    vox_flat = np.zeros(time_bins * 2 * H * W, dtype=np.float32)
    np.add.at(vox_flat, flat_idx, 1.0)
    return torch.from_numpy(vox_flat.reshape(time_bins, 2, H, W))


# ----------------------------------------------------------------------
# SNN feature extractor (conv + LIF with frozen weights)
# ----------------------------------------------------------------------
class SNNFeatureExtractor(nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        weights = torch.load(weight_path, map_location="cpu")
        if not isinstance(weights, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor weights, got {type(weights).__name__}")
        if weights.dim() != 4:
            raise ValueError(f"Expected 4D conv weights, got shape {tuple(weights.shape)}")

        out_ch, in_ch, kH, kW = weights.shape
        padding = (kH // 2, kW // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kH, kW), padding=padding, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weights.to(torch.float32))
        self.conv.requires_grad_(False)
        self.conv.eval()

        self.lif = neuron.LIFNode(
            tau=2.0,
            v_threshold=0.01,
            v_reset=0.0,
            detach_reset=True,
        )

    @torch.no_grad()
    def forward(self, seq_batch: torch.Tensor) -> torch.Tensor:
        if seq_batch.dim() != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(seq_batch.shape)}")
        B, T, C, H, W = seq_batch.shape
        if C != self.conv.in_channels:
            raise ValueError(f"Expected {self.conv.in_channels} input channels, got {C}")

        seq_batch = seq_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        self.lif.reset()
        acc = torch.zeros(B, self.conv.out_channels, H, W, device=device, dtype=torch.float32)
        for t in range(T):
            conv_out = self.conv(seq_batch[:, t])
            spikes = self.lif(conv_out)
            acc += spikes
        self.lif.reset()
        return acc


# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------
def _build_feature_split(
    raw_pos,
    raw_neg,
    n_pos: int,
    n_neg: int,
    W: int,
    H: int,
    time_bins: int,
    extractor: SNNFeatureExtractor,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    extractor.eval()

    pos_idx = _sample_indices(raw_pos, n_pos)
    neg_idx = _sample_indices(raw_neg, n_neg)

    seq_batch: List[torch.Tensor] = []
    label_batch: List[int] = []
    features: List[torch.Tensor] = []
    labels: List[int] = []

    @torch.no_grad()
    def _flush():
        if not seq_batch:
            return
        stacked = torch.stack(seq_batch, dim=0)  # (B,T,2,H,W) on CPU
        feats = extractor(stacked).cpu()
        features.append(feats)
        labels.extend(label_batch)
        seq_batch.clear()
        label_batch.clear()

    for idx in tqdm(pos_idx, desc=f"SNN features (+)", leave=False):
        seq_batch.append(window_to_voxel_sequence(raw_pos, idx, W, H, time_bins))
        label_batch.append(1)
        if len(seq_batch) >= batch_size:
            _flush()

    for idx in tqdm(neg_idx, desc=f"SNN features (-)", leave=False):
        seq_batch.append(window_to_voxel_sequence(raw_neg, idx, W, H, time_bins))
        label_batch.append(0)
        if len(seq_batch) >= batch_size:
            _flush()

    _flush()

    if not features:
        raise RuntimeError("No features were extracted; check dataset sizes.")

    X = torch.cat(features, dim=0).float().contiguous()
    y = torch.tensor(labels, dtype=torch.long)

    sums = X.sum(dim=(1, 2, 3), keepdim=True)
    sums[sums == 0.0] = 1.0
    X = X / sums

    perm = torch.randperm(X.size(0))
    X = X[perm]
    y = y[perm]
    return X, y


# ----------------------------------------------------------------------
# Linear readout (same as baseline)
# ----------------------------------------------------------------------
class LinearHead(nn.Sequential):
    def __init__(self, in_ch: int, H: int, W: int):
        super().__init__(nn.Flatten(), nn.Linear(in_ch * H * W, 2))


def train_and_eval(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    num_workers: int = 8,
    early_stopping_patience: int | None = 5,
) -> dict:
    Ntr, C, H, W = Xtr.shape
    model = LinearHead(C, H, W).to(device)
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
    te_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    print(f"[SNN-LINEAR] training: Ntr={Ntr}, Nte={Xte.size(0)}, Din={C*H*W}, epochs={epochs}")
    best_state = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")
    epochs_no_improve = 0

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

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation / early stopping evaluation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb_val, yb_val in te_loader:
                xb_val = xb_val.to(device, non_blocking=True)
                yb_val = yb_val.to(device, non_blocking=True)
                logits_val = model(xb_val)
                val_correct += (logits_val.argmax(dim=1) == yb_val).sum().item()
                val_total += yb_val.numel()
        val_acc = val_correct / val_total
        print(f"[epoch {ep:2d}] train acc={train_acc:.3f}  loss={train_loss:.4f}  val acc={val_acc:.3f}")

        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"[EARLY STOP] no val improvement for {early_stopping_patience} epochs; stopping at epoch {ep}.")
                break

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

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
    print(f"[SNN-LINEAR] test accuracy: {acc:.3f}")

    cm = torch.zeros(2, 2, dtype=torch.int64)
    for t, p in zip(y_cpu, preds):
        cm[t, p] += 1
    tn, fp, fn, tp = cm[0, 0].item(), cm[0, 1].item(), cm[1, 0].item(), cm[1, 1].item()
    print(f"[SNN-LINEAR] confusion matrix:\n"
          f"            pred 0   pred 1\n"
          f" true 0     {tn:6d}   {fp:6d}\n"
          f" true 1     {fn:6d}   {tp:6d}")

    return {"test_acc": acc, "cm": cm.tolist()}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate STDP conv features with a linear readout.")
    parser.add_argument("--weights", default="/workspace/results/stdp_full_run.pt",
                        help="Path to saved conv weights from SNN training.")
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
    parser.add_argument("--time_bins", type=int, default=5,
                        help="Temporal bins per window; must match the weights' training config.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--feature_batch", type=int, default=16,
                        help="Mini-batch size while extracting SNN features.")
    parser.add_argument("--profile_build_only", action="store_true",
                        help="Extract features, report timing, then exit without training.")

    parser.add_argument("--out_dir", default="/workspace/results")
    parser.add_argument("--run_name", default=None, help="Optional tag for filenames.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    weight_tag = weights_path.stem
    default_name = (
        f"snn_linear_{weight_tag}_T{args.time_bins}_"
        f"train{args.per_class_train}_test{args.per_class_test}_seed{args.seed}"
    )
    run_name = args.run_name or default_name

    log_path = _start_logging(args.out_dir, run_name)
    print(f"[LOG] writing to: {log_path}")

    print(f"[CFG] device={device} | weights={weights_path}")
    set_global_seed(args.seed)

    print("[SNN-LINEAR] loading caches…")
    train_cars = _load_cache(args.train_cars)
    train_bg   = _load_cache(args.train_bg)
    test_cars  = _load_cache(args.test_cars)
    test_bg    = _load_cache(args.test_bg)

    if args.W is not None and args.H is not None:
        W, H = args.W, args.H
        print(f"[SNN-LINEAR] using manual sensor: W={W}, H={H}")
    else:
        W, H = infer_sensor_size_from_cache(train_cars)
        print(f"[SNN-LINEAR] inferred sensor: W={W}, H={H}")

    extractor = SNNFeatureExtractor(str(weights_path)).to(device)
    extractor.eval()

    print("[SNN-LINEAR] building train split…")
    t0 = time.perf_counter()
    Xtr, ytr = _build_feature_split(
        train_cars,
        train_bg,
        args.per_class_train,
        args.per_class_train,
        W,
        H,
        args.time_bins,
        extractor,
        batch_size=args.feature_batch,
    )
    t1 = time.perf_counter()
    print(f"[TIME] build_train: {(t1 - t0):.2f}s  (N={Xtr.size(0)}, C={Xtr.size(1)}, H={H}, W={W})")

    print("[SNN-LINEAR] building test split…")
    t2 = time.perf_counter()
    Xte, yte = _build_feature_split(
        test_cars,
        test_bg,
        args.per_class_test,
        args.per_class_test,
        W,
        H,
        args.time_bins,
        extractor,
        batch_size=args.feature_batch,
    )
    t3 = time.perf_counter()
    print(f"[TIME] build_test : {(t3 - t2):.2f}s  (N={Xte.size(0)}, C={Xte.size(1)}, H={H}, W={W})")
    print(f"[TIME] build_total: {(t3 - t0):.2f}s")

    if args.profile_build_only:
        print("[MODE] profile_build_only → skipping training/eval.")
        return

    print(f"[DEBUG] Xtr shape = {tuple(Xtr.shape)}  (N,C,H,W) | Xte shape = {tuple(Xte.shape)}")
    with torch.no_grad():
        ch_means = Xtr.mean(dim=(0, 2, 3))
        show = min(12, ch_means.numel())
        print("[DEBUG] per-channel mean (first {}): {}".format(
            show, ", ".join(f"{v:.6f}" for v in ch_means[:show])))

    metrics = train_and_eval(
        Xtr, ytr, Xte, yte,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )

    meta = {
        "run_name": run_name,
        "log_path": log_path,
        "model": "snn_linear",
        "weights": str(weights_path),
        "W": W,
        "H": H,
        "time_bins": args.time_bins,
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
    header = ("timestamp	run_name	model	W	H	norm	time_bins	per_class_train	per_class_test	"
              "epochs	batch	lr	wd	seed	device	test_acc	cm	weights")
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}	{run_name}	snn_linear	{W}	{H}	n/a	{args.time_bins}	"
        f"{args.per_class_train}	{args.per_class_test}	{args.epochs}	{args.batch_size}	"
        f"{args.lr}	{args.weight_decay}	{args.seed}	{device.type}	{metrics['test_acc']:.6f}	"
        f"{metrics['cm']}	{weights_path}"
    )
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header + "\n")
    with open(summary_path, "a") as f:
        f.write(line + "\n")
    print(f"[APPEND] {summary_path}")


if __name__ == "__main__":
    main()