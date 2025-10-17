from pathlib import Path
# utils.py
import random
import json
import numpy as np
import torch
from datetime import datetime

# Seeds + device
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Hyperparams
ETA      = 0.05
TAU_PRE  = 2.0
TAU_POST = 2.0
VTH      = 0.5

# Geometry
H = W = 9
K = 3
PAD = K // 2
cy, cx = H // 2, W // 2

# Regularization
ALPHA_DECAY = 2e-3  # gated decay (applied only when STDP produces a nonzero ΔW)


# Dataset geometry (NCARS sensor is 120×100)
NCARS_SENSOR_SIZE = (120, 100)

def _assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

def _assert_ge(x: float, thr: float, msg: str):
    if x < thr:
        raise AssertionError(msg)

def _apply_decay(conv_weight):
    """Tiny uniform shrinkage; applied only when plasticity happened."""
    if ALPHA_DECAY > 0.0:
        conv_weight.mul_(1.0 - ALPHA_DECAY)

# reporting.py
def col_fracs_from_kernel(W3x3):
    k = W3x3.clamp_min(0)
    s = k.sum() + 1e-8
    col_sums = k.sum(dim=0)          # [3]
    fracs = (col_sums / s).detach().cpu().numpy()
    return fracs, col_sums.detach().cpu().numpy()

def print_delta_matrix(name, W_before, W_after):
    dW = (W_after - W_before).detach().cpu().numpy()
    Wb = W_before.detach().cpu().numpy()
    Wa = W_after.detach().cpu().numpy()
    print(f"\n{name}")
    print("W before (3x3):\n", Wb)
    print("W after  (3x3):\n", Wa)
    print("ΔW       (3x3):\n", dW)
    print("Δ column sums [L,C,R]:", dW.sum(axis=0))
    return dW

def decode_addr_to_xy_p(addr: np.ndarray):
    """
    Decode Prophesee NCARS event addresses into (x, y, polarity).
    Returns three arrays of same length.
    """
    x = addr & 0x00003FFF
    y = (addr & 0x0FFFC000) >> 14
    p = -1 + 2 * ((addr & 0x10000000) >> 28).astype(np.int32)
    return x, y, p

def infer_sensor_size_from_cache(raw, max_files_to_scan=50):
    """
    Inspect cached NCARS raw data to infer sensor width (W) and height (H).
    """
    max_x, max_y = 0, 0
    for i, d in enumerate(raw["data"][:max_files_to_scan]):
        if len(d["addr"]) == 0:
            continue
        x, y, _ = decode_addr_to_xy_p(d["addr"])
        max_x = max(max_x, x.max())
        max_y = max(max_y, y.max())
    W = int(max_x + 1)
    H = int(max_y + 1)

    full_W, full_H = NCARS_SENSOR_SIZE
    if W < full_W or H < full_H:
        return full_W, full_H
    return W, H


def window_to_voxel_sequence(raw, idx: int, W: int, H: int, time_bins: int) -> torch.Tensor:
    """
    Build a voxel tensor (T, 2, H, W) for one indexed window.
    Channels correspond to polarity (negative, positive).
    """
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

def window_to_pol_hist(raw, sample_idx, W, H):
    """
    Build a polarity histogram (H×W×2) for one indexed window.
    Channels: [0] = negative polarity, [1] = positive polarity.
    """
    file_idx, t_start, t_end, _ = raw["index"][sample_idx]
    ts = raw["data"][file_idx]["ts"]
    addr = raw["data"][file_idx]["addr"]
    mask = (ts >= t_start) & (ts < t_end)
    x, y, p = decode_addr_to_xy_p(addr[mask])

    hist = np.zeros((H, W, 2), dtype=np.int32)
    for xi, yi, pi in zip(x, y, p):
        if 0 <= xi < W and 0 <= yi < H:
            ch = 1 if pi > 0 else 0
            hist[yi, xi, ch] += 1
    return hist

def log_run(logs_dir: Path, run_name: str, meta: dict, metrics: dict, summary_fields: list[str]):
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta = meta.copy()
    meta.setdefault('timestamp', timestamp)
    meta.setdefault('run_name', run_name)

    log_path = logs_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump({"meta": meta, "metrics": metrics}, f, indent=2)

    summary_path = logs_dir / 'summary.tsv'
    header = '	'.join(summary_fields)
    line = '	'.join(str(meta.get(field, metrics.get(field, ''))) for field in summary_fields)
    if not summary_path.exists():
        with open(summary_path, 'w') as f:
            f.write(header + "\n")
    with open(summary_path, 'a') as f:
        f.write(line + "\n")
    return log_path, summary_path
