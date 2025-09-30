# utils.py
import torch
import random
import numpy as np

# Seeds + device
torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return W, H

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