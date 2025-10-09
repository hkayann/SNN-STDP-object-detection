# snn_minimal_check.py
import torch
import numpy as np
from pathlib import Path
import types
import sys
import logging
from datetime import datetime
from itertools import zip_longest

# Stub tensorboard dependency used by SpikingJelly's monitor module
sys.modules.setdefault("tensorboard", types.ModuleType("tensorboard"))
torch_tb = types.ModuleType("torch.utils.tensorboard")


class _DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


torch_tb.SummaryWriter = _DummyWriter
sys.modules["torch.utils.tensorboard"] = torch_tb

from spikingjelly.activation_based import neuron, learning
from utils import device, infer_sensor_size_from_cache, decode_addr_to_xy_p

TRAIN_CARS = "caches/ncars_train_cars_win50ms.pt"
TRAIN_BG = "caches/ncars_train_background_win50ms.pt"
TIME_BINS = 5
NUM_WINDOWS = 500
WARMUP_WINDOWS = 200
DEBUG_WINDOW_LIMIT = 120
COMP_DEBUG_WINDOWS = 10
ETA = 1e-5
LOGGER = logging.getLogger("snn_minimal_check")


def setup_logger(log_path: str | None = None) -> logging.Logger:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)

    LOGGER.propagate = False
    return LOGGER


def _init_default_logger() -> None:
    if not LOGGER.handlers:
        setup_logger(None)


def load_cache(path: str):
    return torch.load(path, weights_only=False)


def sample_idx(raw):
    return int(torch.randint(len(raw["index"]), (1,)))


def build_voxel(raw, idx, W, H, T):
    file_idx, t0, t1, _ = raw["index"][idx]
    ts = raw["data"][file_idx]["ts"]
    addr = raw["data"][file_idx]["addr"]
    mask = (ts >= t0) & (ts < t1)
    if not np.any(mask):
        return torch.zeros(T, 2, H, W)
    ts_win, addr_win = ts[mask], addr[mask]
    x, y, p = decode_addr_to_xy_p(addr_win)
    bins = np.clip(np.floor((ts_win - t0) * T / max(1, t1 - t0)).astype(np.int64), 0, T - 1)
    vox = np.zeros((T, 2, H, W), dtype=np.float32)
    for xi, yi, pi, bi in zip(x, y, (p > 0).astype(np.int64), bins):
        if 0 <= xi < W and 0 <= yi < H:
            vox[bi, pi, yi, xi] += 1.0
    return torch.from_numpy(vox)


@torch.no_grad()
def save_weight_grid(weights: torch.Tensor, path: str):
    import matplotlib.pyplot as plt
    import os

    w_cpu = weights.detach().cpu()
    out_ch = w_cpu.shape[0]
    cols = max(1, min(8, out_ch))
    rows = (out_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    vmin = w_cpu.min().item()
    vmax = w_cpu.max().item()

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis('off')
        if idx < out_ch:
            img = w_cpu[idx, 1] - w_cpu[idx, 0]
            ax.imshow(img.numpy(), cmap='bwr', vmin=vmin, vmax=vmax)
            ax.set_title(f"f{idx}", fontsize=10)
    fig.tight_layout()
    os.makedirs(Path(path).parent, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


class TinySTDP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 3, padding=1, bias=False)
        torch.nn.init.constant_(self.conv.weight, 0.05)
        self.conv.weight.add_(2e-3 * torch.randn_like(self.conv.weight))
        self.sn = neuron.LIFNode(tau=2.0, v_threshold=0.01, v_reset=0.0, detach_reset=True)
        self.stdp = learning.STDPLearner(step_mode='s', synapse=self.conv, sn=self.sn,
                                         tau_pre=2.0, tau_post=2.0)

    def reset_state(self):
        self.sn.reset()
        self.stdp.reset()


@torch.no_grad()
def run(num_windows: int = NUM_WINDOWS, save_weights: str | None = None,
        full_train: bool = False, logger: logging.Logger | None = None):
    _init_default_logger()
    logger = logger or LOGGER

    pos = load_cache(TRAIN_CARS)
    neg = load_cache(TRAIN_BG)
    W, H = infer_sensor_size_from_cache(pos)

    if full_train:
        torch.manual_seed(42)
        np.random.seed(42)
        pos_order = torch.randperm(len(pos["index"])).tolist()
        neg_order = torch.randperm(len(neg["index"])).tolist()
        total_windows = len(pos_order) + len(neg_order)

        def stream_iter():
            for pos_idx, neg_idx in zip_longest(pos_order, neg_order):
                if pos_idx is not None:
                    yield build_voxel(pos, pos_idx, W, H, TIME_BINS)
                if neg_idx is not None:
                    yield build_voxel(neg, neg_idx, W, H, TIME_BINS)
    else:
        if num_windows <= 0:
            raise ValueError("num_windows must be positive when full_train is False.")
        total_windows = num_windows * 2

        def stream_iter():
            for _ in range(num_windows):
                yield build_voxel(pos, sample_idx(pos), W, H, TIME_BINS)
                yield build_voxel(neg, sample_idx(neg), W, H, TIME_BINS)

    logger.info(
        f"[CFG] device={device} | T={TIME_BINS} | WARMUP={WARMUP_WINDOWS} "
        f"| full_train={full_train} | save_weights={bool(save_weights)}"
    )

    model = TinySTDP().to(device)
    win_counts = torch.zeros(model.conv.out_channels, dtype=torch.long, device=device)
    silent_windows = 0
    first_silent_win = None
    comp_debug_remaining = COMP_DEBUG_WINDOWS
    target_per = None

    for win_id, seq in enumerate(stream_iter(), 1):
        if win_id == WARMUP_WINDOWS + 1:
            win_counts.zero_()

        if win_id <= DEBUG_WINDOW_LIMIT:
            voxel_sums = seq.sum(dim=(0, 2, 3)).tolist()
            logger.info(
                f"[DBG][vox] window={win_id} sum_pos={voxel_sums[0]:.1f} sum_neg={voxel_sums[1]:.1f}"
            )

        seq = seq.to(device)
        total_spikes = 0.0
        comp_scores = torch.zeros(model.conv.out_channels, device=device)

        for t in range(seq.shape[0]):
            x = seq[t].unsqueeze(0)
            conv_out = model.conv(x)
            spikes_t = model.sn(conv_out)
            spikes_now = float(spikes_t.sum().item())
            total_spikes += spikes_now

            if win_id > WARMUP_WINDOWS and spikes_now > 0.0:
                spatial = spikes_t.view(spikes_t.size(0), spikes_t.size(1), -1)
                scores = spatial.sum(dim=-1)
                scores_det = scores.clone()
                scores_sel = scores_det + 1e-3 * torch.randn_like(scores_det)
                comp_scores += scores_sel.squeeze(0)
                winner = scores_sel.argmax(dim=1, keepdim=True)
                win_counts[winner.view(-1)] += 1
                mask = torch.zeros_like(spatial)
                mask.scatter_(1, winner.unsqueeze(-1), 1.0)
                spikes_t = (spatial * mask).view_as(spikes_t)
                if len(model.stdp.out_spike_monitor.records) > 0:
                    model.stdp.out_spike_monitor.records[0] = spikes_t

                with torch.no_grad():
                    inhibition_strength = 0.005
                    winner_idx = winner.view(-1).item()
                    channel_dim = model.sn.v.size(1)
                    if channel_dim > 1:
                        loser_idx = [i for i in range(channel_dim) if i != winner_idx]
                        if loser_idx:
                            model.sn.v[:, loser_idx] -= inhibition_strength

            delta_w = model.stdp.step(on_grad=False)
            if spikes_now > 0.0 and delta_w is not None:
                model.conv.weight.add_(ETA * delta_w)
                model.conv.weight.clamp_(0.0, 1.0)

        if win_id > WARMUP_WINDOWS and comp_debug_remaining > 0 and comp_scores.sum().item() > 0:
            mean_score = float(comp_scores.mean().item())
            topk = torch.topk(comp_scores, k=min(2, comp_scores.numel()))
            best = float(topk.values[0].item())
            runner_up = float(topk.values[1].item()) if topk.values.numel() > 1 else 0.0
            margin = best - runner_up
            norm_margin = 0.0 if mean_score == 0 else margin / max(mean_score, 1e-8)
            logger.info(
                f"[DBG][comp] window={win_id} mean={mean_score:.4f} max={best:.4f} "
                f"runner_up={runner_up:.4f} margin={margin:.4f} norm_margin={norm_margin:.4f}"
            )
            comp_debug_remaining -= 1

        if win_id <= DEBUG_WINDOW_LIMIT:
            logger.info(f"[DBG][spk] window={win_id} total_spikes={total_spikes:.1f}")

        with torch.no_grad():
            Ww = model.conv.weight
            mass = Ww.view(Ww.size(0), -1).abs().sum(dim=1, keepdim=True) + 1e-8
            if target_per is None:
                base = Ww[0].numel() * 0.05
                jitter = torch.empty(Ww.size(0), 1, 1, 1, device=Ww.device).normal_(0, 0.05)
                target_per = (base * (1.0 + jitter).clamp(0.8, 1.2)).clone()
            scale = (target_per.view(-1, 1, 1, 1) / mass.view(-1, 1, 1, 1)).clamp(0.5, 2.0)
            Ww.mul_(scale)

        model.reset_state()

        if win_id <= WARMUP_WINDOWS:
            silent_windows = 0
        else:
            if total_spikes == 0:
                silent_windows += 1
                if first_silent_win is None:
                    first_silent_win = win_id
                    logger.warning(f"[DBG][silent] first_zero_spike_window={first_silent_win}")
            else:
                silent_windows = 0

        logged_wta = False
        if win_id % 100 == 0 or win_id == total_windows:
            wc_tensor = win_counts.detach().cpu()
            total = float(wc_tensor.sum().item())
            entropy = 0.0
            if total > 0:
                probs = wc_tensor / total
                entropy = float((-probs[probs > 0] * torch.log2(probs[probs > 0])).sum().item())
            wc_list = wc_tensor.tolist()
            logger.info(f"[WTA] win_counts={wc_list} sum={total:.0f} entropy={entropy:.4f}")
            logged_wta = True

        if silent_windows >= 20:
            if not logged_wta:
                wc = win_counts.detach().cpu().tolist()
                logger.info(f"[WTA] win_counts={wc} sum={sum(wc)}")
            logger.warning(f"[EARLY-STOP] {silent_windows} silent windows in a row — stopping.")
            break

    save_weight_grid(model.conv.weight, '/workspace/figures/snn_filters.png')

    w = model.conv.weight.detach().cpu()
    if save_weights:
        Path(save_weights).parent.mkdir(parents=True, exist_ok=True)
        torch.save(w, save_weights)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unsupervised STDP smoke test")
    parser.add_argument("--num_windows", type=int, default=NUM_WINDOWS,
                        help="Number of car/background pairs to stream (ignored when --full_train is set)")
    parser.add_argument("--save_weights", type=str, default="/workspace/results/stdp_conv_weights.pt",
                        help="Where to save learned conv weights (set empty string to skip)")
    parser.add_argument("--full_train", action="store_true",
                        help="Stream every cached train window once without replacement")
    parser.add_argument("--log_path", type=str, default="",
                        help="Optional path for the log file (defaults to logs/snn_minimal_check_<timestamp>.log)")
    args = parser.parse_args()

    save_path = args.save_weights if args.save_weights else None
    if args.log_path:
        log_path = args.log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = f"logs/snn_minimal_check_{stamp}.log"

    logger = setup_logger(log_path)

    run(num_windows=args.num_windows, save_weights=save_path, full_train=args.full_train, logger=logger)
