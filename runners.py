# runners.py
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, learning

from utils import (
    device, ETA, TAU_PRE, TAU_POST, VTH,
    H, W, K, PAD, cy, cx
)
from utils import (
    _apply_decay, _assert_ge, _assert_true,
    col_fracs_from_kernel, print_delta_matrix
)
from stimuli import (
    make_center_impulse, make_impulse_at_x,
    make_center_minibar, make_minibar_at_x, make_random_impulse
)

# ---------- Core runner (2 steps: t0 then t1) ----------
@torch.no_grad()
def run_two_step_pair(pre_t0, post_t0, pre_t1, post_t1, title):
    conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(
        step_mode='s', synapse=conv, sn=sn,
        tau_pre=TAU_PRE, tau_post=TAU_POST,
        f_pre=lambda w: ETA, f_post=lambda w: ETA
    )
    conv.weight.fill_(0.10)

    def step(pre_img, teacher_img):
        _ = sn(conv(pre_img))  # record spikes
        if len(stdp.out_spike_monitor.records) > 0:
            stdp.out_spike_monitor.records[0] = teacher_img
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()

    W0 = conv.weight[0, 0].clone()
    step(pre_t0, post_t0)
    step(pre_t1, post_t1)
    W1 = conv.weight[0, 0].clone()

    _ = print_delta_matrix(title, W0, W1)
    return W0, W1, conv

# ---------- Amplify causal pairs (1 out channel) ----------
@torch.no_grad()
def repeat_causal_pairs(n_pairs=200, progress_every=40):
    conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(
        step_mode='s', synapse=conv, sn=sn,
        tau_pre=TAU_PRE, tau_post=TAU_POST,
        f_pre=lambda w: ETA, f_post=lambda w: ETA
    )
    conv.weight.fill_(0.10)

    pre_bar = make_center_minibar(H, W)
    post_impulse = make_center_impulse(H, W)
    zeros = torch.zeros_like(pre_bar)

    def step(pre_img, teacher_img):
        _ = sn(conv(pre_img))
        if len(stdp.out_spike_monitor.records) > 0:
            stdp.out_spike_monitor.records[0] = teacher_img
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()

    W_start = conv.weight[0, 0].clone()
    start_fracs, _ = col_fracs_from_kernel(W_start)
    print("Initial column fractions [L,C,R]:", [f"{v:.4f}" for v in start_fracs])

    for i in range(1, n_pairs + 1):
        step(pre_bar, zeros)         # t0
        step(zeros,  post_impulse)   # t1

        if i % progress_every == 0 or i == 1 or i == n_pairs:
            W_cur = conv.weight[0, 0]
            fracs, col_sums = col_fracs_from_kernel(W_cur)
            wmin = conv.weight.min().item()
            wmax = conv.weight.max().item()
            print(f"[pair {i:4d}] col_fracs(L,C,R)={['%.4f'%v for v in fracs]}  "
                  f"w[min,max]=[{wmin:.4f},{wmax:.4f}]  "
                  f"(center_col_sum={col_sums[1]:.6f})")

    W_end = conv.weight[0, 0].clone()
    end_fracs, _ = col_fracs_from_kernel(W_end)
    print("\nFinal 3x3 kernel:\n", W_end.detach().cpu().numpy())
    print("Final column fractions [L,C,R]:", [f"{v:.4f}" for v in end_fracs])
    print("Center column fraction increased? ", "YES" if end_fracs[1] > start_fracs[1] else "NO")
    _assert_ge(end_fracs[1], 0.38, f"repeat_causal_pairs: center fraction too low ({end_fracs[1]:.4f})")

    return W_start, W_end

# ---------- Two-channel + WTA competition ----------
@torch.no_grad()
def repeat_causal_pairs_two_channels_wta(n_pairs=200, progress_every=40):
    out_ch = 2
    conv = nn.Conv2d(1, out_ch, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(
        step_mode='s', synapse=conv, sn=sn,
        tau_pre=TAU_PRE, tau_post=TAU_POST,
        f_pre=lambda w: ETA, f_post=lambda w: ETA
    )
    conv.weight.fill_(0.10)
    conv.weight.add_(1e-3 * torch.randn_like(conv.weight))  # tiny symmetry breaker

    pre_bar = make_center_minibar(H, W)
    post_impulse_single = make_center_impulse(H, W)
    zeros = torch.zeros_like(pre_bar)
    wins = [0, 0]

    def step_t0_with_wta(pre_img):
        _ = sn(conv(pre_img))
        rec = stdp.out_spike_monitor.records[0]  # [1,2,H,W]
        cvals = rec[0, :, cy, cx]
        win_c = int(torch.argmax(cvals).item())
        wins[win_c] += 1
        lose_c = 1 - win_c
        rec[0, lose_c, :, :] = 0.0
        stdp.out_spike_monitor.records[0] = rec
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()
        return win_c

    def step_t1_teacher_for_winner(win_c):
        _ = sn(conv(zeros))
        rec = stdp.out_spike_monitor.records[0]
        teacher = torch.zeros_like(rec)
        teacher[0, win_c, :, :] = post_impulse_single[0, 0, :, :]
        stdp.out_spike_monitor.records[0] = teacher
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()

    def per_channel_col_fracs():
        fracs = []
        col_sums = []
        for c in range(out_ch):
            W3 = conv.weight[c, 0]
            f, cs = col_fracs_from_kernel(W3)
            fracs.append(f)
            col_sums.append(cs)
        return fracs, col_sums

    f0, _ = per_channel_col_fracs()
    print("Initial col fractions ch0[L,C,R]:", [f"{v:.4f}" for v in f0[0]])
    print("Initial col fractions ch1[L,C,R]:", [f"{v:.4f}" for v in f0[1]])

    for i in range(1, n_pairs + 1):
        winner = step_t0_with_wta(pre_bar)
        step_t1_teacher_for_winner(winner)
        if i % progress_every == 0 or i == 1 or i == n_pairs:
            fracs, _ = per_channel_col_fracs()
            wmin = conv.weight.min().item()
            wmax = conv.weight.max().item()
            print(f"[pair {i:4d}] ch0 col_fracs={['%.4f'%v for v in fracs[0]]}  "
                  f"ch1 col_fracs={['%.4f'%v for v in fracs[1]]}  "
                  f"wins(ch0,ch1)=({wins[0]},{wins[1]})  "
                  f"w[min,max]=[{wmin:.4f},{wmax:.4f}]")

    W0 = conv.weight[0, 0].detach().cpu().numpy()
    W1 = conv.weight[1, 0].detach().cpu().numpy()
    f_end, _ = per_channel_col_fracs()
    print("\nFinal 3x3 kernels (in0->out0, in0->out1):")
    print("W[0,0]:\n", W0)
    print("W[1,0]:\n", W1)
    print("Final col fractions ch0[L,C,R]:", [f"{v:.4f}" for v in f_end[0]])
    print("Final col fractions ch1[L,C,R]:", [f"{v:.4f}" for v in f_end[1]])
    print(f"Win counts → ch0: {wins[0]}, ch1: {wins[1]}")
    sep = abs(f_end[0][1] - f_end[1][1])
    print("PASS ✅" if sep >= 0.05 else "WARN: weak separation")

# ---------- NULL control ----------
@torch.no_grad()
def null_pair_should_do_nothing():
    conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(step_mode='s', synapse=conv, sn=sn,
                                tau_pre=TAU_PRE, tau_post=TAU_POST,
                                f_pre=lambda w: ETA, f_post=lambda w: ETA)
    conv.weight.fill_(0.10)
    zeros = torch.zeros(1, 1, H, W, device=device)

    def step(pre_img, teacher_img):
        _ = sn(conv(pre_img))
        if len(stdp.out_spike_monitor.records) > 0:
            stdp.out_spike_monitor.records[0] = teacher_img
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()

    W0 = conv.weight[0, 0].clone()
    step(zeros, zeros)
    step(zeros, zeros)
    W1 = conv.weight[0, 0].clone()
    dW = (W1 - W0).abs().max().item()
    print(f"\nNULL control: max|ΔW| = {dW:.8f} (expect ≈ 0)  → {'PASS ✅' if dW < 1e-8 else 'FAIL ❌'}")
    _assert_true(dW < 1e-8, f"NULL control drifted: max|ΔW|={dW:.8e}")

# ---------- Noise-only pre ----------
@torch.no_grad()
def noise_pre_no_teacher(n_pairs=200, progress_every=40):
    conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(step_mode='s', synapse=conv, sn=sn,
                                tau_pre=TAU_PRE, tau_post=TAU_POST,
                                f_pre=lambda w: ETA, f_post=lambda w: ETA)
    conv.weight.fill_(0.10)
    zeros = torch.zeros(1, 1, H, W, device=device)

    def step(pre_img, teacher_img):
        _ = sn(conv(pre_img))
        if len(stdp.out_spike_monitor.records) > 0:
            stdp.out_spike_monitor.records[0] = teacher_img
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()

    W_start = conv.weight[0, 0].clone()
    start_fracs, _ = col_fracs_from_kernel(W_start)
    print("Noise-only: initial column fractions [L,C,R]:", [f"{v:.4f}" for v in start_fracs])

    for i in range(1, n_pairs + 1):
        pre = make_random_impulse(H, W)
        step(pre, zeros)
        step(zeros, zeros)
        if i % progress_every == 0 or i == 1 or i == n_pairs:
            W_cur = conv.weight[0, 0]
            fracs, _ = col_fracs_from_kernel(W_cur)
            wmin = conv.weight.min().item()
            wmax = conv.weight.max().item()
            print(f"[noise {i:4d}] col_fracs(L,C,R)={['%.4f'%v for v in fracs]}  "
                  f"w[min,max]=[{wmin:.4f},{wmax:.4f}]")

    W_end = conv.weight[0, 0].clone()
    end_fracs, _ = col_fracs_from_kernel(W_end)
    drift = abs(end_fracs[1] - 1/3)
    print("Noise-only: final column fractions [L,C,R]:", [f"{v:.4f}" for v in end_fracs])
    print(f"Noise-only center drift from 1/3: {drift:.4f}  → {'OK ✅' if drift < 0.03 else 'WARN ❗'}")

# ---------- Fixed-teacher 3-offset mapping ----------
@torch.no_grad()
def fixed_teacher_three_offsets(n_pairs=120, progress_every=30, clamp=True):
    results = []
    for label, xcol, expect_idx in [
        ("LEFT_pre  (x=cx-1)", cx-1, 0),
        ("CENTER_pre(x=cx  )", cx,   1),
        ("RIGHT_pre (x=cx+1)", cx+1, 2),
    ]:
        conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
        sn   = neuron.IFNode(v_threshold=VTH).to(device)
        stdp = learning.STDPLearner(step_mode='s', synapse=conv, sn=sn,
                                    tau_pre=TAU_PRE, tau_post=TAU_POST,
                                    f_pre=lambda w: ETA, f_post=lambda w: ETA)
        conv.weight.fill_(0.10)

        pre_bar = make_minibar_at_x(H, W, xcol)
        post_center = make_center_impulse(H, W)
        zeros = torch.zeros_like(pre_bar)

        def step(pre_img, teacher_img):
            _ = sn(conv(pre_img))
            if len(stdp.out_spike_monitor.records) > 0:
                stdp.out_spike_monitor.records[0] = teacher_img
            dw = stdp.step(on_grad=False)
            if dw is not None:
                conv.weight += dw
                if dw.abs().sum().item() > 0.0:
                    _apply_decay(conv.weight)
            if clamp:
                conv.weight.clamp_(0.0, 1.0)
            sn.reset()

        W0 = conv.weight[0, 0].clone()
        for i in range(1, n_pairs + 1):
            step(pre_bar, zeros)
            step(zeros,  post_center)
            if i % progress_every == 0 or i == 1 or i == n_pairs:
                W_cur = conv.weight[0,0]
                fracs, col_sums = col_fracs_from_kernel(W_cur)
                print(f"[map {label:16s} | pair {i:3d}] col_fracs(L,C,R)={['%.4f'%v for v in fracs]}  "
                      f"(col_sum={['%.6f'%v for v in col_sums]})")

        W1 = conv.weight[0, 0].clone()
        dcol = (W1 - W0).sum(dim=0).detach().cpu().numpy()
        winner = int(dcol.argmax())
        passed = (winner == expect_idx)
        print(f"\nMapping result for {label}: Δcols[L,C,R] = {dcol}  → "
              f"winner idx={winner} ({['LEFT','CENTER','RIGHT'][winner]})  "
              f"{'PASS ✅' if passed else 'FAIL ❌'}\n")
        results.append((label, dcol, passed))

    all_pass = all(p for _, _, p in results)
    print("Fixed-teacher mapping summary:",
          "ALL PASS ✅" if all_pass else "Some FAIL ❌")
    return results

# ---------- NEW (H): Δt sweep for STDP ----------
@torch.no_grad()
def stdp_dt_sweep(max_dt=5):
    """
    Sweep over relative timing Δt between pre and post.
    - Negative Δt: pre before post (causal) → expect LTP.
    - Positive Δt: post before pre (anti)   → expect LTD.
    Each run starts from fresh weights so results are isolated.
    """
    import numpy as np
    from utils import device, H, W, K, PAD, ETA, TAU_PRE, TAU_POST, VTH
    from stimuli import make_center_minibar, make_center_impulse
    from utils import print_delta_matrix

    pre_bar = make_center_minibar(H, W)
    post_center = make_center_impulse(H, W)
    zeros = torch.zeros_like(pre_bar)

    results = {}

    for dt in range(-max_dt, max_dt + 1):
        # fresh conv+stdp each time
        conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
        sn   = neuron.IFNode(v_threshold=VTH).to(device)
        stdp = learning.STDPLearner(
            step_mode='s', synapse=conv, sn=sn,
            tau_pre=TAU_PRE, tau_post=TAU_POST,
            f_pre=lambda w: ETA, f_post=lambda w: ETA
        )
        conv.weight.fill_(0.10)

        def step(pre_img, teacher_img):
            _ = sn(conv(pre_img))
            if len(stdp.out_spike_monitor.records) > 0:
                stdp.out_spike_monitor.records[0] = teacher_img
            dw = stdp.step(on_grad=False)
            if dw is not None:
                conv.weight += dw
            sn.reset()

        W0 = conv.weight[0, 0].clone()

        if dt < 0:
            # pre first, then |dt| steps later post
            step(pre_bar, zeros)
            for _ in range(abs(dt) - 1):
                step(zeros, zeros)
            step(zeros, post_center)
        elif dt > 0:
            # post first, then dt steps later pre
            step(zeros, post_center)
            for _ in range(dt - 1):
                step(zeros, zeros)
            step(pre_bar, zeros)
        else:
            # simultaneous
            step(pre_bar, post_center)

        W1 = conv.weight[0, 0].clone()
        dW = (W1 - W0).sum().item()
        results[dt] = dW

        print(f"[Δt={dt:+d}] net ΔW sum = {dW:.6f}")

    # summary: make a rough STDP curve
    print("\nΔt sweep summary (Δt vs net ΔW):")
    for dt in sorted(results.keys()):
        print(f"Δt={dt:+d} → ΔW_sum={results[dt]:+.6f}")

    return results

# --- ADD to runners.py ---
@torch.no_grad()
def stdp_dt_sweep_numeric(max_dt=5):
    """
    Wrapper around stdp_dt_sweep that prints a clean numeric table
    and summary metrics (no CSV, no plotting).
    """
    results = stdp_dt_sweep(max_dt=max_dt)  # reuse your existing function

    # Ordered lists
    dt_list = sorted(results.keys())
    dw_sums = [results[dt] for dt in dt_list]

    print("\nNumeric STDP window (Δt vs net ΔW):")
    for dt, dw in zip(dt_list, dw_sums):
        print(f"Δt={dt:+d} → ΔW_sum={dw:+.6f}")

    # Summary metrics
    ltp_vals = [results[dt] for dt in dt_list if dt < 0]
    ltd_vals = [results[dt] for dt in dt_list if dt > 0]
    peak_ltp = max(ltp_vals) if ltp_vals else 0.0
    peak_ltd = min(ltd_vals) if ltd_vals else 0.0
    area_ltp = sum(v for v in ltp_vals if v > 0)
    area_ltd = sum(-v for v in ltd_vals if v < 0)  # positive magnitude
    symmetry = (min(area_ltp, area_ltd) / max(area_ltp, area_ltd)) if max(area_ltp, area_ltd) > 0 else 1.0

    print("\nSummary:")
    print(f"  peak LTP (Δt<0): +{peak_ltp:.6f}")
    print(f"  peak LTD (Δt>0): {peak_ltd:.6f}")
    print(f"  area LTP        : {area_ltp:.6f}")
    print(f"  area LTD        : {area_ltd:.6f}")
    print(f"  symmetry        : {symmetry:.3f}")

    return results

# ---------- PNG heatmap for 2-ch WTA: ΔW per synapse ----------
@torch.no_grad()
def two_channel_wta_dW_heatmap_png(n_pairs=200, save_path="/workspace/figure/dW_heatmap.png"):
    """
    Trains the 2-channel WTA setup, then saves a PNG heatmap of ΔW
    (W_end - W_start) for each channel's 3x3 kernel (18 synapses total).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    out_ch = 2
    conv = nn.Conv2d(1, out_ch, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(
        step_mode='s', synapse=conv, sn=sn,
        tau_pre=TAU_PRE, tau_post=TAU_POST,
        f_pre=lambda w: ETA, f_post=lambda w: ETA
    )
    conv.weight.fill_(0.10)
    conv.weight.add_(1e-3 * torch.randn_like(conv.weight))  # tiny symmetry breaker

    pre_bar = make_center_minibar(H, W)
    post_impulse_single = make_center_impulse(H, W)
    zeros = torch.zeros_like(pre_bar)

    W_start = conv.weight.clone()

    def step_t0_with_wta(pre_img):
        _ = sn(conv(pre_img))
        rec = stdp.out_spike_monitor.records[0]
        cvals = rec[0, :, cy, cx]
        win_c = int(torch.argmax(cvals).item())
        lose_c = 1 - win_c
        rec[0, lose_c, :, :] = 0.0
        stdp.out_spike_monitor.records[0] = rec
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
        sn.reset()
        return win_c

    def step_t1_teacher_for_winner(win_c):
        _ = sn(conv(zeros))
        rec = stdp.out_spike_monitor.records[0]
        teacher = torch.zeros_like(rec)
        teacher[0, win_c, :, :] = post_impulse_single[0, 0, :, :]
        stdp.out_spike_monitor.records[0] = teacher
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
        sn.reset()

    for _ in range(1, n_pairs + 1):
        winner = step_t0_with_wta(pre_bar)
        step_t1_teacher_for_winner(winner)

    W_end = conv.weight.clone()
    dW0 = (W_end[0, 0] - W_start[0, 0]).detach().cpu().numpy()
    dW1 = (W_end[1, 0] - W_start[1, 0]).detach().cpu().numpy()

    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    vmax = max(np.abs(dW0).max(), np.abs(dW1).max())  # symmetric colorbar
    for i, (dW, ax) in enumerate(zip([dW0, dW1], axes)):
        im = ax.imshow(dW, cmap="bwr", vmin=-vmax, vmax=+vmax)
        ax.set_title(f"ΔW channel {i}")
        for (r, c), val in np.ndenumerate(dW):
            ax.text(c, r, f"{val:.3f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle("ΔW Heatmaps (3x3 kernels)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ΔW heatmap PNG → {save_path}")