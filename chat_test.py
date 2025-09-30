# chat_test.py
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, learning
import random

torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Tiny helpers (assert + optional decay) ----------
ALPHA_DECAY = 2e-3  # 0.2% shrink; now gated so NULL control stays unchanged

def _assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

def _assert_ge(x: float, thr: float, msg: str):
    if x < thr:
        raise AssertionError(msg)

def _apply_decay(conv_weight: torch.Tensor):
    """Tiny L2-like shrinkage to avoid runaway; acts uniformly on all weights."""
    if ALPHA_DECAY > 0.0:
        conv_weight.mul_(1.0 - ALPHA_DECAY)

# ---------- Hyperparams ----------
ETA      = 0.05   # large so ΔW is easy to see; we add dw directly (no optimizer)
TAU_PRE  = 2.0
TAU_POST = 2.0
VTH      = 0.5

H = W = 9         # small canvas with clear center
K = 3             # 3x3 kernel
PAD = K // 2      # same spatial size
cy, cx = H // 2, W // 2

# ---------- Stimuli ----------
@torch.no_grad()
def make_center_impulse(h, w):
    x = torch.zeros(1, 1, h, w, device=device)
    x[0, 0, cy, cx] = 1.0
    return x  # [1,1,H,W]

@torch.no_grad()
def make_impulse_at_x(h, w, xcol):
    xcol = max(0, min(w-1, xcol))
    img = torch.zeros(1, 1, h, w, device=device)
    img[0, 0, cy, xcol] = 1.0
    return img

@torch.no_grad()
def make_center_minibar(h, w):
    """
    Ones at (cy-1,cx), (cy,cx), (cy+1,cx).
    The 3x3 receptive field at the center sees a vertical column of ones.
    """
    x = torch.zeros(1, 1, h, w, device=device)
    x[0, 0, cy-1, cx] = 1.0
    x[0, 0, cy,   cx] = 1.0
    x[0, 0, cy+1, cx] = 1.0
    return x  # [1,1,H,W]

@torch.no_grad()
def make_minibar_at_x(h, w, xcol):
    """
    Same as make_center_minibar but shifted to column xcol.
    """
    xcol = max(1, min(w-2, xcol))  # keep vertical 3-pixel bar inside image
    x = torch.zeros(1, 1, h, w, device=device)
    x[0, 0, cy-1, xcol] = 1.0
    x[0, 0, cy,   xcol] = 1.0
    x[0, 0, cy+1, xcol] = 1.0
    return x

@torch.no_grad()
def make_random_impulse(h, w):
    y = random.randrange(0, h)
    x = random.randrange(0, w)
    img = torch.zeros(1, 1, h, w, device=device)
    img[0, 0, y, x] = 1.0
    return img

# ---------- Reporting helpers ----------
def col_fracs_from_kernel(W3x3: torch.Tensor):
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

# ---------- Core runner (2 steps: t0 then t1) ----------
@torch.no_grad()
def run_two_step_pair(pre_t0, post_t0, pre_t1, post_t1, title):
    conv = nn.Conv2d(1, 1, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(
        step_mode='s', synapse=conv, sn=sn,
        tau_pre=TAU_PRE, tau_post=TAU_POST,
        # Library: pre-term has an internal minus. Setting both to +ETA yields LTD on pre-term,
        # LTP on post-term → causal (pre@t0 → post@t1) is net potentiation.
        f_pre=lambda w: ETA, f_post=lambda w: ETA
    )
    conv.weight.fill_(0.10)

    def step(pre_img, teacher_img):
        _ = sn(conv(pre_img))                       # record spikes
        if len(stdp.out_spike_monitor.records) > 0:
            stdp.out_spike_monitor.records[0] = teacher_img  # teacher-forced postsyn spike map
        dw = stdp.step(on_grad=False)               # compute ΔW and return
        if dw is not None:
            conv.weight += dw
            # apply decay ONLY if something actually changed due to STDP
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()                                  # reset membrane; learner keeps traces

    W0 = conv.weight[0, 0].clone()
    # t0
    step(pre_t0, post_t0)
    # t1
    step(pre_t1, post_t1)
    W1 = conv.weight[0, 0].clone()

    dW = print_delta_matrix(title, W0, W1)
    return W0, W1, dW, conv

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
        # CAUSAL: t0 pre=mini-bar, t1 post=center impulse
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
    end_fracs, end_cols = col_fracs_from_kernel(W_end)

    print("\nFinal 3x3 kernel:\n", W_end.detach().cpu().numpy())
    print("Final column fractions [L,C,R]:", [f"{v:.4f}" for v in end_fracs])

    # Compare FRACTIONS, not absolute column mass (robust to uniform decay)
    frac_gain = end_fracs[1] > (start_fracs[1] + 1e-9)
    print("Center column fraction increased? ", "YES" if frac_gain else "NO")
    _assert_ge(end_fracs[1], 0.38, f"repeat_causal_pairs: center fraction too low ({end_fracs[1]:.4f})")

    return W_start, W_end

# ---------- Two-channel + WTA competition ----------
@torch.no_grad()
def repeat_causal_pairs_two_channels_wta(n_pairs=200, progress_every=40):
    """
    Two output channels compete via WTA. Winner at t0 (from pre activity) also
    receives the teacher credit at t1. We report per-channel column fractions
    and how often each channel won the competition.
    """
    out_ch = 2
    conv = nn.Conv2d(1, out_ch, kernel_size=K, padding=PAD, bias=False).to(device)
    sn   = neuron.IFNode(v_threshold=VTH).to(device)
    stdp = learning.STDPLearner(
        step_mode='s', synapse=conv, sn=sn,
        tau_pre=TAU_PRE, tau_post=TAU_POST,
        f_pre=lambda w: ETA, f_post=lambda w: ETA
    )

    # Small uniform init + tiny noise to break symmetry
    conv.weight.fill_(0.10)
    conv.weight.add_(1e-3 * torch.randn_like(conv.weight))

    pre_bar = make_center_minibar(H, W)
    post_impulse_single = make_center_impulse(H, W)   # [1,1,H,W]
    zeros = torch.zeros_like(pre_bar)

    wins = [0, 0]  # win counts for channel 0 and 1

    def step_t0_with_wta(pre_img):
        """t0: run pre, WTA decides the winner channel based on post activity at center."""
        _ = sn(conv(pre_img))
        # Get the record the learner will consume
        rec = stdp.out_spike_monitor.records[0]  # [B=1, C=2, H, W]
        # Decide winner by comparing the two channels at the center pixel
        cvals = rec[0, :, cy, cx]  # [2]
        win_c = int(torch.argmax(cvals).item())
        wins[win_c] += 1
        # Zero losing channel everywhere (strong WTA)
        lose_c = 1 - win_c
        rec[0, lose_c, :, :] = 0.0
        stdp.out_spike_monitor.records[0] = rec

        # No teacher at t0
        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()
        return win_c

    def step_t1_teacher_for_winner(win_c):
        """t1: give the teacher impulse ONLY to the winner channel."""
        _ = sn(conv(zeros))  # just to push a record
        rec = stdp.out_spike_monitor.records[0]  # [1,2,H,W]
        # Build a 2-channel teacher: only winner channel gets the impulse
        teacher_two = torch.zeros_like(rec)
        teacher_two[0, win_c, :, :] = post_impulse_single[0, 0, :, :]
        stdp.out_spike_monitor.records[0] = teacher_two

        dw = stdp.step(on_grad=False)
        if dw is not None:
            conv.weight += dw
            if dw.abs().sum().item() > 0.0:
                _apply_decay(conv.weight)
        sn.reset()

    # Helpers to compute per-channel column fractions for in_channel=0 (ON)
    def per_channel_col_fracs():
        fracs = []
        col_sums = []
        for c in range(out_ch):
            W3 = conv.weight[c, 0]  # [3,3] for channel c, input 0
            f, cs = col_fracs_from_kernel(W3)
            fracs.append(f)
            col_sums.append(cs)
        return fracs, col_sums

    # Progress header
    f0, _ = per_channel_col_fracs()
    print("Initial col fractions ch0[L,C,R]:", [f"{v:.4f}" for v in f0[0]])
    print("Initial col fractions ch1[L,C,R]:", [f"{v:.4f}" for v in f0[1]])

    for i in range(1, n_pairs + 1):
        # CAUSAL: t0 pre mini-bar (WTA decides winner), t1 teacher impulse to the winner
        winner = step_t0_with_wta(pre_bar)
        step_t1_teacher_for_winner(winner)

        if i % progress_every == 0 or i == 1 or i == n_pairs:
            fracs, cols = per_channel_col_fracs()
            wmin = conv.weight.min().item()
            wmax = conv.weight.max().item()
            print(f"[pair {i:4d}] ch0 col_fracs={['%.4f'%v for v in fracs[0]]}  "
                  f"ch1 col_fracs={['%.4f'%v for v in fracs[1]]}  "
                  f"wins(ch0,ch1)=({wins[0]},{wins[1]})  "
                  f"w[min,max]=[{wmin:.4f},{wmax:.4f}]")

    # Final report
    W0 = conv.weight[0, 0].detach().cpu().numpy()
    W1 = conv.weight[1, 0].detach().cpu().numpy()
    f_end, _ = per_channel_col_fracs()
    print("\nFinal 3x3 kernels (in0->out0, in0->out1):")
    print("W[0,0]:\n", W0)
    print("W[1,0]:\n", W1)
    print("Final col fractions ch0[L,C,R]:", [f"{v:.4f}" for v in f_end[0]])
    print("Final col fractions ch1[L,C,R]:", [f"{v:.4f}" for v in f_end[1]])
    print(f"Win counts → ch0: {wins[0]}, ch1: {wins[1]}")
    # Success: clear separation between channels on center column
    sep = abs(f_end[0][1] - f_end[1][1])
    print("PASS ✅" if sep >= 0.05 else "WARN: weak separation")

# ---------- NEW (E): NULL control (no pre, no post) ----------
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
    # t0: no spikes; t1: no spikes
    step(zeros, zeros)
    step(zeros, zeros)
    W1 = conv.weight[0, 0].clone()
    dW = (W1 - W0).abs().max().item()
    print(f"\nNULL control: max|ΔW| = {dW:.8f} (expect ≈ 0)  → {'PASS ✅' if dW < 1e-8 else 'FAIL ❌'}")
    _assert_true(dW < 1e-8, f"NULL control drifted: max|ΔW|={dW:.8e}")

# ---------- NEW (F): Noise-only pre, no teacher → no bias should accumulate ----------
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
        # t0: random pre impulse, no post
        pre = make_random_impulse(H, W)
        step(pre, zeros)
        # t1: no pre, no post
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

# ---------- NEW (G): Fixed-teacher 3-offset mapping (left/center/right) ----------
@torch.no_grad()
def fixed_teacher_three_offsets(n_pairs=120, progress_every=30, clamp=True):
    """
    Baby-step mapping check:
      Teacher is ALWAYS at image center (cx).
      We run 3 separate blocks:
        - pre mini-bar at (cx-1)  → expect LEFT column gain > others
        - pre mini-bar at (cx)    → expect CENTER column gain > others
        - pre mini-bar at (cx+1)  → expect RIGHT column gain > others
    Each block uses a fresh conv so effects don’t mix.
    """
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
            step(pre_bar, zeros)        # t0
            step(zeros,  post_center)   # t1
            if i % progress_every == 0 or i == 1 or i == n_pairs:
                W_cur = conv.weight[0,0]
                fracs, col_sums = col_fracs_from_kernel(W_cur)
                print(f"[map {label:16s} | pair {i:3d}] col_fracs(L,C,R)={['%.4f'%v for v in fracs]}  "
                      f"(col_sum={['%.6f'%v for v in col_sums]})")

        W1 = conv.weight[0, 0].clone()
        dcol = (W1 - W0).sum(dim=0).detach().cpu().numpy()  # [L,C,R]
        winner = int(dcol.argmax())
        passed = (winner == expect_idx)
        print(f"\nMapping result for {label}: Δcols[L,C,R] = {dcol}  → "
              f"winner idx={winner} ({['LEFT','CENTER','RIGHT'][winner]})  "
              f"{'PASS ✅' if passed else 'FAIL ❌'}\n")
        results.append((label, dcol, passed))

    # Summary
    all_pass = all(p for _, _, p in results)
    print("Fixed-teacher mapping summary:",
          "ALL PASS ✅" if all_pass else "Some FAIL ❌")
    return results

# ---------- Main ----------
if __name__ == "__main__":
    zeros = torch.zeros(1, 1, H, W, device=device)
    pre_bar    = make_center_minibar(H, W)
    post_center= make_center_impulse(H, W)

    # A) Single CAUSAL pair: pre mini-bar at t0, post impulse at t1 → center column ↑
    run_two_step_pair(
        pre_t0=pre_bar,           post_t0=zeros,
        pre_t1=zeros,             post_t1=post_center,
        title="CAUSAL (mini-bar → center impulse)"
    )

    # B) Single ANTI pair: post impulse at t0, pre mini-bar at t1 → center column ↓
    run_two_step_pair(
        pre_t0=zeros,             post_t0=post_center,
        pre_t1=pre_bar,           post_t1=zeros,
        title="ANTI   (center impulse → mini-bar)"
    )

    # C) Repeat CAUSAL pairs (1 channel): center column fraction should rise from ~1/3
    repeat_causal_pairs(n_pairs=200, progress_every=40)

    # D) Repeat CAUSAL pairs with 2 output channels + WTA competition
    repeat_causal_pairs_two_channels_wta(n_pairs=200, progress_every=40)

    # E) NULL control (no spikes anywhere) → no change
    null_pair_should_do_nothing()

    # F) Noise-only pre, no teacher → no bias should accumulate
    noise_pre_no_teacher(n_pairs=200, progress_every=40)

    # G) Fixed-teacher 3-offset mapping (left/center/right)
    fixed_teacher_three_offsets(n_pairs=120, progress_every=30, clamp=True)