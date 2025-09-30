# main.py
import os
import torch

# --- Local imports ---
from utils import device, H, W
from stimuli import make_center_minibar, make_center_impulse
from runners import (
    run_two_step_pair,
    repeat_causal_pairs,
    repeat_causal_pairs_two_channels_wta,
    null_pair_should_do_nothing,
    noise_pre_no_teacher,
    fixed_teacher_three_offsets,
)

# NCARS helpers
from utils import (
    decode_addr_to_xy_p,
    infer_sensor_size_from_cache,
    window_to_pol_hist,
)

# =========================
#       RUNTIME FLAGS
# =========================
RUN_STDP_A_CAUSAL              = False  # A) single causal pair
RUN_STDP_B_ANTI                = False  # B) single anti-causal pair
RUN_STDP_C_REPEAT_1CH          = False  # C) repeat causal pairs (1ch)
RUN_STDP_D_WTA_2CH             = False  # D) WTA two-channel
RUN_STDP_E_NULL                = False  # E) null control
RUN_STDP_F_NOISE               = False  # F) noise-only pre
RUN_STDP_G_MAPPING             = False  # G) fixed-teacher mapping
RUN_STDP_H_DT_SWEEP            = False  # H) Δt sweep (print)
RUN_STDP_H_DT_SWEEP_NUMERIC    = False  # H2) Δt sweep numeric summary
RUN_STDP_WTA_DW_HEATMAP        = False  # heatmap PNG (2ch WTA)

RUN_NCARS_NUMERIC_DEMO         = True   # Minimal NCARS sanity check (no plotting)

# =========================
#         MAIN
# =========================
if __name__ == "__main__":
    # Common small stimuli used by STDP demos
    zeros       = torch.zeros(1, 1, H, W, device=device)
    pre_bar     = make_center_minibar(H, W)
    post_center = make_center_impulse(H, W)

    # ---------- STDP demos (enable with flags above) ----------
    if RUN_STDP_A_CAUSAL:
        run_two_step_pair(
            pre_t0=pre_bar,           post_t0=zeros,
            pre_t1=zeros,             post_t1=post_center,
            title="CAUSAL (mini-bar → center impulse)"
        )

    if RUN_STDP_B_ANTI:
        run_two_step_pair(
            pre_t0=zeros,             post_t0=post_center,
            pre_t1=pre_bar,           post_t1=zeros,
            title="ANTI   (center impulse → mini-bar)"
        )

    if RUN_STDP_C_REPEAT_1CH:
        repeat_causal_pairs(n_pairs=200, progress_every=40)

    if RUN_STDP_D_WTA_2CH:
        repeat_causal_pairs_two_channels_wta(n_pairs=200, progress_every=40)

    if RUN_STDP_E_NULL:
        null_pair_should_do_nothing()

    if RUN_STDP_F_NOISE:
        noise_pre_no_teacher(n_pairs=200, progress_every=40)

    if RUN_STDP_G_MAPPING:
        fixed_teacher_three_offsets(n_pairs=120, progress_every=30, clamp=True)

    if RUN_STDP_H_DT_SWEEP:
        from runners import stdp_dt_sweep
        stdp_dt_sweep(max_dt=5)

    if RUN_STDP_H_DT_SWEEP_NUMERIC:
        from runners import stdp_dt_sweep_numeric
        stdp_dt_sweep_numeric(max_dt=5)

    if RUN_STDP_WTA_DW_HEATMAP:
        from runners import two_channel_wta_dW_heatmap_png
        two_channel_wta_dW_heatmap_png(n_pairs=200)

    # ---------- NCARS tiny numeric sanity check (no plots) ----------
    if RUN_NCARS_NUMERIC_DEMO:
        cache_demo = "caches/ncars_train_cars_win50ms.pt"  # choose any of the four caches you built
        if os.path.exists(cache_demo):
            print("\n[NCARS] Loading cache:", cache_demo)
            raw = torch.load(cache_demo, weights_only=False)

            # Infer sensor size from cached addresses
            W_ncars, H_ncars = infer_sensor_size_from_cache(raw)
            print(f"[NCARS] inferred sensor size: W={W_ncars}, H={H_ncars}")

            # Pick a sample window and build a polarity histogram (H×W×2)
            sample_idx = 0
            hist = window_to_pol_hist(raw, sample_idx, W_ncars, H_ncars)
            total_neg = int(hist[..., 0].sum())
            total_pos = int(hist[..., 1].sum())
            total_all = total_neg + total_pos

            file_idx, t_start, t_end, lbl = raw["index"][sample_idx]
            print(f"[NCARS] sample window idx={sample_idx} → file={file_idx}, t=[{t_start},{t_end}), label={lbl}")
            print(f"[NCARS] events in window: total={total_all}, neg={total_neg}, pos={total_pos}")

            # Show a few “hottest” pixels numerically (top-5 by total count)
            flat_counts = (hist[..., 0] + hist[..., 1]).reshape(-1)
            if flat_counts.size > 0:
                topk = min(5, flat_counts.size)
                top_idx = flat_counts.argsort()[-topk:][::-1]
                print("[NCARS] top-5 pixels by count (y, x, total, neg, pos):")
                for idx in top_idx:
                    y = int(idx // W_ncars)
                    x = int(idx % W_ncars)
                    c  = int(hist[y, x].sum())
                    cn = int(hist[y, x, 0])
                    cp = int(hist[y, x, 1])
                    print(f"  (y={y:3d}, x={x:3d}) → total={c:4d}, neg={cn:4d}, pos={cp:4d}")
        else:
            print("\n[NCARS] Skipping numeric demo—cache not found:", cache_demo)