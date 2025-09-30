import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import pandas as pd

# SpikingJelly imports
from spikingjelly.activation_based import neuron, learning

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---- 1. Dataset Class (Final) ----
class NCARSSpikeFrameDataset(Dataset):
    def __init__(self, cached_data, n_time_steps=50, height=100, width=120):
        self.data_cache = cached_data['data']
        self.index = cached_data['index']
        self.n_time_steps = n_time_steps
        self.height = height
        self.width = width
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        file_idx, t_start, t_end, label = self.index[idx]
        ts_full = self.data_cache[file_idx]['ts']
        addr_full = self.data_cache[file_idx]['addr']
        mask = (ts_full >= t_start) & (ts_full < t_end)
        if mask.sum() == 0:
            return torch.zeros(self.n_time_steps, 2, self.height, self.width), torch.tensor(label, dtype=torch.long)
        x = (addr_full[mask] & 0x00003FFF)
        y = (addr_full[mask] & 0x0FFFC000) >> 14
        p = -1 + 2 * ((addr_full[mask] & 0x10000000) >> 28).astype(np.int32)
        t = ts_full[mask]
        time_bins = np.floor((t - t_start) / ((t_end - t_start) / self.n_time_steps)).astype(np.int64)
        time_bins = np.clip(time_bins, 0, self.n_time_steps - 1)
        channels = ((p + 1) // 2).astype(np.int64)
        frames = torch.zeros(self.n_time_steps, 2, self.height, self.width, dtype=torch.float)
        frames[time_bins, channels, y, x] = 1.0
        return frames, torch.tensor(label, dtype=torch.long)

# ---- 2. Final SpikingJelly Model (EXPANDED for Grid Search) ----
class SjSTDPConv(nn.Module):
    def __init__(self, v_threshold=1.0, stdp_eta=0.03, weight_decay=0.001,
                 in_channels=2, out_channels=8, kernel_size=5,
                 tau_pre=30.0, tau_post=30.0, stdp_rule='additive'):
        super().__init__()
        self.weight_decay = weight_decay
        padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding, bias=False)
        self.conv.weight.requires_grad = False
        self.lif = neuron.LIFNode(tau=100.0, v_threshold=v_threshold)

        if stdp_rule == 'additive':
            f_pre = lambda _: stdp_eta
            f_post = lambda _: -stdp_eta
        elif stdp_rule == 'multiplicative':
            f_pre = lambda w: stdp_eta * (1.0 - w)
            f_post = lambda w: -stdp_eta * w
        else:
            raise ValueError(f"Unknown STDP rule: {stdp_rule}")

        self.stdp = learning.STDPLearner(
            step_mode='s', synapse=self.conv, sn=self.lif,
            tau_pre=tau_pre, tau_post=tau_post,
            f_pre=f_pre, f_post=f_post
        )

    @torch.no_grad()
    def forward(self, x_seq):
        spike_train = []
        for t in range(x_seq.shape[0]):
            post_spikes = self.lif(self.conv(x_seq[t]))
            delta_w = self.stdp.step(on_grad=False)
            if delta_w is not None:
                self.conv.weight.data += delta_w
            spike_train.append(post_spikes)
        return torch.stack(spike_train)

    @torch.no_grad()
    def reset_state(self):
        self.lif.reset()
        self.stdp.reset()

# ---- 3. Main Script for Expanded Grid Search ----
if __name__ == '__main__':
    # --- Hyperparameters ---
    n_time_steps = 50
    batch_size = 32
    num_epochs = 10
    torch.backends.cudnn.benchmark = True

    # EXPANDED: A larger, more comprehensive parameter grid
    param_grid = [
        # GROUP 1: Baseline and Varying Threshold/Eta
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.3, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.2, 'stdp_eta': 0.05, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.1, 'stdp_eta': 0.05, 'weight_decay': 0.002, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},

        # GROUP 2: Exploring Kernel Size and Output Channels
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 3, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 7, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 16, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 3, 'out_channels': 16, 'stdp_rule': 'additive', 'tau_pre': 30.0, 'tau_post': 30.0},

        # GROUP 3: Exploring STDP Rule and Tau Asymmetry
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'multiplicative', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.1, 'stdp_eta': 0.05, 'weight_decay': 0.002, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'multiplicative', 'tau_pre': 30.0, 'tau_post': 30.0},
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 15.0, 'tau_post': 60.0},
        {'v_threshold': 0.2, 'stdp_eta': 0.03, 'weight_decay': 0.001, 'kernel_size': 5, 'out_channels': 8, 'stdp_rule': 'additive', 'tau_pre': 60.0, 'tau_post': 15.0},
    ]

    # ... (Setup and Data Loading are the same) ...
    LOG_DIR_BASE = "/workspace/logs"; os.makedirs(LOG_DIR_BASE, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
    logging.info("--- Loading Cached Data ---")
    CACHE_DIR = "/workspace/caches"
    cache_paths = {'train_cars': f"{CACHE_DIR}/ncars_train_cars_win{n_time_steps}ms.pt",
                   'train_background': f"{CACHE_DIR}/ncars_train_background_win{n_time_steps}ms.pt"}
    train_cars_data = torch.load(cache_paths['train_cars'], weights_only=False)
    train_bg_data = torch.load(cache_paths['train_background'], weights_only=False)
    combined_train_data = {'data': train_cars_data['data'] + train_bg_data['data'],
                           'index': train_cars_data['index'] + [(idx + len(train_cars_data['data']), s, e, l) for idx, s, e, l in train_bg_data['index']]}
    train_dataset = NCARSSpikeFrameDataset(combined_train_data, n_time_steps=n_time_steps)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count()//2 or 2)
    results_summary = []

    for params in param_grid:
        run_name = f"vth_{params['v_threshold']}_eta_{params['stdp_eta']}_wd_{params['weight_decay']}_ks_{params['kernel_size']}_ch_{params['out_channels']}_tau_{params['tau_pre']}-{params['tau_post']}_{params['stdp_rule']}"
        run_dir = os.path.join(LOG_DIR_BASE, run_name); os.makedirs(run_dir, exist_ok=True)
        # ... (logging setup is the same) ...
        logger = logging.getLogger(run_name); logger.setLevel(logging.INFO); logger.propagate = False
        for handler in logger.handlers[:]: handler.close(); logger.removeHandler(handler)
        file_handler = logging.FileHandler(os.path.join(run_dir, "training.log"), mode='w'); formatter = logging.Formatter('%(asctime)s %(message)s'); file_handler.setFormatter(formatter); logger.addHandler(file_handler)
        
        logger.info(f"\n{'='*20}\nSTARTING RUN: {run_name}\n{'='*20}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = SjSTDPConv(**params).to(device)
        with torch.no_grad():
            layer.conv.weight.uniform_(0.2, 0.5)

        weight_change_history = []
        firing_rate_history = []
        
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | {run_name}")
            for i, (spike_frames, labels) in enumerate(pbar):
                x_seq = spike_frames.permute(1, 0, 2, 3, 4).to(device)
                with torch.no_grad():
                    weights_before = layer.conv.weight.data.clone()
                    output_spikes = layer(x_seq)
                    layer.conv.weight.mul_(1.0 - layer.weight_decay)
                    layer.conv.weight.clamp_(0.0, 1.0)
                    layer.reset_state()
                    weights_after = layer.conv.weight.data
                    mean_abs_change = torch.mean(torch.abs(weights_after - weights_before)).item()
                    firing_rate = (output_spikes.sum() / output_spikes.numel()).item()
                    weight_change_history.append(mean_abs_change)
                    firing_rate_history.append(firing_rate)
        
        # ... (Summary calculation and saving is the same) ...
        w = layer.conv.weight.data
        summary = {**params, 'avg_firing_rate': float(np.mean(firing_rate_history)), 'final_firing_rate': float(firing_rate_history[-1]),
                   'avg_weight_change': float(np.mean(weight_change_history)), 'final_weight_std': float(w.std().item()),
                   'final_weight_min': float(w.min().item()), 'final_weight_max': float(w.max().item())}
        summary['flag_dead'] = summary['avg_firing_rate'] < 1e-4
        summary['flag_saturated'] = summary['avg_firing_rate'] > 0.2
        results_summary.append(summary)
        torch.save(layer.state_dict(), os.path.join(run_dir, "weights.pth"))
        pd.DataFrame({'iter': np.arange(len(firing_rate_history)), 'firing_rate': firing_rate_history,
                      'mean_abs_dW': weight_change_history}).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
        logger.info(f"--- FINISHED RUN: {run_name} ---")
        for handler in logger.handlers[:]: handler.close(); logger.removeHandler(handler)

    # Print a final sorted summary table
    summary_df = pd.DataFrame(results_summary)
    summary_df['score'] = summary_df['final_weight_std'] * (1 - summary_df['flag_dead']) * (1 - summary_df['flag_saturated'])
    summary_df = summary_df.sort_values(['flag_dead', 'flag_saturated', 'score'], ascending=[True, True, False])
    print("\n\n" + "="*80)
    print("                           GRID SEARCH SUMMARY (best first)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)