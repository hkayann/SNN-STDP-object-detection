# stdp_reservoir.py
import torch
from pathlib import Path

import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import set_global_seed, infer_sensor_size_from_cache, window_to_voxel_sequence, log_run
WEIGHTS_PATH = Path("results/stdp_full_run_8.pt")


@torch.no_grad()
def compute_reservoir_states(pooled: torch.Tensor, w_in: torch.Tensor, w_rec: torch.Tensor, leak: float, input_gain: float) -> torch.Tensor:
    """Leaky tanh reservoir update."""
    hidden_size = w_rec.size(0)
    state = torch.zeros(pooled.size(0), hidden_size, device=pooled.device, dtype=pooled.dtype)
    states = []
    for t in range(pooled.size(1)):
        drive = input_gain * (pooled[:, t] @ w_in) + state @ w_rec.T
        state = (1.0 - leak) * state + leak * torch.tanh(drive)
        states.append(state.unsqueeze(1))
    return torch.cat(states, dim=1)

def extract_pooled_features(raw_pos, raw_neg, num_per_class, W, H, time_bins, conv, device, batch_size=32):
    def sample_indices(raw):
        total = len(raw['index'])
        if num_per_class <= 0 or num_per_class >= total:
            return np.arange(total)
        return np.random.choice(total, size=num_per_class, replace=False)

    def load_and_pool(raw, indices, label):
        pooled_list, label_list = [], []
        Gh, Gw = 4, 5
        feature_dim = conv.out_channels * Gh * Gw
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            if hasattr(batch_idx, "tolist"):
                batch_idx = batch_idx.tolist()
            if len(batch_idx) == 0:
                continue
            seqs = [window_to_voxel_sequence(raw, int(idx), W, H, time_bins).unsqueeze(0)
                    for idx in batch_idx]
            stacked = torch.cat(seqs, dim=0).to(device=device, dtype=torch.float32)  # (B,T,2,H,W)
            B, T = stacked.size(0), stacked.size(1)
            flat = stacked.view(B * T, 2, H, W)
            with torch.no_grad():
                conv_flat = conv(flat)  # (B*T,F,H,W)
                pooled_flat = F.adaptive_avg_pool2d(conv_flat, (Gh, Gw))  # (B*T,F,Gh,Gw)
            pooled = pooled_flat.view(B, T, conv_flat.size(1) * Gh * Gw)
            pooled_list.append(pooled)
            label_list.extend([label] * pooled.size(0))
        if not pooled_list:
            empty_feats = torch.empty(0, time_bins, feature_dim, dtype=torch.float32, device=device)
            empty_labels = torch.empty(0, dtype=torch.long, device=device)
            return empty_feats, empty_labels
        return torch.cat(pooled_list, dim=0), torch.tensor(label_list, dtype=torch.long, device=device)

    pos_idx = sample_indices(raw_pos)
    neg_idx = sample_indices(raw_neg)

    pos_feats, pos_labels = load_and_pool(raw_pos, pos_idx, 1)
    neg_feats, neg_labels = load_and_pool(raw_neg, neg_idx, 0)

    if pos_feats.numel() == 0 and neg_feats.numel() == 0:
        return torch.empty(0, time_bins, conv.out_channels, dtype=torch.float32, device=device),                torch.empty(0, dtype=torch.long, device=device)

    X = torch.cat([pos_feats, neg_feats], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    perm = torch.randperm(X.size(0), device=device)
    return X[perm], y[perm]



def main():
    parser = argparse.ArgumentParser(description="STDP features + simple reservoir readout")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH))
    parser.add_argument("--train_pos", default="caches/ncars_train_cars_win50ms.pt")
    parser.add_argument("--train_neg", default="caches/ncars_train_background_win50ms.pt")
    parser.add_argument("--test_pos", default="caches/ncars_test_cars_win50ms.pt")
    parser.add_argument("--test_neg", default="caches/ncars_test_background_win50ms.pt")
    parser.add_argument("--per_class_train", type=int, default=-1)
    parser.add_argument("--per_class_test", type=int, default=-1)
    parser.add_argument("--time_bins", type=int, default=5)
    parser.add_argument("--feature_batch", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--leak", type=float, default=0.3, help="Reservoir leak rate (alpha)")
    parser.add_argument("--input_gain", type=float, default=1.0, help="Scaling applied to reservoir inputs")
    parser.add_argument("--rho", type=float, default=0.9, help="Target spectral radius for recurrent weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device name: {torch.cuda.get_device_name(device)}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[WARN] weights file not found: {weights_path}")
        return

    weights = torch.load(weights_path, map_location="cpu")
    print(f"[INFO] loaded conv weights with shape {tuple(weights.shape)}")

    # Build pooled feature sequences for a small batch
    train_pos = torch.load(args.train_pos, weights_only=False)
    train_neg = torch.load(args.train_neg, weights_only=False)
    W, H = infer_sensor_size_from_cache(train_pos)

    conv = torch.nn.Conv2d(2, weights.shape[0], kernel_size=3, padding=1, bias=False).to(device)
    with torch.no_grad():
        conv.weight.copy_(weights.to(device))
    conv.eval()

    feature_batch = args.feature_batch
    Xtr, ytr = extract_pooled_features(train_pos, train_neg, num_per_class=args.per_class_train, W=W, H=H, time_bins=args.time_bins, conv=conv, device=device, batch_size=feature_batch)
    print(f"[INFO] train pooled batch shape: {tuple(Xtr.shape)} labels shape: {tuple(ytr.shape)}")

    test_pos = torch.load(args.test_pos, weights_only=False)
    test_neg = torch.load(args.test_neg, weights_only=False)
    Xte, yte = extract_pooled_features(test_pos, test_neg, num_per_class=args.per_class_test, W=W, H=H, time_bins=args.time_bins, conv=conv, device=device, batch_size=feature_batch)
    print(f"[INFO] test pooled batch shape: {tuple(Xte.shape)} labels shape: {tuple(yte.shape)}")

    hidden_size = args.hidden_size
    with torch.no_grad():
        w_in = torch.randn(Xtr.size(-1), hidden_size, device=device)
        w_rec = torch.randn(hidden_size, hidden_size, device=device)
        eigvals = torch.linalg.eigvals(w_rec).abs()
        spectral_radius = eigvals.max().real
        if spectral_radius > 0:
            w_rec = w_rec / spectral_radius * args.rho

    train_states = compute_reservoir_states(Xtr, w_in, w_rec, leak=args.leak, input_gain=args.input_gain)
    test_states = compute_reservoir_states(Xte, w_in, w_rec, leak=args.leak, input_gain=args.input_gain)
    print(f"[INFO] reservoir states (train) shape: {tuple(train_states.shape)}")

    train_feats = train_states[:, -1, :]
    test_feats = test_states[:, -1, :]

    train_dataset = TensorDataset(train_feats, ytr)
    test_dataset = TensorDataset(test_feats, yte)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    model = nn.Linear(hidden_size, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for feats_batch, labels_batch in train_loader:
            feats_batch = feats_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            logits = model(feats_batch)
            loss = loss_fn(logits, labels_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * labels_batch.size(0)
            correct += (logits.argmax(dim=1) == labels_batch).sum().item()
            total += labels_batch.size(0)
        if total > 0:
            train_loss = epoch_loss / total
            train_acc = correct / total
        else:
            train_loss = float('nan')
            train_acc = float('nan')
        if ep % 5 == 0 or ep == 1:
            print(f"[TRAIN] epoch {ep:02d}: loss={train_loss:.4f} acc={train_acc:.3f}")

    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for feats_batch, labels_batch in test_loader:
            feats_batch = feats_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            logits = model(feats_batch)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels_batch).sum().item()
            total_samples += labels_batch.size(0)
    test_acc = total_correct / max(total_samples, 1)
    print(f"[TEST] accuracy={test_acc:.3f}")

    run_name = f"reservoir_{args.hidden_size}h_lr{args.lr}_leak{args.leak}"
    meta = {
        "weights": str(weights_path),
        "hidden_size": args.hidden_size,
        "leak": args.leak,
        "input_gain": args.input_gain,
        "rho": args.rho,
        "per_class_train": args.per_class_train,
        "per_class_test": args.per_class_test,
        "time_bins": args.time_bins,
        "feature_batch": args.feature_batch,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
    }
    summary_fields = [
        "timestamp", "run_name", "weights", "hidden_size", "leak", "input_gain", "rho",
        "per_class_train", "per_class_test", "time_bins", "feature_batch", "batch_size",
        "epochs", "lr", "seed", "device", "test_acc",
    ]
    log_path, summary_path = log_run(Path('results'), run_name, meta, {"test_acc": test_acc}, summary_fields)
    print(f"[SAVE] metrics -> {log_path}")
    print(f"[APPEND] {summary_path}")
if __name__ == "__main__":
    main() 