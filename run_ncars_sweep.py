# run_ncars_sweep.py
import os, sys, itertools, subprocess, time, json
from pathlib import Path

BASE = Path("/workspace")
SCRIPT = BASE / "ncars_linear_baseline.py"
OUT_DIR = BASE / "results"

# --- fixed dataset geometry / sampling ---
W, H = 120, 100
PER_CLASS_TRAIN = 999999
PER_CLASS_TEST  = 999999
EPOCHS = 20

# --- grids to sweep ---
MODELS      = ["linear", "cnn"]                # your two heads
TIME_BINS   = [1, 5]                           # hist vs voxel(5)
LRS         = [1e-2, 3e-3]                     # two learning rates
WDS         = [0.0, 1e-4]                      # regularization
BATCHES     = [64, 128]                        # batch sizes
SEEDS       = [0, 1, 2]                        # three seeds

# You can narrow/expand any of the above lists as you like.

def build_run_name(model, T, lr, wd, bs, seed):
    tb = "hist" if T == 1 else f"voxT{T}"
    return f"{model}_{tb}_lr{lr:g}_wd{wd:g}_b{bs}_s{seed}"

def metrics_json_path_from_log(log_path: Path) -> Path:
    return log_path.with_name(log_path.name.replace(".log", "_metrics.json"))

def expected_log_path(run_name: str) -> Path:
    # matches ncars_linear_baseline.py naming: {timestamp}_{run_name}.log
    # We can't predict timestamp, so we scan OUT_DIR for the newest that contains run_name.
    # If none exists yet, we’ll return a placeholder path (for display only).
    candidates = sorted(OUT_DIR.glob(f"*_{run_name}.log"))
    return candidates[-1] if candidates else OUT_DIR / f"??????_{run_name}.log"

def main():
    combos = list(itertools.product(MODELS, TIME_BINS, LRS, WDS, BATCHES, SEEDS))
    print(f"[SWEEP] total planned runs: {len(combos)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Nice to the GPU memory allocator
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for i, (model, T, lr, wd, bs, seed) in enumerate(combos, start=1):
        run_name = build_run_name(model, T, lr, wd, bs, seed)
        print(f"\n[{i}/{len(combos)}] RUN {run_name}")

        # If a matching metrics file already exists, skip
        # (We check for *any* file ending with _{run_name}_metrics.json)
        existing_metrics = sorted(OUT_DIR.glob(f"*_{run_name}_metrics.json"))
        if existing_metrics:
            print(f"  -> found metrics: {existing_metrics[-1].name}  (skip)")
            continue

        # build command
        cmd = [
            sys.executable, str(SCRIPT),
            "--W", str(W), "--H", str(H),
            "--per_class_train", str(PER_CLASS_TRAIN),
            "--per_class_test",  str(PER_CLASS_TEST),
            "--epochs",          str(EPOCHS),
            "--batch_size",      str(bs),
            "--lr",              str(lr),
            "--weight_decay",    str(wd),
            "--seed",            str(seed),
            "--time_bins",       str(T),
            "--model",           model,
            "--out_dir",         str(OUT_DIR),
            "--run_name",        run_name,
            # DataLoader workers: tune if you want
            "--num_workers",     "8",
        ]

        print("  ->", " ".join(cmd))
        t0 = time.time()
        try:
            # Stream output live; ncars_linear_baseline.py already tees to its own log file
            proc = subprocess.run(cmd, env=env)
            dt = time.time() - t0
            print(f"  -> finished in {dt/60:.1f} min with returncode={proc.returncode}")

            # Quick check for metrics json
            # We don’t know the exact timestamped filename beforehand, so just search it
            metrics_files = sorted(OUT_DIR.glob(f"*_{run_name}_metrics.json"))
            if metrics_files:
                with open(metrics_files[-1], "r") as f:
                    meta = json.load(f)
                print(f"  -> test_acc: {meta.get('metrics',{}).get('test_acc')}")
            else:
                print("  -> WARNING: metrics json not found (run may have failed before saving).")
        except KeyboardInterrupt:
            print("\n[SWEEP] Interrupted by user.")
            break
        except Exception as e:
            print(f"  -> ERROR during run: {e}")

    print("\n[SWEEP] All planned runs attempted.")
    # Print summary.tsv tail for convenience
    summary = OUT_DIR / "summary.tsv"
    if summary.exists():
        print("\n[SWEEP] Summary tail:")
        try:
            lines = summary.read_text().strip().splitlines()
            for line in lines[-10:]:
                print(line)
        except Exception:
            pass
    else:
        print("[SWEEP] No summary.tsv found (did runs fail early?).")

if __name__ == "__main__":
    main()