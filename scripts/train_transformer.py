#!/usr/bin/env python3
"""
Apollo Time-Series Transformer — Training Script
=================================================

Trains a small self-supervised Transformer on telemetry data collected by
Apollo's TelemetryLogger.  The model learns to predict the next system
state vector given the previous 120 observations (60 seconds at 500ms/cycle).

At inference time (inside the Rust daemon via tract-onnx), high reconstruction
error signals anomalous system behaviour that none of Apollo's existing
univariate models captured.

The script uses a **data maturity system** that adapts training strategy
based on how much diverse, quality data is available:

  - Immature  (< 7 days):  don't train — not enough diversity
  - Juvenile  (7–14 days): train with high regularisation, provisional model
  - Stable    (14–28 days): full training, reliable model
  - Mature    (28+ days):   full training, skip if < 20% new data since last run

Architecture
------------
- 2-layer TransformerEncoder, d_model=64, nhead=4, FFN=128
- ~120K parameters, ~480KB on disk (fp32)
- Input: [batch, seq_len=120, n_features=16]
- Output: [batch, seq_len=120, n_features=16] (shifted by 1)

References
----------
- Vaswani et al. 2017, "Attention Is All You Need"
- Zerveas et al. 2021, "Multivariate Time Series Representation Learning"
- Tuli et al. 2022, "TranAD"
- Bishop 2006, "Pattern Recognition and Machine Learning" — 10x rule
- Ash & Adams 2020, "On Warm-Starting Neural Network Training"
- Srivastava et al. 2014, "Dropout" — regularisation for small datasets

Usage
-----
    # Automatic mode (called by launchd, handles everything):
    python3 scripts/train_transformer.py --auto

    # Manual training (ignores maturity gates):
    python3 scripts/train_transformer.py --data-dir /var/lib/apollo/telemetry

Requirements
------------
    pip install torch numpy
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ── Constants (must match telemetry_logger.rs) ───────────────────────────

MAGIC = 0x41504F4C  # "APOL"
HEADER_SIZE = 32
N_FEATURES = 16
SEQ_LEN = 120

FEATURE_NAMES = [
    "pressure_smooth", "pressure_velocity", "pressure_predicted_5s",
    "swap_velocity_smooth", "pressure_integral", "cusum_score",
    "entropy_anomaly", "p_oom_30s", "monopoly_risk", "urgency",
    "cpu_total", "compressor_ratio", "dominant_share", "latency_score",
    "active_proc_count", "thermal_score",
]

# ── Data maturity thresholds ─────────────────────────────────────────────
# Based on ~144 periodic dumps/day + event dumps.
# Bishop 2006 10x rule: 120K params → need ≥1.2M training samples.
# With sliding window over 240-vector dumps:
#   1000 dumps × 240 vectors = 240K vectors → 240K - 120 = ~240K sequences
#   5000 dumps × 240 vectors = 1.2M vectors → 1.2M sequences (10x rule met)

# Periodic dumps = 6/hour × 24h = 144/day.  Plus event dumps.
MATURITY_IMMATURE_FILES = 400       # ~3 days of active use
MATURITY_JUVENILE_FILES = 1000      # ~7 days of active use — train with caution
MATURITY_STABLE_FILES   = 2500      # ~18 days of active use — reliable training
MATURITY_SKIP_NEW_PCT   = 0.20      # mature: skip if < 20% new data


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_bin_file(path: Path) -> tuple[np.ndarray | None, dict]:
    """Load a single .bin telemetry file.

    Returns (array, metadata) where metadata has timestamp and event_kind.
    """
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        return None, {}

    magic, n_vecs, n_feat = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or n_feat != N_FEATURES:
        return None, {}

    # Read header metadata.
    _reserved = struct.unpack_from("<I", data, 12)[0]
    timestamp = struct.unpack_from("<q", data, 16)[0]
    event_kind = struct.unpack_from("<I", data, 24)[0]

    expected = HEADER_SIZE + n_vecs * N_FEATURES * 4
    if len(data) < expected:
        return None, {}

    arr = np.frombuffer(data, dtype=np.float32, offset=HEADER_SIZE)
    arr = arr[: n_vecs * N_FEATURES].reshape(n_vecs, N_FEATURES)

    meta = {"timestamp": timestamp, "event_kind": event_kind, "n_vecs": n_vecs}
    return arr, meta


def load_dataset(data_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Load all .bin files, return (data_array, metadata_list)."""
    arrays = []
    metas = []
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        print(f"[ERROR] no .bin files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    for f in bin_files:
        arr, meta = load_bin_file(f)
        if arr is not None and len(arr) > 0:
            arrays.append(arr)
            metas.append(meta)

    if not arrays:
        print("[ERROR] no valid data loaded", file=sys.stderr)
        sys.exit(1)

    combined = np.concatenate(arrays, axis=0)
    print(f"Loaded {len(arrays)} files, {len(combined):,} total vectors")
    return combined, metas


def make_sequences(data: np.ndarray, seq_len: int = SEQ_LEN) -> tuple:
    """Slide a window over the data to create (input, target) pairs."""
    n = len(data) - seq_len
    if n <= 0:
        print("[ERROR] not enough data for even one sequence", file=sys.stderr)
        sys.exit(1)

    X = np.zeros((n, seq_len, N_FEATURES), dtype=np.float32)
    Y = np.zeros((n, seq_len, N_FEATURES), dtype=np.float32)

    for i in range(n):
        X[i] = data[i : i + seq_len]
        Y[i] = data[i + 1 : i + seq_len + 1]

    return X, Y


# ═══════════════════════════════════════════════════════════════════════════
# Data quality assessment
# ═══════════════════════════════════════════════════════════════════════════

def assess_data_quality(data: np.ndarray, metas: list[dict]) -> dict:
    """Evaluate data diversity and quality for training readiness.

    Returns a dict with quality metrics and a maturity level.
    """
    n_files = len(metas)

    # 1. Temporal coverage: how many distinct hours have data?
    # We don't require 24h coverage — the user may only use the Mac
    # a few hours per day.  We just need ≥4 distinct hours to have
    # some variety in system state.
    hours_seen = set()
    for m in metas:
        ts = m.get("timestamp", 0)
        if ts > 0:
            hour = (ts // 3600) % 24
            hours_seen.add(hour)
    hours_covered = len(hours_seen)

    # 2. Event diversity: ratio of event-triggered vs periodic dumps.
    n_events = sum(1 for m in metas if m.get("event_kind", 0) > 0)
    n_periodic = n_files - n_events
    event_ratio = n_events / max(n_files, 1)

    # 3. Feature variance: are features diverse or constant?
    # Low variance = system was idle the whole time = poor training data.
    feature_std = data.std(axis=0)
    n_active_features = int(np.sum(feature_std > 1e-4))
    feature_diversity = n_active_features / N_FEATURES  # 0-1

    # 4. Temporal span: how many days of data?
    timestamps = [m.get("timestamp", 0) for m in metas if m.get("timestamp", 0) > 0]
    if len(timestamps) >= 2:
        span_days = (max(timestamps) - min(timestamps)) / 86400.0
    else:
        span_days = 0.0

    # 5. Determine maturity level.
    if n_files < MATURITY_IMMATURE_FILES or span_days < 3:
        maturity = "immature"
    elif n_files < MATURITY_JUVENILE_FILES or span_days < 7:
        maturity = "juvenile"
    elif n_files < MATURITY_STABLE_FILES or span_days < 14:
        maturity = "stable"
    else:
        maturity = "mature"

    quality = {
        "n_files": n_files,
        "n_vectors": len(data),
        "span_days": round(span_days, 1),
        "hours_covered": hours_covered,
        "n_events": n_events,
        "n_periodic": n_periodic,
        "event_ratio": round(event_ratio, 3),
        "active_features": n_active_features,
        "feature_diversity": round(feature_diversity, 2),
        "maturity": maturity,
    }
    return quality


def print_quality_report(q: dict):
    """Print a human-readable data quality report."""
    print(f"\n{'='*60}")
    print(f"  DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Files:            {q['n_files']:,}")
    print(f"  Vectors:          {q['n_vectors']:,}")
    print(f"  Span:             {q['span_days']} days")
    print(f"  Hours covered:    {q['hours_covered']} distinct hours")
    print(f"  Event dumps:      {q['n_events']} ({q['event_ratio']*100:.1f}%)")
    print(f"  Periodic dumps:   {q['n_periodic']}")
    print(f"  Active features:  {q['active_features']}/{N_FEATURES}")
    print(f"  Maturity:         {q['maturity'].upper()}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

def build_model(n_features: int = N_FEATURES, d_model: int = 64,
                nhead: int = 4, n_layers: int = 2, seq_len: int = SEQ_LEN,
                dropout: float = 0.1):
    """Build the Apollo Transformer model."""
    import torch
    from torch import nn

    class ApolloTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(n_features, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN (Xiong et al. 2020)
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, n_features)

        def forward(self, x):
            seq_len_actual = x.shape[1]
            pos_ids = torch.arange(seq_len_actual, device=x.device)
            h = self.embed(x) + self.pos(pos_ids)
            h = self.encoder(h)
            return self.head(h)

    model = ApolloTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    return model


def train(model, X: np.ndarray, Y: np.ndarray, epochs: int = 50,
          batch_size: int = 64, lr: float = 1e-3, val_split: float = 0.2):
    """Train with AdamW + cosine annealing on MPS/CPU."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)

    # Temporal train/val split (no shuffling across time).
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    max_patience = 8  # Early stopping: stop if no improvement for 8 epochs.

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= max(n_val, 1)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"patience={patience}/{max_patience}")

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {max_patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\nBest val_loss: {best_val_loss:.6f}")
    return model, best_val_loss


def export_onnx(model, output_path: Path, seq_len: int = SEQ_LEN):
    """Export trained model to ONNX for tract inference in Rust."""
    import torch

    model.eval()
    model = model.cpu()
    dummy = torch.randn(1, seq_len, N_FEATURES)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(output_path), opset_version=17,
        input_names=["sequence"], output_names=["prediction"],
        dynamic_axes={"sequence": {0: "batch"}, "prediction": {0: "batch"}},
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"Exported ONNX model to {output_path} ({size_kb:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# System checks
# ═══════════════════════════════════════════════════════════════════════════

def check_system_idle() -> bool:
    """Check if the system is idle enough for training."""
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5,
        )
        level = int(result.stdout.strip())
        if level > 1:
            print(f"[AUTO] Memory pressure level {level}, skipping")
            return False
    except Exception:
        pass
    return True


def should_retrain(deploy_dir: Path, n_files_now: int) -> bool:
    """Check if retraining is worthwhile based on data growth since last run."""
    log_path = deploy_dir / "transformer_training.log"
    if not log_path.exists():
        return True  # Never trained before.

    try:
        lines = log_path.read_text().strip().split("\n")
        last = json.loads(lines[-1])
        last_n_files = last.get("n_files", 0)
        last_ts = last.get("timestamp", "")

        # Skip if trained less than 12 hours ago.
        if last_ts:
            last_time = time.mktime(time.strptime(last_ts, "%Y-%m-%dT%H:%M:%S"))
            hours_since = (time.time() - last_time) / 3600
            if hours_since < 12:
                print(f"[AUTO] Last training was {hours_since:.1f}h ago, skipping")
                return False

        # Skip if less than 20% new data (mature model doesn't need constant retraining).
        if last_n_files > 0:
            growth = (n_files_now - last_n_files) / last_n_files
            if growth < MATURITY_SKIP_NEW_PCT:
                print(f"[AUTO] Only {growth*100:.1f}% new data since last training, skipping")
                return False

    except Exception:
        pass  # If we can't parse the log, retrain.

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Apollo Time-Series Transformer")
    parser.add_argument("--data-dir", type=Path, default=Path("/var/lib/apollo/telemetry"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("/var/lib/apollo"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--auto", action="store_true",
                        help="Automatic mode: maturity gates, idle check, warm-start")
    args = parser.parse_args()

    model_deploy_path = args.deploy_dir / "apollo_transformer.onnx"
    stats_deploy_path = args.deploy_dir / "feature_stats.json"
    checkpoint_path = args.deploy_dir / "apollo_transformer_checkpoint.pt"

    # ── Load data + assess quality ───────────────────────────────────────
    if not args.data_dir.exists() or not list(args.data_dir.glob("*.bin")):
        print("[AUTO] No telemetry data yet, exiting")
        sys.exit(0)

    data, metas = load_dataset(args.data_dir)
    quality = assess_data_quality(data, metas)
    print_quality_report(quality)

    # ── Auto mode gates ──────────────────────────────────────────────────
    if args.auto:
        # Gate 1: data maturity.
        if quality["maturity"] == "immature":
            print(f"[AUTO] Data immature ({quality['span_days']} days, "
                  f"{quality['n_files']} files). Need ≥3 days + ≥{MATURITY_IMMATURE_FILES} files.")
            print(f"[AUTO] Keep collecting data. Will train automatically when ready.")
            sys.exit(0)

        # Gate 2: feature diversity (data can't be all zeros).
        if quality["feature_diversity"] < 0.5:
            print(f"[AUTO] Low feature diversity ({quality['active_features']}/{N_FEATURES}). "
                  f"Data may be too uniform. Waiting for more varied workloads.")
            sys.exit(0)

        # Gate 3: minimal temporal variety (at least 4 distinct hours seen).
        if quality["hours_covered"] < 4:
            print(f"[AUTO] Only {quality['hours_covered']} distinct hours seen. "
                  f"Need at least 4 for minimal variety.")
            sys.exit(0)

        # Gate 4: system idle.
        if not check_system_idle():
            sys.exit(0)

        # Gate 5: enough new data since last training?
        if not should_retrain(args.deploy_dir, quality["n_files"]):
            sys.exit(0)

        print(f"[AUTO] Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Adapt training params based on maturity ──────────────────────────
    maturity = quality["maturity"]

    if maturity == "juvenile":
        # Less data → more regularisation, more epochs per sample.
        dropout = 0.2      # Srivastava et al. 2014: higher dropout = more regularisation
        epochs = 60        # More passes over limited data
        lr = args.lr * 0.5 # Slower learning to avoid overfitting
        print(f"[JUVENILE] dropout=0.2, epochs={epochs}, lr={lr:.1e}")
    elif maturity == "stable":
        dropout = 0.1
        epochs = args.epochs
        lr = args.lr
        print(f"[STABLE] dropout=0.1, epochs={epochs}, lr={lr:.1e}")
    else:  # mature
        dropout = 0.1
        epochs = args.epochs
        lr = args.lr
        print(f"[MATURE] dropout=0.1, epochs={epochs}, lr={lr:.1e}")

    # ── Normalise ────────────────────────────────────────────────────────
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0
    data_norm = (data - mean) / std

    X, Y = make_sequences(data_norm, seq_len=args.seq_len)
    print(f"Sequences: {len(X):,} samples of length {args.seq_len}")

    # ── Build model + warm-start ─────────────────────────────────────────
    import torch

    model = build_model(seq_len=args.seq_len, dropout=dropout)

    warm_started = False
    if checkpoint_path.exists():
        try:
            state = torch.load(str(checkpoint_path), map_location="cpu",
                               weights_only=True)
            model.load_state_dict(state)
            warm_started = True
            print(f"Warm-start: loaded weights from {checkpoint_path}")
            # Ash & Adams 2020: warm-start converges faster.
            epochs = min(epochs, 15)
            lr = lr * 0.3
            print(f"  → epochs={epochs}, lr={lr:.1e}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}, training from scratch")

    # ── Train ────────────────────────────────────────────────────────────
    model, best_val_loss = train(model, X, Y, epochs=epochs,
                                 batch_size=args.batch_size, lr=lr)

    # ── Save checkpoint ──────────────────────────────────────────────────
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(checkpoint_path))
    print(f"Saved checkpoint to {checkpoint_path}")

    # ── Export ONNX ──────────────────────────────────────────────────────
    export_onnx(model, model_deploy_path, seq_len=args.seq_len)

    # ── Save feature stats ───────────────────────────────────────────────
    stats_deploy_path.write_text(json.dumps({
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": FEATURE_NAMES,
    }, indent=2))
    print(f"Saved feature stats to {stats_deploy_path}")

    # ── Training log ─────────────────────────────────────────────────────
    log_path = args.deploy_dir / "transformer_training.log"
    with open(str(log_path), "a") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_files": quality["n_files"],
            "n_vectors": quality["n_vectors"],
            "n_sequences": len(X),
            "maturity": maturity,
            "warm_start": warm_started,
            "epochs_used": epochs,
            "best_val_loss": round(best_val_loss, 8),
            "span_days": quality["span_days"],
            "hours_covered": quality["hours_covered"],
            "event_ratio": quality["event_ratio"],
        }) + "\n")

    print(f"\nModel deployed to {model_deploy_path}")
    print(f"Daemon will hot-reload automatically")


if __name__ == "__main__":
    main()
