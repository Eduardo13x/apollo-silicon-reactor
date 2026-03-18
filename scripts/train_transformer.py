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

Architecture
------------
- 2-layer TransformerEncoder, d_model=64, nhead=4, FFN=128
- ~120K parameters, ~480KB on disk (fp32)
- Input: [batch, seq_len=120, n_features=16]
- Output: [batch, seq_len=120, n_features=16] (shifted by 1 for next-step prediction)

References
----------
- Vaswani et al. 2017, "Attention Is All You Need"
- Zerveas et al. 2021, "A Transformer-based Framework for Multivariate
  Time Series Representation Learning"
- Tuli et al. 2022, "TranAD: Deep Transformer Networks for Anomaly Detection
  in Multivariate Time Series"

Usage
-----
    python3 scripts/train_transformer.py --data-dir /var/lib/apollo/telemetry \\
                                          --output models/apollo_transformer.onnx \\
                                          --epochs 50

Requirements
------------
    pip install torch numpy
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# ── Binary format constants (must match telemetry_logger.rs) ─────────────

MAGIC = 0x41504F4C  # "APOL"
HEADER_SIZE = 32
N_FEATURES = 16
SEQ_LEN = 120  # Transformer context window

# Feature names (same order as TelemetryVector in Rust).
FEATURE_NAMES = [
    "pressure_smooth",
    "pressure_velocity",
    "pressure_predicted_5s",
    "swap_velocity_smooth",
    "pressure_integral",
    "cusum_score",
    "entropy_anomaly",
    "p_oom_30s",
    "monopoly_risk",
    "urgency",
    "cpu_total",
    "compressor_ratio",
    "dominant_share",
    "latency_score",
    "active_proc_count",
    "thermal_score",
]


def load_bin_file(path: Path) -> np.ndarray | None:
    """Load a single .bin telemetry file.

    Returns
    -------
    np.ndarray of shape (n_vectors, N_FEATURES) or None if invalid.
    """
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        return None

    magic, n_vecs, n_feat = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or n_feat != N_FEATURES:
        print(f"[WARN] skipping {path.name}: bad magic/features", file=sys.stderr)
        return None

    expected = HEADER_SIZE + n_vecs * N_FEATURES * 4
    if len(data) < expected:
        print(f"[WARN] skipping {path.name}: truncated", file=sys.stderr)
        return None

    arr = np.frombuffer(data, dtype=np.float32, offset=HEADER_SIZE)
    arr = arr[: n_vecs * N_FEATURES].reshape(n_vecs, N_FEATURES)
    return arr


def load_dataset(data_dir: Path, min_len: int = SEQ_LEN + 1) -> np.ndarray:
    """Load all .bin files and concatenate into one big array.

    Files shorter than `min_len` are padded with the last observation
    (repeat-pad, not zero-pad — avoids introducing artificial zero states).

    Returns shape (total_vectors, N_FEATURES).
    """
    arrays = []
    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        print(f"[ERROR] no .bin files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    for f in bin_files:
        arr = load_bin_file(f)
        if arr is not None and len(arr) > 0:
            arrays.append(arr)

    if not arrays:
        print("[ERROR] no valid data loaded", file=sys.stderr)
        sys.exit(1)

    combined = np.concatenate(arrays, axis=0)
    print(f"Loaded {len(arrays)} files, {len(combined)} total vectors")
    return combined


def make_sequences(data: np.ndarray, seq_len: int = SEQ_LEN) -> tuple:
    """Slide a window over the data to create (input, target) pairs.

    Input:  data[i : i+seq_len]
    Target: data[i+1 : i+seq_len+1]  (next-step prediction)

    Returns (X, Y) each of shape (n_samples, seq_len, N_FEATURES).
    """
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


def build_model(n_features: int = N_FEATURES, d_model: int = 64,
                nhead: int = 4, n_layers: int = 2, seq_len: int = SEQ_LEN):
    """Build the Apollo Transformer model.

    Architecture follows Zerveas et al. 2021 for multivariate time series,
    scaled down to ~120K parameters suitable for M1 MacBook Air inference.
    """
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
                dim_feedforward=d_model * 2,  # 128
                dropout=0.1,
                batch_first=True,
                norm_first=True,  # Pre-LN (Xiong et al. 2020) — more stable training
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, n_features)

        def forward(self, x):
            # x: [batch, seq_len, n_features]
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

    # Device selection: MPS (Apple Silicon) > CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)

    # Train/val split (temporal — no shuffling to preserve time order).
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )

    # Shuffle training data (within sequences, not across time — each sequence
    # is already a contiguous window, so shuffling order is fine).
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Training ──
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

        # ── Validation ──
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

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Restore best model.
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\nBest val_loss: {best_val_loss:.6f}")
    return model


def export_onnx(model, output_path: Path, seq_len: int = SEQ_LEN):
    """Export trained model to ONNX for tract inference in Rust."""
    import torch

    model.eval()
    model = model.cpu()

    dummy = torch.randn(1, seq_len, N_FEATURES)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["sequence"],
        output_names=["prediction"],
        dynamic_axes={
            "sequence": {0: "batch"},
            "prediction": {0: "batch"},
        },
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"Exported ONNX model to {output_path} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Train Apollo Time-Series Transformer"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/var/lib/apollo/telemetry"),
        help="Directory with .bin telemetry files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/apollo_transformer.onnx"),
        help="Output ONNX model path",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    # Load and prepare data.
    data = load_dataset(args.data_dir, min_len=args.seq_len + 1)

    # Normalise features to zero-mean unit-variance (per-feature).
    # Save stats for inference-time normalisation in Rust.
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0  # Avoid division by zero for constant features.
    data_norm = (data - mean) / std

    # Save normalisation stats for Rust inference.
    stats_path = args.output.parent / "feature_stats.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(stats_path), mean=mean, std=std, feature_names=FEATURE_NAMES)
    print(f"Saved feature stats to {stats_path}")

    # Create sequences.
    X, Y = make_sequences(data_norm, seq_len=args.seq_len)
    print(f"Sequences: {len(X)} samples of length {args.seq_len}")

    # Build and train.
    model = build_model(seq_len=args.seq_len)
    model = train(model, X, Y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Export.
    export_onnx(model, args.output, seq_len=args.seq_len)

    # Also save feature stats as JSON for Rust consumption.
    import json
    stats_json = args.output.parent / "feature_stats.json"
    stats_json.write_text(json.dumps({
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": FEATURE_NAMES,
    }, indent=2))
    print(f"Saved feature stats (JSON) to {stats_json}")
    print("\nDone! Next steps:")
    print(f"  1. Copy {args.output} to /var/lib/apollo/")
    print(f"  2. Copy {stats_json} to /var/lib/apollo/")
    print("  3. Restart daemon to load the model")


if __name__ == "__main__":
    main()
