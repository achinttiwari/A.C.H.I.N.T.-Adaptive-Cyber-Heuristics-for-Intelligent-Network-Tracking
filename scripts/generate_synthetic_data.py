"""
=============================================================================
generate_synthetic_data.py — Synthetic CICIDS2017-Compatible Dataset
=============================================================================
Purpose:
    Generate a small (~50,000 row) synthetic dataset that mimics the
    structure, column names, and class distribution of CICIDS2017.
    This lets you run and test the FULL pipeline end-to-end in any
    environment without needing to download the 8 GB real dataset.

    The synthetic data is NOT suitable for publication — it is for
    pipeline validation only.  Replace data/raw/ with the real CSVs
    before any academic evaluation.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --rows 10000
    python scripts/generate_synthetic_data.py --rows 100000 --output-dir /tmp/data
=============================================================================
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ---------------------------------------------------------------------------
# CICIDS2017 column names (79 features + label + timestamp)
# Matches the exact CICFlowMeter output schema used by the real dataset.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# Attack types present in CICIDS2017 with approximate class proportions
ATTACK_TYPES = {
    "BENIGN":                   0.80,   # 80% benign — realistic imbalance
    "DDoS":                     0.06,
    "PortScan":                 0.05,
    "FTP-Patator":              0.02,
    "SSH-Patator":              0.02,
    "DoS Hulk":                 0.02,
    "DoS GoldenEye":            0.01,
    "DoS slowloris":            0.01,
    "DoS Slowhttptest":         0.005,
    "Web Attack – Brute Force": 0.003,
    "Web Attack – XSS":         0.002,
    "Bot":                      0.002,
    "Infiltration":             0.001,
    "Heartbleed":               0.001,
}


def _generate_features(n: int, label: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic feature values for n rows of a given attack class.

    Different attack types have distinct statistical signatures:
    - DDoS: very high packet rate, short flow duration, small packets
    - PortScan: many very short flows, low byte count
    - Brute Force: moderate duration, many SYN flags
    - BENIGN: longer flows, variable packet size, low flag counts
    """
    n_features = len(FEATURE_COLUMNS)
    X = np.zeros((n, n_features), dtype=np.float64)

    if label == "BENIGN":
        # Normal traffic: long flows, variable sizes, low flag activity
        X[:, 0]  = rng.exponential(scale=1_000_000, size=n)  # Flow Duration (μs)
        X[:, 1]  = rng.integers(2, 200, size=n)               # Total Fwd Packets
        X[:, 2]  = rng.integers(1, 150, size=n)               # Total Bwd Packets
        X[:, 3]  = rng.integers(100, 50_000, size=n)          # Total Length Fwd
        X[:, 4]  = rng.integers(50,  40_000, size=n)          # Total Length Bwd
        X[:, 13] = rng.uniform(1_000, 100_000, size=n)        # Flow Bytes/s
        X[:, 14] = rng.uniform(1, 500, size=n)                # Flow Packets/s
        X[:, 44] = rng.integers(0, 2, size=n)                 # SYN Flag Count
        X[:, 46] = rng.integers(0, 10, size=n)                # ACK Flag Count

    elif label in ("DDoS", "DoS Hulk", "DoS GoldenEye",
                   "DoS slowloris", "DoS Slowhttptest"):
        # DoS/DDoS: very high packet/byte rates, short or long flows
        X[:, 0]  = rng.exponential(scale=500_000, size=n)
        X[:, 1]  = rng.integers(100, 10_000, size=n)          # High fwd packets
        X[:, 2]  = rng.integers(0, 5, size=n)                 # Near-zero bwd
        X[:, 3]  = rng.integers(10_000, 1_000_000, size=n)
        X[:, 13] = rng.uniform(100_000, 10_000_000, size=n)   # Very high byte rate
        X[:, 14] = rng.uniform(500, 50_000, size=n)           # Very high pkt rate
        X[:, 44] = rng.integers(1, 5, size=n)                 # SYN flags
        X[:, 42] = rng.integers(0, 3, size=n)                 # FIN flags

    elif label == "PortScan":
        # Port scan: many very short single-packet flows
        X[:, 0]  = rng.exponential(scale=1_000, size=n)       # Very short duration
        X[:, 1]  = rng.integers(1, 3, size=n)                 # 1-2 packets
        X[:, 2]  = rng.integers(0, 2, size=n)
        X[:, 3]  = rng.integers(0, 100, size=n)
        X[:, 4]  = rng.integers(0, 100, size=n)
        X[:, 13] = rng.uniform(0, 5_000, size=n)
        X[:, 44] = rng.integers(1, 3, size=n)                 # SYN flags

    elif label in ("FTP-Patator", "SSH-Patator",
                   "Web Attack – Brute Force", "Heartbleed"):
        # Brute force: repeated connections, medium packet counts, SYN bursts
        X[:, 0]  = rng.exponential(scale=200_000, size=n)
        X[:, 1]  = rng.integers(5, 50, size=n)
        X[:, 2]  = rng.integers(3, 40, size=n)
        X[:, 3]  = rng.integers(200, 5_000, size=n)
        X[:, 13] = rng.uniform(500, 20_000, size=n)
        X[:, 44] = rng.integers(1, 10, size=n)                # SYN flags (repeated)

    else:
        # Generic attack: slightly off-distribution from benign
        X[:, 0]  = rng.exponential(scale=300_000, size=n)
        X[:, 1]  = rng.integers(1, 100, size=n)
        X[:, 2]  = rng.integers(0, 80, size=n)
        X[:, 3]  = rng.integers(0, 20_000, size=n)
        X[:, 13] = rng.uniform(100, 50_000, size=n)

    # Fill remaining features with correlated noise
    for i in range(n_features):
        if X[:, i].sum() == 0:
            X[:, i] = np.abs(rng.normal(loc=10, scale=30, size=n))

    # Add realistic noise to all features
    noise = rng.normal(loc=0, scale=0.05, size=X.shape)
    X = np.abs(X * (1 + noise))   # keep non-negative

    return X


def generate(n_rows: int, output_dir: str) -> None:
    rng = np.random.default_rng(GLOBAL_SEED)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Generating %d synthetic CICIDS2017-compatible rows ...", n_rows
    )

    all_frames = []
    start_ts = pd.Timestamp("2017-07-03 08:00:00")  # Monday morning

    for label, proportion in ATTACK_TYPES.items():
        n = max(1, int(n_rows * proportion))
        logger.info("  %-35s %6d rows", label, n)

        X = _generate_features(n, label, rng)
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df["Label"] = label

        # Synthetic timestamps (evenly spaced within the day)
        df["Timestamp"] = pd.date_range(
            start=start_ts, periods=n, freq="100ms"
        ).strftime("%d/%m/%Y %H:%M")
        start_ts += pd.Timedelta(hours=1)

        all_frames.append(df)

    df_full = pd.concat(all_frames, ignore_index=True)

    # Shuffle rows so attacks aren't all grouped together
    df_full = df_full.sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)

    # Save as a single CSV (mimics one CICIDS2017 day file)
    out_path = os.path.join(output_dir, "synthetic_cicids2017.csv")
    df_full.to_csv(out_path, index=False)

    logger.info(
        "Saved %d rows to: %s  (%.1f MB)",
        len(df_full), out_path, os.path.getsize(out_path) / 1e6,
    )
    logger.info(
        "Class distribution:\n%s",
        df_full["Label"].value_counts().to_string(),
    )
    logger.info("Done. Run 'python main.py' to start the pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic CICIDS2017-compatible dataset."
    )
    parser.add_argument(
        "--rows", type=int, default=50_000,
        help="Total number of rows to generate (default: 50000)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        ),
        help="Directory to write the CSV (default: data/raw/)",
    )
    args = parser.parse_args()
    generate(args.rows, args.output_dir)


if __name__ == "__main__":
    # Ensure numpy and pandas are available
    try:
        import numpy, pandas  # noqa: F401
    except ImportError:
        logger.error(
            "numpy and pandas are required. "
            "Run: pip install numpy pandas"
        )
        sys.exit(1)

    main()
