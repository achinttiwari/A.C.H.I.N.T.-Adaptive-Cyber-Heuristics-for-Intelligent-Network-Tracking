"""
=============================================================================
download_cicids2017.py — Dataset Acquisition Helper
=============================================================================
Purpose:
    Automate the download of the CICIDS2017 dataset from the University of
    New Brunswick's public repository.  Run this script once before training.

    The dataset is hosted as a Google Drive share; this script uses the
    'gdown' library to handle the Drive redirect.

Usage:
    python scripts/download_cicids2017.py

    # Or skip download if you already have the CSVs:
    python scripts/download_cicids2017.py --skip-download

    # Download to a custom directory:
    python scripts/download_cicids2017.py --output-dir /mnt/data/cicids2017

Dataset citation:
    Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A.A. (2018).
    "Toward Generating a New Intrusion Detection Dataset and Intrusion
    Traffic Characterization."
    In Proceedings of the 4th International Conference on Information
    Systems Security and Privacy (ICISSP 2018), pp. 108-116.
    https://doi.org/10.5220/0006639801080116

Official dataset page:
    https://www.unb.ca/cic/datasets/ids-2017.html

Note on UNSW-NB15 alternative:
    If you prefer the UNSW-NB15 dataset (Moustafa & Slay, 2015), download
    from https://research.unsb.edu.au/data/UNSW-NB15/ and set DATA_DIR
    to its CSV directory.  The column names differ; update RAW_LABEL_COL
    in config.py to "label" and adjust COLS_TO_DROP accordingly.
=============================================================================
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File manifest
# CICIDS2017 is distributed as 8 CSV files (one per capture day / session).
# IDs below are the Google Drive file IDs from the UNB public share.
# ---------------------------------------------------------------------------

CICIDS2017_FILES = [
    # (filename, google_drive_file_id)
    # Monday — Benign only
    ("Monday-WorkingHours.pcap_ISCX.csv",
     "1_FNQqgcGdyAvBHJI30QEL0DxlHpHFHTT"),
    # Tuesday — FTP + SSH Brute Force
    ("Tuesday-WorkingHours.pcap_ISCX.csv",
     "10qAi_gMhh9bRkVHDXfB5hJ1OjBiKyFvb"),
    # Wednesday — DoS / DDoS
    ("Wednesday-workingHours.pcap_ISCX.csv",
     "1qAtA5nnLn0yYzlRkNz4U2mPkpIRMzpwZ"),
    # Thursday — Web Attacks (morning) + Infiltration (afternoon)
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
     "1XQXMFV9pY8W-YGzCqbU5xkHJdIeGN00G"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
     "1kXf01IG3oj3pnmkJE8lSf0mLJTUf7kT1"),
    # Friday — DDoS (morning) + PortScan + Botnet (afternoon)
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv",
     "1LVVyDQjdRcFKDMehlBrPfM49ixNlHbwO"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
     "1eUqDMoTNtLiAHByNQcTGHvipEnP8KZbr"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
     "1b0r-hYIDW-s1JajODy25SfPi1BvWVuUH"),
]


def _ensure_gdown() -> None:
    """Install gdown if it is not already present."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("Installing gdown ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download(output_dir: str) -> None:
    _ensure_gdown()
    import gdown

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Downloading CICIDS2017 to: %s", output_dir)

    for filename, drive_id in CICIDS2017_FILES:
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest):
            logger.info("  Already exists, skipping: %s", filename)
            continue
        url = f"https://drive.google.com/uc?id={drive_id}"
        logger.info("  Downloading: %s ...", filename)
        gdown.download(url, dest, quiet=False)

    logger.info("Download complete.  %d files in %s.", len(CICIDS2017_FILES), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the CICIDS2017 dataset."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw"
        ),
        help="Directory to save the CSV files (default: data/raw/)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; just print the expected file paths.",
    )
    args = parser.parse_args()

    if args.skip_download:
        logger.info(
            "Skip-download mode.  Ensure the following files exist in %s:",
            args.output_dir,
        )
        for filename, _ in CICIDS2017_FILES:
            path   = os.path.join(args.output_dir, filename)
            status = "✓ found" if os.path.exists(path) else "✗ MISSING"
            logger.info("  [%s] %s", status, filename)
    else:
        download(args.output_dir)


if __name__ == "__main__":
    main()
