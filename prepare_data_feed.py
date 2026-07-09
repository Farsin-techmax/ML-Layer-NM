"""
Convert Q2-2026 xlsx data files to the csv filenames expected by the
pmstrainfeatureEng and predservicemil pipelines.

Applies column renames:
  - RFM:  Contact Key  →  Customer ID
  - VHC:  Refined Description  →  Refined Description (Enhanced)

Usage:
    python prepare_data_feed.py              # defaults: data/ → data/
    python prepare_data_feed.py --src ./raw  # custom source dir
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Source xlsx files ────────────────────────────────────────────────
SOURCE_FILES = [
    "EDA_Q2-2026.xlsx",
    "Service_History_Q2-2026.xlsx",
    "Appointments_Q2-2026.xlsx",
    "VHC_Q2-2026.xlsx",
    "RFM_Q2-2026.xlsx",
    "Digital_Sessions_Q2-2026.xlsx",
    "Service_Code_Desc.xlsx",
]

# ── Target csv filenames per pipeline ───────────────────────────────
#   key   = source xlsx basename
#   value = list of csv names to produce (train + pred copies)
TARGET_NAMES: dict[str, list[str]] = {
    "EDA_Q2-2026.xlsx": [
        "EDA - Q3 2025.csv",
        "EDA Datasheet Till 2025.csv",
    ],
    "Service_History_Q2-2026.xlsx": [
        "Service History Q1 - 2026.csv",
        "Service History Feb 2026.csv",
    ],
    "Appointments_Q2-2026.xlsx": [
        "Appoinments - Q3 - 2025.csv",
        "Appoinments2025.csv",
    ],
    "VHC_Q2-2026.xlsx": [
        "VHC Q3 - 2025.csv",
        "VHC PMS till 2025.csv",
    ],
    "RFM_Q2-2026.xlsx": [
        "RFM Segments Q2 - 2025.csv",
        "RFM till 2025.csv",
    ],
    "Digital_Sessions_Q2-2026.xlsx": [
        "Digital sessions Q3 - 2025.csv",
        "DigitalData2025.csv",
    ],
    "Service_Code_Desc.xlsx": [
        "Service Code Desc.csv",
    ],
}

# ── Column renames per source file ──────────────────────────────────
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "RFM_Q2-2026.xlsx": {"Contact Key": "Customer ID"},
    "VHC_Q2-2026.xlsx": {"Refined Description": "Refined Description (Enhanced)"},
}


def convert_all(src_dir: Path, out_dir: Path) -> None:
    """Read each xlsx, apply renames, and write csv copies."""
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for xlsx_name in SOURCE_FILES:
        xlsx_path = src_dir / xlsx_name
        if not xlsx_path.exists():
            print(f"[SKIP] {xlsx_name} not found in {src_dir}")
            continue

        print(f"[READ] {xlsx_path.name} …", end=" ", flush=True)
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        print(f"{len(df):,} rows × {len(df.columns)} cols")

        # apply column renames if needed
        renames = COLUMN_RENAMES.get(xlsx_name)
        if renames:
            missing = [c for c in renames if c not in df.columns]
            if missing:
                print(f"  [WARN] columns to rename not found: {missing}")
            df = df.rename(columns=renames)
            print(f"  [RENAME] {renames}")

        # write csv copies
        for csv_name in TARGET_NAMES[xlsx_name]:
            csv_path = out_dir / csv_name
            df.to_csv(csv_path, index=False)
            print(f"  -> {csv_name}")

    print("\n[DONE] All conversions complete.")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Convert Q2-2026 xlsx data to pipeline-ready csv files."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing source xlsx files (default: ./data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for csv files (default: same as --src)",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"[ERROR] Source directory does not exist: {args.src}", file=sys.stderr)
        sys.exit(1)

    convert_all(args.src, args.out)


if __name__ == "__main__":
    main()
