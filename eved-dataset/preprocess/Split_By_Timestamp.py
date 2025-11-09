import argparse
import os
from typing import Optional, Tuple

import pandas as pd

# === Configuration / Usage ===
# This script reads an Excel file (default: VED_Static_Data_PHEV&EV.xlsx),
# detects a timestamp column, sorts rows by timestamp and splits the dataset
# into train/val/test sets either by ratio or by explicit date thresholds.
# Outputs are written as CSV files in the same folder by default.


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {"timestamp", "time", "date", "datetime", "ts", "time_stamp", "time stamp", "record_time"}]
    if candidates:
        return candidates[0]
    # fallback: pick first datetime-typed column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # try to parse common columns
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return None


def split_by_ratio(df: pd.DataFrame, ratios: Tuple[float, float, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    n = len(df)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def split_by_dates(df: pd.DataFrame, ts_col: str, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # train: <= train_end, val: >train_end and <= val_end, test: > val_end
    te = pd.to_datetime(train_end)
    ve = pd.to_datetime(val_end)
    train = df[df[ts_col] <= te].copy()
    val = df[(df[ts_col] > te) & (df[ts_col] <= ve)].copy()
    test = df[df[ts_col] > ve].copy()
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Split VED static Excel by timestamp into train/val/test")
    parser.add_argument("--input", "-i", default="./data/VED_Static_Data_PHEV&EV.xlsx", help="Path to input Excel file")
    parser.add_argument("--out_dir", "-o", default=None, help="Output directory (default: same folder as input)")
    parser.add_argument("--mode", choices=["ratio", "date"], default="ratio", help="Split mode: ratio or date")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="Train/Val/Test ratios (sum must be 1)")
    parser.add_argument("--train_end", type=str, default=None, help="If mode=date: train end timestamp (ISO format) e.g. 2020-06-30")
    parser.add_argument("--val_end", type=str, default=None, help="If mode=date: val end timestamp (ISO format) e.g. 2020-09-30")
    parser.add_argument("--timestamp_col", type=str, default=None, help="Name of timestamp column (auto-detected if omitted)")
    parser.add_argument("--save_excel", action="store_true", help="Also save outputs as Excel files in addition to CSV")

    args = parser.parse_args()

    input_file = args.input
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_excel(input_file)

    # detect timestamp column
    ts_col = args.timestamp_col or detect_timestamp_column(df)
    if ts_col is None:
        raise ValueError("Cannot detect timestamp column. Please provide --timestamp_col explicitly.")

    print(f"Using timestamp column: {ts_col}")

    # convert to datetime
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    n_null_ts = df[ts_col].isna().sum()
    if n_null_ts > 0:
        print(f"Warning: {n_null_ts} rows have invalid/NaT timestamps and will be dropped.")
        df = df.dropna(subset=[ts_col]).reset_index(drop=True)

    # sort by time ascending
    df = df.sort_values(ts_col).reset_index(drop=True)

    if args.mode == "ratio":
        train, val, test = split_by_ratio(df, tuple(args.ratios))
        print(f"Split by ratio {args.ratios}: train={len(train)}, val={len(val)}, test={len(test)}")
    else:
        if not args.train_end or not args.val_end:
            raise ValueError("For mode=date you must provide --train_end and --val_end")
        train, val, test = split_by_dates(df, ts_col, args.train_end, args.val_end)
        print(f"Split by dates: train<={args.train_end} ({len(train)}), {args.train_end}<val<={args.val_end} ({len(val)}), test>{args.val_end} ({len(test)})")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(input_file))
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    train_csv = os.path.join(out_dir, f"{base_name}_train.csv")
    val_csv = os.path.join(out_dir, f"{base_name}_val.csv")
    test_csv = os.path.join(out_dir, f"{base_name}_test.csv")

    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)
    test.to_csv(test_csv, index=False)
    print(f"Saved CSVs to: {train_csv}, {val_csv}, {test_csv}")

    if args.save_excel:
        train_x = os.path.join(out_dir, f"{base_name}_train.xlsx")
        val_x = os.path.join(out_dir, f"{base_name}_val.xlsx")
        test_x = os.path.join(out_dir, f"{base_name}_test.xlsx")
        with pd.ExcelWriter(train_x) as w:
            train.to_excel(w, index=False)
        with pd.ExcelWriter(val_x) as w:
            val.to_excel(w, index=False)
        with pd.ExcelWriter(test_x) as w:
            test.to_excel(w, index=False)
        print(f"Also saved Excel files to: {train_x}, {val_x}, {test_x}")


if __name__ == "__main__":
    main()
