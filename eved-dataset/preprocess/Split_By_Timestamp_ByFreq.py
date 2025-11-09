import argparse
import os
from typing import Optional

import pandas as pd


# Script: Split_By_Timestamp_ByFreq.py
# Purpose: Read VED_Static_Data_PHEV&EV.xlsx, detect timestamp column, and split
# the dataset into multiple files by a given pandas frequency (e.g. 'D' for day,
# 'H' for hour). Outputs CSV files named with the time window (safe filenames).


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {
        "timestamp",
        "time",
        "date",
        "datetime",
        "ts",
        "time_stamp",
        "time stamp",
        "record_time",
    }]
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


def safe_fname(s: str) -> str:
    # replace problematic chars
    return s.replace(" ", "_").replace(":", "-").replace("/", "-")


def main():
    parser = argparse.ArgumentParser(description="Split Excel by timestamp frequency (day/hour/month)")
    parser.add_argument("--input", "-i", default="./data/VED_Static_Data_PHEV&EV.xlsx", help="Input Excel file path")
    parser.add_argument("--out_dir", "-o", default=None, help="Output directory (default: same folder as input)/split_by_freq")
    parser.add_argument("--freq", "-f", default="D", help="Pandas frequency string, e.g. 'D' (day), 'H' (hour), 'M' (month). Default 'D'")
    parser.add_argument("--timestamp_col", default=None, help="Timestamp column name (auto-detected if omitted)")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum rows in a group to be saved")
    parser.add_argument("--downsample_ms", type=int, default=None, help="If provided, resample the whole dataset keeping the first row every N milliseconds (e.g. 1000)")
    parser.add_argument("--save_excel", action="store_true", help="Also save each group as an Excel file")

    args = parser.parse_args()

    input_file = args.input
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_excel(input_file)

    ts_col = args.timestamp_col or detect_timestamp_column(df)
    if ts_col is None:
        raise ValueError("Cannot detect timestamp column. Provide --timestamp_col explicitly.")

    print(f"Using timestamp column: {ts_col}")

    # parse and drop invalid timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    n_null = df[ts_col].isna().sum()
    if n_null > 0:
        print(f"Warning: {n_null} rows have invalid timestamps and will be dropped.")
        df = df.dropna(subset=[ts_col]).reset_index(drop=True)

    df = df.sort_values(ts_col).reset_index(drop=True)

    # If downsample_ms is provided, perform a global resample (keep first row per interval)
    if args.downsample_ms is not None:
        ms = int(args.downsample_ms)
        if ms <= 0:
            raise ValueError("--downsample_ms must be a positive integer representing milliseconds")
        # set index to timestamp and resample
        df_idx = df.set_index(ts_col)
        try:
            down = df_idx.resample(f"{ms}ms").first().dropna().reset_index()
        except Exception as e:
            raise RuntimeError(f"Resampling failed: {e}")

        out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(input_file)), "split_by_freq")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_file))[0]
        out_csv = os.path.join(out_dir, f"{base}_downsampled_{ms}ms.csv")
        down.to_csv(out_csv, index=False)
        if args.save_excel:
            out_x = os.path.join(out_dir, f"{base}_downsampled_{ms}ms.xlsx")
            with pd.ExcelWriter(out_x) as w:
                down.to_excel(w, index=False)
        print(f"Saved downsampled file ({len(down)} rows) to: {out_csv}")
        if args.save_excel:
            print(f"Also saved Excel to: {out_x}")
        # finish early
        return

    # build output directory
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(input_file)), "split_by_freq")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_file))[0]

    # group using pandas.Grouper
    freq = args.freq
    grouper = pd.Grouper(key=ts_col, freq=freq)

    grouped = df.groupby(grouper)

    saved = 0
    skipped = 0
    for grp_name, grp_df in grouped:
        # grp_name may be Timestamp or NaT
        if grp_name is pd.NaT or grp_df.empty:
            skipped += 1
            continue

        if len(grp_df) < args.min_count:
            skipped += 1
            continue

        # format group name to string depending on freq
        if isinstance(grp_name, pd.Timestamp):
            # use ISO-like formatting; include time when freq is hourly or finer
            if freq.upper().startswith("H") or freq.upper().startswith("T") or freq.upper().startswith("S"):
                name_str = grp_name.strftime("%Y-%m-%d_%H-%M-%S")
            elif freq.upper().startswith("D"):
                name_str = grp_name.strftime("%Y-%m-%d")
            elif freq.upper().startswith("M"):
                name_str = grp_name.strftime("%Y-%m")
            else:
                # generic
                name_str = grp_name.isoformat()
        else:
            name_str = str(grp_name)

        fname_safe = safe_fname(name_str)
        csv_path = os.path.join(out_dir, f"{base}_{fname_safe}.csv")
        grp_df.to_csv(csv_path, index=False)
        if args.save_excel:
            xlsx_path = os.path.join(out_dir, f"{base}_{fname_safe}.xlsx")
            with pd.ExcelWriter(xlsx_path) as w:
                grp_df.to_excel(w, index=False)
        saved += 1

    print(f"Finished splitting. Saved groups: {saved}. Skipped groups: {skipped}. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
