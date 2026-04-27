#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
from datetime import datetime
import time

import yaml
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.parallel_blocks_v2 import *


def elapsed_str(start_time):
    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        help="Folder name inside scratch containing ATL03/08 files",
    )
    parser.add_argument(
        "output",
        help="Output pickle name.",
    )

    args = parser.parse_args()

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Build dirpath from user argument
    dirpath = f"../../scratch/{args.folder}/"

    all_ATL03, all_ATL08, failed_ATL03 = track_pairs(dirpath, failed=True)
    N = len(all_ATL03)

    output_name = args.output
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / output_name
    backup_path = output_dir / f"{Path(output_name).stem}_BACKUP.pkl"

    print(f"Started at: {start_datetime}")
    print(f"Elapsed: {elapsed_str(start_time)}")
    print(f"N = {N}")

    all_data = []
    completed_indices = set()

    # Resume from backup if it exists
    if backup_path.exists():
        print(f"Found backup pickle, resuming from:")
        print(backup_path)

        backup_df = pd.read_pickle(backup_path)

        if "file_index" not in backup_df.columns:
            raise ValueError(
                f"Backup file exists but does not contain 'file_index': {backup_path}"
            )

        completed_indices = set(backup_df["file_index"].dropna().astype(int).unique())
        all_data = [backup_df]

        print(f"Loaded backup with {len(backup_df)} rows.")
        print(f"Already completed {len(completed_indices)} files.")
        print(f"Continuing from remaining files.")

    for i in range(N):
        if i in completed_indices:
            print(f"Skipping already completed file_index={i}: {all_ATL03[i]}")
            continue

        print(f"Processing {i + 1}/{N}: {all_ATL03[i]}")

        try:
            data = pvpg_parallel(
                dirpath,
                all_ATL03[i],
                all_ATL08[i],
                cfg=cfg,
                coords=None,
                file_index=i,
                graph_detail=0,
            )

            if data is not None and len(data) > 0:
                print(f"Valid data returned after {elapsed_str(start_time)} for i={i}")

                data = data.copy()
                data["file_index"] = i
                data["ATL03"] = all_ATL03[i]
                data["ATL08"] = all_ATL08[i]
                all_data.append(data)
                completed_indices.add(i)

                # Save intermediate backup after each successful file
                df_backup = pd.concat(all_data, ignore_index=True)
                df_backup.to_pickle(backup_path)

                print(f"Saved backup with {len(df_backup)} rows to:")
                print(backup_path)

        except Exception as e:
            print(f"Failed on file_index={i}: {all_ATL03[i]}")
            print(e)

    if len(all_data) == 0:
        print("No dataframes were returned. Nothing saved.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    df_all.to_pickle(out_path)

    print(f"Saved combined dataframe with {len(df_all)} rows to:")
    print(out_path)

    # Delete backup after successful final save
    if backup_path.exists():
        backup_path.unlink()
        print(f"Deleted backup pickle:")
        print(backup_path)

    print(f"Finished after: {elapsed_str(start_time)}")


if __name__ == "__main__":
    main()
