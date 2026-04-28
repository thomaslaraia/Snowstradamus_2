#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
from datetime import datetime
import time
import gc

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

def make_output_filename(atl03_file, file_index):
    """
    Create filename like 181105_N0023.parquet from ATL03 datetime and index.
    """
    dt = parse_filename_datetime(atl03_file)
    date_tag = dt.strftime("%y%m%d")

    return f"{date_tag}_N{file_index:04d}.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        help="Folder name inside scratch containing ATL03/08 files",
    )
    parser.add_argument(
        "output",
        help="Output dataset name. Example: ontario_test",
    )

    args = parser.parse_args()

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dirpath = f"../../scratch/{args.folder}/"

    all_ATL03, all_ATL08, failed_ATL03 = track_pairs(dirpath, failed=True)
    N = len(all_ATL03)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = Path(args.output).stem
    dataset_dir = output_dir / f"{output_stem}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Started at: {start_datetime}")
    print(f"Elapsed: {elapsed_str(start_time)}")
    print(f"N = {N}")
    print(f"Writing Parquet files to:")
    print(dataset_dir)

    for i in range(N):
        output_filename = make_output_filename(all_ATL03[i], i)

        out_file = dataset_dir / output_filename
        tmp_file = dataset_dir / f".{output_filename}.tmp"

        if out_file.exists():
            print(f"Skipping existing file_index={i}: {out_file.name}")
            continue

        print(f"Processing {i + 1}/{N}: {all_ATL03[i]}")
        print(f"Output file: {out_file.name}")

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

            if data is None or len(data) == 0:
                print(f"No valid data returned for file_index={i}")
                continue

            print(f"Valid data returned after {elapsed_str(start_time)} for i={i}")

            data = data.copy()
            data["file_index"] = i
            data["ATL03"] = all_ATL03[i]
            data["ATL08"] = all_ATL08[i]

            data.to_parquet(
                tmp_file,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )

            tmp_file.replace(out_file)

            print(f"Saved {len(data)} rows to:")
            print(out_file)

            del data
            gc.collect()

        except Exception as e:
            print(f"Failed on file_index={i}: {all_ATL03[i]}")
            print(e)

            if tmp_file.exists():
                tmp_file.unlink()

    print(f"Finished after: {elapsed_str(start_time)}")
    print(f"Output dataset:")
    print(dataset_dir)


if __name__ == "__main__":
    main()
