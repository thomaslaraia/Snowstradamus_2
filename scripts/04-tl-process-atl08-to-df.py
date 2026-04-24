#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse

import yaml
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.parallel_blocks_v2 import *


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

    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Build dirpath from user argument
    dirpath = f"../../scratch/{args.folder}/"

    all_ATL03, all_ATL08, failed_ATL03 = track_pairs(dirpath, failed=True)
    N = len(all_ATL03)

    all_data = []

    for i in range(N):
        print(f"Processing {i + 1}/{N}: {all_ATL03[i]}")

        try:
            data = pvpg_parallel(
                dirpath,
                all_ATL03[i],
                all_ATL08[i],
                cfg=cfg,
                coords=None,
                file_index=i,
                graph_detail=1,
            )

            if data is not None and len(data) > 0:
                data = data.copy()
                data["file_index"] = i
                data["ATL03"] = all_ATL03[i]
                data["ATL08"] = all_ATL08[i]
                all_data.append(data)

        except Exception as e:
            print(f"Failed on file_index={i}: {all_ATL03[i]}")
            print(e)

    if len(all_data) == 0:
        print("No dataframes were returned. Nothing saved.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    output_name = args.output
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / output_name

    df_all.to_pickle(out_path)

    print(f"Saved combined dataframe with {len(df_all)} rows to:")
    print(out_path)


if __name__ == "__main__":
    main()