#!/usr/bin/env python3
"""Combine forest-cell time series into one regional NetCDF."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


def assemble(template, cells, output):
    out = output / (cells.resolve().name + ".nc")
    if out.exists():
        raise FileExistsError(out)

    with xr.open_dataset(template) as src:
        grid = src[["VAI", "vegh"]].transpose("latitude", "longitude").load()
    mask = np.isfinite(grid.VAI) & np.isfinite(grid.vegh) & (grid.VAI > 0) & (grid.vegh > 2)
    grid["simulated_forest_mask"] = mask.astype("int8")
    grid.simulated_forest_mask.attrs["description"] = "1: eligible forest cell; 0: excluded"
    cells_to_read = [
        (y, x, cells / f"cell_y{y:04d}_x{x:04d}.nc")
        for y, x in zip(*np.where(mask.values))
    ]
    if not cells_to_read:
        raise ValueError("No eligible forest cells")
    available = []
    status = mask.values.astype("int8")  # 0 excluded, 1 eligible but missing
    for y, x, path in cells_to_read:
        if path.is_file():
            available.append((y, x, path))
            status[y, x] = 2
        else:
            print(f"Missing eligible cell: {path.name}", flush=True)
    print(f"Assembling {len(available)} of {len(cells_to_read)} eligible cells", flush=True)
    if not available:
        raise ValueError("No completed eligible cell files found")
    cells_to_read = available
    grid["cell_status"] = (("latitude", "longitude"), status)
    grid.cell_status.attrs.update(flag_values=np.array([0, 1, 2], dtype="int8"),
                                  flag_meanings="excluded eligible_missing assembled")
    grid.simulated_forest_mask.attrs["description"] = "Forest eligibility: 1 eligible, 0 excluded; see cell_status for available results"

    with xr.open_dataset(cells_to_read[0][2]) as first:
        times = first.time.load()
        grid = grid.assign_coords(time=times.variable)
        grid.attrs = {k: v for k, v in first.attrs.items() if not k.startswith("ssrd_fill_") and k not in {"VAI", "vegh_m"}}
        signature = grid.attrs["run_signature"]
        variables = {name: dict(var.attrs) for name, var in first.data_vars.items()}

    match = re.search(r"^\s*#define\s+SNFRAC\s+([123])\b", grid.attrs.get("model_options", ""), re.MULTILINE)
    if match is None or variables.get("snow_depth", {}).get("units") != "m":
        raise ValueError("Expected recorded SNFRAC option and snow_depth in metres")
    option = int(match.group(1))
    hfsn = 0.10  # FSM2 default used by the runner's empty &params namelist
    fraction_name = "ground_snow_cover_fraction"
    output_variables = dict(variables)
    output_variables[fraction_name] = {
        "long_name": "Ground snow-cover fraction derived from hourly snow depth",
        "units": "1", "SNFRAC": option, "hfsn_m": hfsn,
        "formula": {1: "min(snow_depth / hfsn, 1)", 2: "tanh(snow_depth / hfsn)",
                    3: "snow_depth / (snow_depth + hfsn)"}[option],
    }

    output.mkdir(parents=True, exist_ok=True)
    # Only closed snapshots are committed. A killed writer can damage its
    # working copy without touching the last committed NetCDF.
    checkpoint = output / f".{out.stem}.assembly.nc"
    working = output / f".{out.stem}.working.nc"
    digest = hashlib.sha256(b"assembly-cell-status-ssrd-fraction-v3")
    for path in [template, *(entry[2] for entry in cells_to_read)]:
        stat = path.stat()
        digest.update(repr((str(path.resolve()), stat.st_size, stat.st_mtime_ns)).encode())
    fingerprint = digest.hexdigest()
    total = len(cells_to_read)
    with (output / f".{out.stem}.assembly.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            raise FileExistsError(out)
        working.unlink(missing_ok=True)
        if checkpoint.exists():
            with Dataset(checkpoint) as saved:
                if saved.assembly_inputs != fingerprint:
                    raise ValueError(f"Inputs changed since {checkpoint} was created")
                done = int(saved.assembly_completed_cells)
            print(f"Resuming at {done}/{total} completed cells", flush=True)
        else:
            done = 0
            grid.to_netcdf(working, engine="netcdf4")
            with Dataset(working, "a") as target:
                for name, attrs in output_variables.items():
                    var = target.createVariable(
                        name, "f4", ("time", "latitude", "longitude"),
                        chunksizes=(min(8760, times.size), 1, 1),
                        zlib=True, complevel=4, fill_value=np.float32(np.nan),
                    )
                    var.setncatts(attrs)
                target.assembly_inputs = fingerprint
                target.assembly_completed_cells = 0
            os.replace(working, checkpoint)

        while done < total:
            end = min(done + 50, total)
            shutil.copyfile(checkpoint, working)
            with Dataset(working, "a") as target:
                repairs = json.loads(getattr(target, "ssrd_fill_audit", "{}"))
                for y, x, path in cells_to_read[done:end]:
                    with xr.open_dataset(path) as ds:
                        if ds.attrs.get("run_signature") != signature:
                            raise ValueError(f"Mixed configurations: {path}")
                        np.testing.assert_array_equal(ds.time.values, times.values, err_msg=str(path))
                        np.testing.assert_allclose(
                            [ds.latitude.item(), ds.longitude.item()],
                            [grid.latitude[y].item(), grid.longitude[x].item()], err_msg=str(path),
                        )
                        records = json.loads(ds.attrs.get("ssrd_fill_records", "[]"))
                        if records:
                            repairs[path.stem] = dict(method=ds.attrs["ssrd_fill_method"],
                                                      runner=ds.attrs["ssrd_fill_runner_sha256"], gaps=records)
                        if set(ds.data_vars) != set(variables):
                            raise ValueError(f"Unexpected output variables in {path}")
                        for name in variables:
                            if ds[name].dims != ("time",):
                                raise ValueError(f"Expected hourly series for {name} in {path}")
                            values = ds[name].values
                            target[name][:, y, x] = values
                            if name == "snow_depth":
                                if option == 1:
                                    fraction = np.minimum(values / hfsn, 1)
                                elif option == 2:
                                    fraction = np.tanh(values / hfsn)
                                else:
                                    fraction = values / (values + hfsn)
                                target[fraction_name][:, y, x] = fraction
                target.ssrd_fill_audit = json.dumps(repairs)
                target.assembly_completed_cells = end
            os.replace(working, checkpoint)
            done = end
            print(f"Checkpoint saved: {done}/{total} cells", flush=True)

        os.link(checkpoint, out)
        checkpoint.unlink()
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="Regional driving_data.nc")
    parser.add_argument("--cells", type=Path, required=True, help="Regional cell directory, e.g. S01")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory")
    args = parser.parse_args()
    assemble(args.template, args.cells, args.output)
