"""Publish one complete regional NetCDF before removing its cell intermediates."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

FIELDS = ("snow_depth", "ground_snow_mass", "canopy_snow_mass")


def cell_path(dest, y, x):
    return dest / f"cell_y{y:04d}_x{x:04d}.nc"


def check_cell(ds, grid, times, signature, y, x):
    if ds.attrs["run_signature"] != signature or set(ds.data_vars) != set(FIELDS):
        raise ValueError(f"Incompatible cell at ({y}, {x})")
    np.testing.assert_array_equal(ds.time.values, times)
    np.testing.assert_array_equal([ds.latitude.item(), ds.longitude.item()],
                                  [grid.latitude[y].item(), grid.longitude[x].item()])


def finish(site, dest):
    final = dest / (dest.name + ".nc")
    tmp = dest / ("." + final.name + ".partial")
    files = sorted(site.glob("????_??/driving_data.nc"))
    with xr.open_dataset(files[0]) as forcing:
        grid = forcing[["VAI", "vegh"]].transpose("latitude", "longitude").load()
    mask = np.isfinite(grid.VAI) & np.isfinite(grid.vegh) & (grid.VAI > 0) & (grid.vegh > 2)
    cells = list(zip(*np.where(mask.values)))
    if not cells:
        raise ValueError("No eligible forest cells")
    config = (dest / "model.toml").read_text()
    if not final.exists():
        for y, x in cells:
            if not cell_path(dest, y, x).is_file():
                raise FileNotFoundError(cell_path(dest, y, x))
        with xr.open_dataset(cell_path(dest, *cells[0])) as first:
            times = first.time.values
            grid = grid.assign_coords(time=first.time.load().variable)
            grid.attrs = {k: v for k, v in first.attrs.items()
                          if k not in {"VAI", "vegh_m", "ssrd_fill_records"}}
            attrs = {k: first[k].attrs for k in FIELDS}
        grid["simulated_forest_mask"] = mask.astype("int8")
        grid.attrs.update(settings_toml=config, input_site=str(site.resolve()), run_name=dest.name)
        grid.to_netcdf(tmp, engine="netcdf4")
        audit = {}
        with Dataset(tmp, "a") as out:
            for k in FIELDS:
                var = out.createVariable(k, "f4", ("time", "latitude", "longitude"),
                                         chunksizes=(min(len(times), 8760), 1, 1),
                                         zlib=True, complevel=4, fill_value=np.float32(np.nan))
                var.setncatts(attrs[k])
            for y, x in cells:
                path = cell_path(dest, y, x)
                with xr.open_dataset(path) as ds:
                    check_cell(ds, grid, times, grid.attrs["run_signature"], y, x)
                    for k in FIELDS:
                        if ds[k].dims != ("time",):
                            raise ValueError(f"{path}: unexpected dimensions for {k}")
                        out[k][:, y, x] = ds[k].values
                    records = json.loads(ds.attrs.get("ssrd_fill_records", "[]"))
                    if records:
                        audit[path.stem] = records
            out.ssrd_fill_audit = json.dumps(audit)
    # Reopen the closed file and compare every cell before publication or cleanup.
    with xr.open_dataset(final if final.exists() else tmp) as region:
        if region.attrs["settings_toml"] != config or region.attrs["input_site"] != str(site.resolve()):
            raise ValueError("Existing regional file belongs to a different run")
        np.testing.assert_array_equal(region.simulated_forest_mask.values, mask.values)
        for y, x in cells:
            path = cell_path(dest, y, x)
            if final.exists() and not path.exists():  # cleanup resumed after publication
                continue
            with xr.open_dataset(path) as ds:
                check_cell(ds, grid, region.time.values, region.attrs["run_signature"], y, x)
                point = region.isel(latitude=y, longitude=x)
                for k in FIELDS:
                    np.testing.assert_array_equal(point[k].values, ds[k].values)
    if not final.exists():
        os.replace(tmp, final)
    for y, x in cells:
        cell_path(dest, y, x).unlink(missing_ok=True)
    print(final, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    finish(args.site, args.destination.resolve())
