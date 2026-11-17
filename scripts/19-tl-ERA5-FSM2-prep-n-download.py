#!/usr/bin/env python3
"""Prepare monthly ERA5-Land hourly mean fluxes and static FSM2 vegetation."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import tempfile
import zipfile

import cdsapi
import dask
import numpy as np
import pandas as pd
import xarray as xr

from fsm2_flux import FORMAT, prepare_fluxes, time_name


DATASET = "reanalysis-era5-land"
VARIABLE_GROUPS = {
    "temp": ["2m_dewpoint_temperature", "2m_temperature"],
    "met": [
        "snowfall",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation",
    ],
}
CHUNKS = {"time": 24, "valid_time": 24, "latitude": 200, "longitude": 600}


def log(message):
    print(message, flush=True)


def write_nc(ds, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = {
        dim: min(size, CHUNKS[dim])
        for dim, size in ds.sizes.items() if dim in CHUNKS
    }
    ds = ds.drop_encoding().chunk(chunks)
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": tuple(chunks.get(dim, ds.sizes[dim]) for dim in var.dims),
        }
        for name, var in ds.data_vars.items() if var.ndim
    }

    with tempfile.TemporaryDirectory(dir=target.parent) as folder:
        temporary = Path(folder) / target.name
        ds.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        temporary.replace(target)

    log(f"Saved {target}")


def check_manifest(folder, settings):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "region_settings.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"Settings differ in {folder}; choose a new output root")
        return
    if any(folder.glob("????_??/*.nc")):
        raise ValueError(f"Existing NetCDFs without a settings manifest in {folder}")
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(settings, indent=2))
    temporary.replace(path)


def download_month(client, folder, month, area):
    tag = month.strftime("%Y_%m")
    target = folder / tag / f"{tag}.nc"
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "year": f"{month.year:04d}",
        "month": f"{month.month:02d}",
        "day": [f"{day:02d}" for day in range(1, month.days_in_month + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": area,
        "data_format": "netcdf",
        "download_format": "zip",
    }

    with tempfile.TemporaryDirectory(dir=target.parent) as folder_name, ExitStack() as stack:
        work = Path(folder_name)
        datasets = []

        for group, variables in VARIABLE_GROUPS.items():
            archive = work / f"{group}.zip"
            log(f"Downloading {tag} {group}")
            client.retrieve(DATASET, {**request, "variable": variables}, str(archive))

            with zipfile.ZipFile(archive) as zipped:
                members = [name for name in zipped.namelist() if name.endswith(".nc")]
                if not members:
                    raise ValueError(f"No NetCDF files inside {archive.name}")
                for number, member in enumerate(members):
                    part = work / f"{group}_{number}.nc"
                    with zipped.open(member) as source, part.open("wb") as output:
                        import shutil
                        shutil.copyfileobj(source, output)
                    datasets.append(stack.enter_context(xr.open_dataset(part, chunks={})))

        combined = xr.merge(datasets, compat="no_conflicts", join="exact")
        combined["w10"] = np.hypot(combined.u10, combined.v10)
        combined.w10.attrs = {"long_name": "10 m wind speed", "units": "m s**-1"}
        write_nc(combined, target)

    return target


def open_month(stack, client, raw_dir, month, area):
    path = download_month(client, raw_dir, month, area)
    ds = stack.enter_context(xr.open_dataset(path, chunks={}))
    time = time_name(ds)
    expected = pd.date_range(
        month.start_time, (month + 1).start_time, freq="h", inclusive="left"
    )
    if not pd.DatetimeIndex(ds[time].values).equals(expected):
        raise ValueError(f"Incomplete or out-of-order hourly timestamps: {path}")

    return ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])


def vegetation(path, variable, weather):
    with xr.open_dataset(path) as src:
        layer = src[variable]
        rename = {
            old: new for old, new in (("lat", "latitude"), ("lon", "longitude"))
            if old in layer.dims
        }
        layer = layer.rename(rename)
        if set(layer.dims) != {"latitude", "longitude"}:
            raise ValueError(f"Expected a static 2D {variable} layer in {path}")

        return (
            layer.sortby("latitude").sortby("longitude")
            .interp(latitude=weather.latitude, longitude=weather.longitude)
            .astype("float32").load()
        )


def select_regions(args):
    if args.sites_csv is None:
        regions = [(args.name, args.area)]
    else:
        table = pd.read_csv(args.sites_csv)
        columns = ["north", "west", "south", "east"]
        if not set(columns).issubset(table.columns):
            raise ValueError(f"CSV requires columns {columns}")
        rows = (
            list(range(len(table))) if args.candidates.lower() == "none"
            else json.loads(args.candidates)
        )
        if not isinstance(rows, list) or not rows:
            raise ValueError("--candidates must be a nonempty list or None")
        if any(type(i) is not int or not 0 <= i < len(table) for i in rows):
            raise ValueError("Candidate indices must be valid zero-based CSV rows")
        regions = []
        for i in dict.fromkeys(rows):
            row = table.iloc[i]
            name = row.get("site_id", row.get("candidate_id", f"row_{i:03d}"))
            regions.append((name, row[columns].astype(float).tolist()))

    result = []
    for name, area in regions:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        if name in ("", ".", ".."):
            raise ValueError("Invalid region name")
        north, west, south, east = area
        if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
            raise ValueError(f"Invalid region bounds for {name}: {area}")
        result.append((name, list(map(float, area))))

    if len({name for name, _ in result}) != len(result):
        raise ValueError("Region names must be unique")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-csv", type=Path)
    parser.add_argument("--candidates", default="None")
    parser.add_argument("--name", default="ontario")
    parser.add_argument("--area", type=float, nargs=4, default=[53, -95, 48, -87])
    parser.add_argument("--start-month", default="2018-07")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--era5-root", type=Path, default=Path("../../scratch/ERA5"))
    parser.add_argument("--fsm2-root", type=Path, default=Path("../../scratch/fsm2_flux"))
    parser.add_argument("--vai-path", type=Path, required=True)
    parser.add_argument("--vegh-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    months = pd.period_range(args.start_month, args.end_month, freq="M")
    if not len(months):
        parser.error("--end-month precedes --start-month")
    regions = select_regions(args)
    client = cdsapi.Client()

    for name, area in regions:
        raw_dir = args.era5_root / name
        output_dir = args.fsm2_root / name
        check_manifest(raw_dir, {
            "area": area, "dataset": DATASET, "variables": VARIABLE_GROUPS,
        })
        check_manifest(output_dir, {
            "area": area,
            "forcing_format": FORMAT,
            "vegetation": [
                {"path": str(p.resolve()), "size": p.stat().st_size,
                 "mtime_ns": p.stat().st_mtime_ns}
                for p in (args.vai_path, args.vegh_path)
            ],
        })

        for i, month in enumerate(months, 1):
            tag = month.strftime("%Y_%m")
            target = output_dir / tag / "driving_data.nc"
            if target.is_file():
                log(f"Skipping {target}")
                continue

            log(f"{name}: preparing {tag} ({i}/{len(months)})")
            with dask.config.set(scheduler="threads", num_workers=args.workers), ExitStack() as stack:
                weather = open_month(stack, client, raw_dir, month, area)
                previous = open_month(stack, client, raw_dir, month - 1, area)

                # A repairable gap at 23 UTC may need the next 00 UTC value.
                time = time_name(weather)
                tail = weather.ssrd.isel({time: slice(-5, None)}).load()
                needs_endpoint = (
                    ~np.isfinite(tail.isel({time: -1}))
                    & np.isfinite(tail).any(time)
                ).any().item()
                following = (
                    open_month(stack, client, raw_dir, month + 1, area)
                    if needs_endpoint else None
                )

                result, audit = prepare_fluxes(weather, previous, following)
                result["VAI"] = vegetation(args.vai_path, "VAI", weather)
                result["vegh"] = vegetation(args.vegh_path, "vegh", weather)
                result.attrs.update(
                    forcing_format=FORMAT,
                    interval="Mean over preceding hour ending at UTC timestamp",
                    ssrd_fill_method="neighbour_plateau_v1",
                    ssrd_fill_records=json.dumps(audit),
                )
                write_nc(result, target)


if __name__ == "__main__":
    main()
