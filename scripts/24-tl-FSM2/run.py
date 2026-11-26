#!/usr/bin/env python3
"""Run independent FSM2 forest cells from monthly regional forcing files."""

from model import prepare
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import dask
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsm2_flux import FORMAT, FIELDS, normalized

MET = ["t2m", "d2m", "sf", "tp", "ssrd", "strd", "u10", "v10", "sp"]
PIN = "a76945e75b36628161f4679428ed72dce090b368"


def report(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


@contextmanager
def stage(name):
    report(name)
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            report(f"{name} | {time.monotonic() - start:.0f}s elapsed")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def open_region(root):
    files = sorted(root.glob("????_??/driving_data.nc"))
    if not files:
        raise ValueError(f"No monthly driving_data.nc under {root}")

    with xr.open_dataset(files[0]) as src:
        static = src[["VAI", "vegh"]].transpose("latitude", "longitude").load()

    audit = {}
    for i, path in enumerate(files, 1):
        report(f"Checking month {i}/{len(files)}: {path.parent.name}")
        with xr.open_dataset(path) as src:
            ds = normalized(src)
            for dim in ("latitude", "longitude"):
                np.testing.assert_array_equal(ds[dim].values, static[dim].values)
            for name in ("VAI", "vegh"):
                layer = ds[name].transpose("latitude", "longitude")
                np.testing.assert_allclose(
                    layer.values, static[name].values,
                    rtol=0, atol=0, equal_nan=True,
                )
            for name in MET:
                if set(ds[name].dims) != {"time", "latitude", "longitude"}:
                    raise ValueError(f"{path}: unexpected dimensions for {name}")

            fmt = ds.attrs.get("forcing_format")
            if fmt != FORMAT:
                raise ValueError(f"{path}: expected {FORMAT}, found {fmt!r}")
            for name, (unit, _, _, _) in FIELDS.items():
                if ds[name].attrs.get("units") != unit:
                    raise ValueError(f"{path}: expected {unit} for {name}")
            for key, gaps in json.loads(ds.attrs["ssrd_fill_records"]).items():
                audit.setdefault(key, []).extend(gaps)

    ds = xr.open_mfdataset(
        files,
        preprocess=lambda d: normalized(d)[MET],
        combine="nested",
        concat_dim="time",
        join="exact",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        chunks={},
    )

    t = pd.DatetimeIndex(ds.time.values)
    if (len(t) < 2 or t.hasnans or not (t == t.floor("h")).all()
            or not np.all(np.diff(t.values) == np.timedelta64(1, "h"))):
        ds.close()
        raise ValueError("Expected continuous whole-hour UTC timestamps")

    return files, ds, static, audit


def forcing(point):
    t = pd.DatetimeIndex(point.time.values)
    a = {name: np.asarray(point[name].values, dtype="float64") for name in MET}

    for name, values in a.items():
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad):
            raise ValueError(f"{name}: {len(bad)} nonfinite hours; first {t[bad[0]]} UTC")

    if np.any((a["t2m"] < 180) | (a["t2m"] > 340)):
        raise ValueError("Air temperature outside expected Kelvin range")
    if np.any((a["d2m"] < 170) | (a["d2m"] > 340)):
        raise ValueError("Dewpoint outside expected Kelvin range")
    if np.any((a["sp"] < 20000) | (a["sp"] > 110000)):
        raise ValueError("Surface pressure outside expected Pa range")

    rates = {name: a[name] for name in FIELDS}
    for name, values in rates.items():
        if np.any(values < 0):
            raise ValueError(f"{name}: negative flux")

    rain = rates["tp"] - rates["sf"]
    if np.any(rain < -1e-7 * 1000 / 3600):
        raise ValueError("Hourly snowfall exceeds total precipitation beyond tolerance")

    td = a["d2m"] - 273.15
    tc = a["t2m"] - 273.15
    vapour = 611.2 * np.exp(17.67 * td / (td + 243.5))
    saturation = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    q = 0.622 * vapour / (a["sp"] - 0.378 * vapour)
    rh = 100 * vapour / saturation

    mid = t - pd.Timedelta(minutes=30)
    matrix = np.column_stack([
        mid.year, mid.month, mid.day, mid.hour + mid.minute / 60,
        rates["ssrd"], rates["strd"], np.maximum(rain, 0), rates["sf"],
        a["t2m"], q, rh, np.hypot(a["u10"], a["v10"]), a["sp"],
    ])
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite converted forcing")
    return t, matrix


def namelist(vai, height, lat, lon, restart, parameters):
    return f"""&params
 {parameters}
/
&gridpnts
 Npnts=1
/
&gridlevs
/
&drive
 met_file='met.txt', dt=3600, lat={lat:.8f}, noon={12-lon/15:.8f}, zT=2, zU=10
/
&veg
 VAI={vai:.8f}, vegh={height:.8f}, alb0=0.2
/
&initial
 start_file='{restart}'
/
&outputs
 runid='', dump_file='dump.txt'
/
"""


def execute(exe, work, matrix, settings, canopy_layers):
    np.savetxt(
        work / "met.txt", matrix,
        fmt=["%d", "%d", "%d", "%.3f"] + ["%.9e"] * 9,
    )
    (work / "namelist.txt").write_text(settings)
    for name in ("stat.txt", "flux.txt", "subc.txt", "dump.txt"):
        (work / name).unlink(missing_ok=True)

    count = len(matrix)
    with (work / "namelist.txt").open() as stdin, (work / "model.log").open("w") as log:
        with subprocess.Popen(
            [str(exe)], cwd=work, stdin=stdin, stdout=log, stderr=subprocess.STDOUT
        ) as proc:
            start = time.monotonic()
            position = complete = 0
            while True:
                try:
                    status = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    stats = work / "stat.txt"
                    if stats.exists():
                        with stats.open("rb") as f:
                            f.seek(position)
                            complete += f.read().count(b"\n")
                            position = f.tell()
                    report(f"FSM2: {complete}/{count} hours | "
                           f"{time.monotonic() - start:.0f}s elapsed")
            if status:
                raise subprocess.CalledProcessError(status, str(exe))

    result = np.loadtxt(work / "stat.txt", ndmin=2)
    if result.shape != (count, 12 + canopy_layers) or not np.isfinite(result).all():
        raise ValueError(f"Unexpected/nonfinite FSM2 output: {result.shape}")
    if np.any(result[:, 4:7] < -1e-6):
        raise ValueError("Negative snow state in model output")
    if not (work / "dump.txt").is_file():
        raise ValueError("FSM2 did not write its restart state")
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-stop", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg, exe = prepare(args.config)
    args.spinup_cycles = cfg.get("run", {}).get("spinup_cycles", 2)
    if args.limit is not None and args.limit < 1:
        parser.error("Use a positive limit")
    parameters = ",\n ".join(f"{k}={v}" for k, v in cfg.get("params", {}).items())
    opts = (exe.parent / "OPTS.h").read_text()
    model_settings = json.dumps(cfg, sort_keys=True)

    with stage("Preparing regional input"):
        files, ds, static, audit = open_region(args.input.resolve())

    args.output.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve().parents[1] / "fsm2_flux.py"
    payload = {
        "inputs": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        "executable_sha256": sha256(exe),
        "wrapper_sha256": sha256(Path(__file__)),
        "flux_module_sha256": sha256(helper),
        "spinup_cycles": args.spinup_cycles,
        "accumulation": "flux", "model": args.config.stem, "model_settings": model_settings, "parameters": parameters,
        "source_commit": PIN,
        "minimum_height": 2.0,
    }
    sig = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    try:
        stop = ds.sizes["latitude"] if args.row_stop is None else min(
            args.row_stop, ds.sizes["latitude"]
        )
        if not 0 <= args.row_start < stop:
            parser.error("Empty or invalid latitude row range")

        mask = (
            np.isfinite(static.VAI) & np.isfinite(static.vegh)
            & (static.VAI > 0) & (static.vegh > 2)
        )
        ys, xs = np.where(mask.isel(latitude=slice(args.row_start, stop)).values)
        cells = list(zip(ys + args.row_start, xs))
        if args.limit is not None:
            cells = cells[:args.limit]

        report(f"{len(cells)} eligible cells selected")
        failures = 0
        saved = 0
        elapsed_saved = 0.0

        for i, (y, x) in enumerate(cells, 1):
            stem = f"cell_y{y:04d}_x{x:04d}"
            out = args.output / f"{stem}.nc"
            report(f"Cell {i}/{len(cells)}: {out.name}")

            if out.is_file():
                with xr.open_dataset(out) as old:
                    if old.attrs.get("run_signature") != sig:
                        raise ValueError(f"{out}: existing result has a different signature")
                report(f"Already complete: {out.name}")
                continue

            started = time.monotonic()
            work = Path(tempfile.mkdtemp(prefix=f"work_y{y}_x{x}_", dir=args.output))
            try:
                with stage("Loading hourly weather for cell"), dask.config.set(scheduler="synchronous"):
                    point = ds.isel(latitude=y, longitude=x).load()
                    records = audit.get(stem, [])


                if records:
                    report("SSRD fills: " + json.dumps(records))
                times, matrix = forcing(point)
                lat, lon = float(point.latitude), float(point.longitude)
                vai = float(static.VAI.values[y, x])
                height = float(static.vegh.values[y, x])
                restart = "none"

                if args.spinup_cycles:
                    anniversary = times[0] + pd.DateOffset(years=1)
                    if times[-1] < anniversary - pd.Timedelta(hours=1):
                        raise ValueError("Spin-up requires at least one calendar year")
                    spin = matrix[times < anniversary]
                    for cycle in range(args.spinup_cycles):
                        report(f"Spin-up {cycle + 1}/{args.spinup_cycles}: {len(spin)} hours")
                        execute(exe, work, spin, namelist(vai, height, lat, lon, restart, parameters), cfg["physics"]["CANMOD"])
                        shutil.copy2(work / "dump.txt", work / "start.txt")
                        restart = "start.txt"

                report(f"Production run: {len(matrix)} hours")
                result = execute(exe, work, matrix, namelist(vai, height, lat, lon, restart, parameters), cfg["physics"]["CANMOD"])
                np.testing.assert_allclose(result[:, :4], matrix[:, :4], rtol=0, atol=0.001)

                data = xr.Dataset(
                    {
                        "snow_depth": ("time", result[:, 4].astype("float32"), {"units": "m"}),
                        "ground_snow_mass": (
                            "time", result[:, 5].astype("float32"), {"units": "kg m-2"}
                        ),
                        "canopy_snow_mass": (
                            "time", result[:, 6].astype("float32"),
                            {"units": "kg m-2",
                             "long_name": "Total snow mass on vegetation per model ground area"},
                        ),
                    },
                    coords={"time": times, "latitude": lat, "longitude": lon},
                    attrs={
                        "run_signature": sig, "source_commit": PIN,
                        "VAI": vai, "vegh_m": height,
                        "spinup_cycles": args.spinup_cycles, "accumulation": "flux", "model": args.config.stem, "model_settings": model_settings, "parameters": parameters,
                        "time_convention": "UTC interval end; state after one-hour step; solar geometry at midpoint",
                        "forcing_note": "Instantaneous T, humidity, pressure and wind sampled at interval end",
                        "model_options": opts,
                        "ssrd_fill_method": "neighbour_plateau_v1",
                        "ssrd_fill_runner_sha256": payload["flux_module_sha256"],
                        "ssrd_fill_records": json.dumps(records),
                    },
                )
                temporary = work / "cell.nc"
                data.to_netcdf(
                    temporary,
                    encoding={n: {"zlib": True, "complevel": 4} for n in data.data_vars},
                )
                temporary.replace(out)
                shutil.rmtree(work)
                saved += 1
                elapsed_saved += time.monotonic() - started
                eta = (len(cells) - i) * elapsed_saved / saved / 60
                report(f"Saved {out.name} | {i}/{len(cells)} | estimated remaining {eta:.1f} min")

            except Exception as exc:
                failures += 1
                print(f"FAILED ({y},{x}): {exc}; retained work: {work}", flush=True)

        if failures:
            raise SystemExit(f"{failures} cells failed; successful results were preserved")
        report(f"Finished: {len(cells)} eligible cells considered")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
