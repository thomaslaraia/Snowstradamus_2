#!/usr/bin/env python3

from pathlib import Path
import argparse
import calendar
import gc
import os
import resource
import shutil
import zipfile

import cdsapi
import dask
import xarray as xr
from dask.callbacks import Callback


DATASET = "reanalysis-era5-land"

# CDS area order is [north, west, south, east].
AREA = [85, -180, 25, 180]

VARIABLE_GROUPS = {
    "temp": [
        "2m_dewpoint_temperature",
        "2m_temperature",
    ],
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

# Each data chunk is only around 10 MiB for a float32 variable. Even with
# several variables and workers active together, this stays far below 32 GiB.
CHUNK_SIZES = {
    "valid_time": 24,
    "time": 24,
    "latitude": 200,
    "longitude": 600,
}


class MemoryCallback(Callback):
    """Report memory periodically while Dask evaluates the write graph."""

    def __init__(self, report_every=250):
        super().__init__()
        self.report_every = report_every
        self.task_count = 0

    def _pretask(self, key, dask_graph, state):
        self.task_count += 1

        if self.task_count == 1 or self.task_count % self.report_every == 0:
            report_memory(f"Dask task {self.task_count}")


def gibibytes(number_of_bytes):
    return number_of_bytes / 1024**3


def current_rss_bytes():
    """Return this process's current resident memory on Linux."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_rss_bytes():
    """Return this process's peak resident memory on Linux."""
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def cgroup_memory_bytes():
    """Return current and maximum cgroup memory where available."""
    candidates = [
        (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory.max"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    ]

    for current_path, maximum_path in candidates:
        try:
            current_text = current_path.read_text(encoding="utf-8").strip()
            maximum_text = maximum_path.read_text(encoding="utf-8").strip()
            current = int(current_text)
            maximum = None if maximum_text == "max" else int(maximum_text)
            return current, maximum
        except (OSError, ValueError):
            continue

    return None, None


def report_memory(stage):
    """Print process and job/cgroup memory information immediately."""
    current_rss = current_rss_bytes()
    peak_rss = peak_rss_bytes()
    cgroup_current, cgroup_maximum = cgroup_memory_bytes()

    fields = [f"[memory] {stage}"]

    if current_rss is not None:
        fields.append(f"process RSS={gibibytes(current_rss):.2f} GiB")

    fields.append(f"process peak={gibibytes(peak_rss):.2f} GiB")

    if cgroup_current is not None:
        fields.append(f"cgroup current={gibibytes(cgroup_current):.2f} GiB")

    if cgroup_maximum is not None:
        fields.append(f"cgroup limit={gibibytes(cgroup_maximum):.2f} GiB")

    print(" | ".join(fields), flush=True)


def report_slurm_memory_request():
    job_id = os.environ.get("SLURM_JOB_ID")
    memory_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    memory_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")

    if job_id is None:
        print("Not running inside a detected Slurm job", flush=True)
        return

    print(f"Slurm job ID: {job_id}", flush=True)

    if memory_per_node is not None:
        print(f"Slurm memory per node: {memory_per_node} MiB", flush=True)

    if memory_per_cpu is not None:
        print(f"Slurm memory per CPU: {memory_per_cpu} MiB", flush=True)


def report_dataset(dataset, label, disk_path):
    """Print disk size and the uncompressed in-memory size from metadata."""
    disk_size = disk_path.stat().st_size

    print(f"Dataset: {label}", flush=True)
    print(f"  Path: {disk_path}", flush=True)
    print(f"  Disk size: {gibibytes(disk_size):.2f} GiB", flush=True)
    print(
        f"  Logical uncompressed size: {gibibytes(dataset.nbytes):.2f} GiB",
        flush=True,
    )
    print(f"  Dimensions: {dict(dataset.sizes)}", flush=True)

    for variable_name, variable in dataset.data_vars.items():
        if variable.chunks is None:
            chunk_text = "not chunked"
        else:
            largest_chunks = tuple(max(axis_chunks) for axis_chunks in variable.chunks)
            chunk_text = f"largest chunk={largest_chunks}"

        print(
            f"  {variable_name}: dtype={variable.dtype}, "
            f"shape={variable.shape}, size={gibibytes(variable.nbytes):.2f} GiB, "
            f"{chunk_text}",
            flush=True,
        )


def chunk_specification(dataset):
    """Return chunk sizes only for dimensions present in this dataset."""
    return {
        dimension: min(CHUNK_SIZES[dimension], size)
        for dimension, size in dataset.sizes.items()
        if dimension in CHUNK_SIZES
    }


def netcdf_encoding(dataset):
    """Use compressed, reasonably sized chunks in the combined output."""
    encoding = {}

    for variable_name, variable in dataset.data_vars.items():
        if variable.ndim == 0:
            encoding[variable_name] = {}
            continue

        chunksizes = tuple(
            min(CHUNK_SIZES.get(dimension, variable.sizes[dimension]),
                variable.sizes[dimension])
            for dimension in variable.dims
        )

        encoding[variable_name] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": chunksizes,
        }

    return encoding


def iter_months(start_year=2018, start_month=10, end_year=2025, end_month=9):
    y, m = start_year, start_month

    while (y < end_year) or (y == end_year and m <= end_month):
        yield str(y), f"{m:02d}"
        m += 1

        if m == 13:
            y += 1
            m = 1


def download_month(
    client,
    out_dir,
    year,
    month,
    overwrite=False,
    workers=2,
    memory_log_every=250,
):
    extract_dir = out_dir / f"{year}_{month}"
    final_path = extract_dir / f"{year}_{month}.nc"

    print(f"\n{'=' * 72}", flush=True)
    print(f"Starting {year}-{month}", flush=True)
    report_memory(f"start of {year}-{month}")

    if final_path.exists() and not overwrite:
        print(f"Skipping existing: {final_path}", flush=True)
        return

    extract_dir.mkdir(parents=True, exist_ok=True)

    number_of_days = calendar.monthrange(int(year), int(month))[1]
    days = [f"{d:02d}" for d in range(1, number_of_days + 1)]
    times = [f"{h:02d}:00" for h in range(24)]

    base_request = {
        "year": year,
        "month": month,
        "day": days,
        "time": times,
        "data_format": "netcdf",
        "download_format": "zip",
        "area": AREA,
    }

    part_paths = []

    for name, variables in VARIABLE_GROUPS.items():
        zip_path = extract_dir / f"{year}_{month}_{name}.zip"
        part_dir = extract_dir / name
        part_nc = extract_dir / f"{year}_{month}_{name}.nc"

        shutil.rmtree(part_dir, ignore_errors=True)
        part_dir.mkdir(parents=True, exist_ok=True)

        request = base_request.copy()
        request["variable"] = variables

        report_memory(f"before downloading {year}-{month} {name}")
        print(f"Downloading {year}-{month} {name}", flush=True)
        client.retrieve(DATASET, request, str(zip_path))
        print(
            f"Downloaded ZIP: {zip_path} "
            f"({gibibytes(zip_path.stat().st_size):.2f} GiB)",
            flush=True,
        )
        report_memory(f"after downloading {year}-{month} {name}")

        print(f"Extracting {zip_path.name}", flush=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(part_dir)
        report_memory(f"after extracting {year}-{month} {name}")

        nc_files = list(part_dir.rglob("*.nc"))

        if len(nc_files) != 1:
            raise ValueError(
                f"{year}-{month} {name}: found {len(nc_files)} .nc files"
            )

        if part_nc.exists():
            part_nc.unlink()

        nc_files[0].replace(part_nc)
        part_paths.append(part_nc)

        print(
            f"Prepared part file: {part_nc} "
            f"({gibibytes(part_nc.stat().st_size):.2f} GiB on disk)",
            flush=True,
        )

        zip_path.unlink(missing_ok=True)
        shutil.rmtree(part_dir, ignore_errors=True)
        report_memory(f"after cleaning download files for {year}-{month} {name}")

    datasets = []
    combined = None

    try:
        for part_path in part_paths:
            report_memory(f"before opening {part_path.name}")

            # Open once without chunks to inspect the dimensions. This reads
            # metadata only and is closed before the lazy dataset is opened.
            with xr.open_dataset(part_path) as metadata:
                chunks = chunk_specification(metadata)

            source = xr.open_dataset(part_path, chunks=chunks)
            datasets.append(source)

            report_dataset(source, part_path.stem, part_path)
            print(f"  Dask chunk specification: {chunks}", flush=True)
            report_memory(f"after lazily opening {part_path.name}")

        report_memory("before xr.merge")
        print("Merging lazy datasets", flush=True)
        combined = xr.merge(datasets, compat="override", join="exact")
        report_memory("after xr.merge")

        print("Calculating 10 m wind speed", flush=True)
        report_memory("before calculating w10")
        combined["w10"] = (
            combined["u10"] ** 2 + combined["v10"] ** 2
        ) ** 0.5
        combined["w10"].attrs = {
            "long_name": "10 m wind speed",
            "units": "m s**-1",
        }
        report_memory("after calculating w10")

        if final_path.exists():
            final_path.unlink()

        print(f"Writing combined file: {final_path}", flush=True)
        report_memory("immediately before to_netcdf")

        print(
            f"Computing with {workers} Dask worker(s); reporting memory every "
            f"{memory_log_every} tasks",
            flush=True,
        )

        with dask.config.set(scheduler="threads", num_workers=workers):
            with MemoryCallback(report_every=memory_log_every):
                combined.to_netcdf(
                    final_path,
                    engine="netcdf4",
                    encoding=netcdf_encoding(combined),
                )

        report_memory("immediately after to_netcdf")

    finally:
        print("Closing and releasing xarray datasets", flush=True)

        if combined is not None:
            combined.close()
        combined = None

        for index in range(len(datasets)):
            datasets[index].close()
        datasets.clear()

        gc.collect()
        report_memory(f"after dataset cleanup for {year}-{month}")

    for part_path in part_paths:
        part_path.unlink(missing_ok=True)

    print(f"Saved: {final_path}", flush=True)
    report_memory(f"completed {year}-{month}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="../../data_store/ERA5",
        help="Output directory for monthly ERA5 files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly .nc files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of Dask worker threads. By default, use "
            "SLURM_CPUS_PER_TASK or 2 outside Slurm"
        ),
    )
    parser.add_argument(
        "--memory-log-every",
        type=int,
        default=250,
        help="Report memory after this many Dask tasks (default: 250)",
    )
    args = parser.parse_args()

    if args.workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    else:
        workers = args.workers

    if workers < 1:
        parser.error("--workers must be at least 1")

    if args.memory_log_every < 1:
        parser.error("--memory-log-every must be at least 1")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ERA5-Land area [north, west, south, east]: {AREA}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Dask workers: {workers}", flush=True)
    print(f"Target chunk sizes: {CHUNK_SIZES}", flush=True)
    report_slurm_memory_request()
    report_memory("program start")

    client = cdsapi.Client()

    for year, month in iter_months():
        download_month(
            client,
            out_dir,
            year,
            month,
            overwrite=args.overwrite,
            workers=workers,
            memory_log_every=args.memory_log_every,
        )

    report_memory("program complete")


if __name__ == "__main__":
    main()
