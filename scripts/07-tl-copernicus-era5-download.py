from pathlib import Path
import calendar
import zipfile
import shutil
import argparse

import cdsapi
import xarray as xr

DATASET = "reanalysis-era5-land"
AREA = [53, -95, 48, -87]

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

def iter_months(start_year=2018, start_month=10, end_year=2025, end_month=9):
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield str(y), f"{m:02d}"
        m += 1
        if m == 13:
            y += 1
            m = 1

def download_month(client, out_dir, year, month, overwrite=False):
    extract_dir = out_dir / f"{year}_{month}"
    final_path = extract_dir / f"{year}_{month}.nc"

    if final_path.exists() and not overwrite:
        print(f"Skipping existing: {final_path}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)

    days = [f"{d:02d}" for d in range(1, calendar.monthrange(int(year), int(month))[1] + 1)]
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

        print(f"Downloading {year}-{month} {name}")
        client.retrieve(DATASET, request, str(zip_path))

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(part_dir)

        nc_files = list(part_dir.rglob("*.nc"))

        if len(nc_files) != 1:
            raise ValueError(f"{year}-{month} {name}: found {len(nc_files)} .nc files")

        if part_nc.exists():
            part_nc.unlink()

        nc_files[0].replace(part_nc)
        part_paths.append(part_nc)

        zip_path.unlink(missing_ok=True)
        shutil.rmtree(part_dir, ignore_errors=True)

    datasets = []
    for p in part_paths:
        with xr.open_dataset(p) as d:
            datasets.append(d.load())

    ds = xr.merge(datasets, compat="override", join="exact")

    ds["w10"] = (ds["u10"]**2 + ds["v10"]**2) ** 0.5
    ds["w10"].attrs = {
        "long_name": "10 m wind speed",
        "units": "m s**-1",
    }

    if final_path.exists():
        final_path.unlink()

    ds.to_netcdf(final_path)
    ds.close()

    for p in part_paths:
        p.unlink(missing_ok=True)

    print(f"Saved: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="../../scratch/ERA5",
        help="Output directory for monthly ERA5 files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly .nc files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    for year, month in iter_months():
        download_month(client, out_dir, year, month, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
