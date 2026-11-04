#!/usr/bin/env python3
"""Download ERA5-Land daily snowfall and derive daily hourly-wind p95.

Install: python -m pip install cdsapi numpy pandas xarray netCDF4
Configure CDS credentials: https://cds.climate.copernicus.eu/how-to-api
Accept the ERA5-Land licence on the CDS dataset page before running.

Example:
 python download_era5_land_daily.py --output ../../scratch/era5_land_daily
Small test:
 python download_era5_land_daily.py --start-year 2018 --end-year 2018 \
     --months 1 --area 57 -4 56 -3 --output ./era5_small_test

Defaults: all months 2018-2023, area [73, -180, 45, 180], native 0.1 deg.
Dates are UTC. Snowfall is mm water equivalent, NOT snow depth.
One compressed NetCDF per month; completed months are skipped on rerun.
Temporary downloads are deleted ONLY after the monthly output is validated
and atomically installed. Failed/interrupted months retain completed downloads.
Run only one process per output directory. Paths are relative to the working
 directory unless absolute. No data from existing climatologies are modified.

Snowfall: original ERA5-Land 00 UTC accumulation represents previous UTC day.
https://confluence.ecmwf.int/spaces/CKB/pages/462888259
Wind: percentile(hypot(u10, v10), 95) over hours 00..23, linear interpolation.
All 24 values must be finite at a pixel, otherwise the daily result is missing.
This is a percentile of hourly wind samples, not a gust estimate.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import warnings
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

DATASET = 'reanalysis-era5-land'
VERSION = '1'


def daily_wind_p95(u, v):
    """First dimension must contain the 24 ordered hourly samples."""
    if u.shape != v.shape or u.shape[0] != 24:
        raise ValueError('Expected matching u/v arrays with exactly 24 hours')
    speed = np.hypot(u, v)
    valid = np.isfinite(speed).all(axis=0)
    speed = np.where(np.isfinite(speed), speed, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = np.percentile(speed, 95, axis=0, method='linear')
    return np.where(valid, result, np.nan).astype('float32')


def download(client, request, target):
    if target.exists():
        print(f'  Reusing {target.name}', flush=True)
        return
    partial = target.with_suffix('.download')
    partial.unlink(missing_ok=True)
    client.retrieve(DATASET, request).download(str(partial))
    partial.replace(target)


def open_download(stack, path):
    # CDS can return a ZIP even with download_format=unarchived.
    if zipfile.is_zipfile(path):
        extracted = path.with_suffix('.extracted')
        extracted.mkdir(exist_ok=True)
        paths = []
        with zipfile.ZipFile(path) as archive:
            for i, info in enumerate(archive.infolist()):
                if info.filename.endswith('.nc'):
                    dest = extracted / f'{i}.nc'
                    with archive.open(info) as src, dest.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    paths.append(dest)
        if not paths:
            raise ValueError(f'No NetCDF files in {path}')
    else:
        paths = [path]
    datasets = []
    for p in paths:
        ds = stack.enter_context(xr.open_dataset(p))
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        if 'time' not in ds.dims:
            raise ValueError(f'Unexpected time dimensions in {p}: {ds.dims}')
        datasets.append(ds)
    return xr.merge(datasets, compat='no_conflicts', join='exact')


def variable(ds, *names):
    for name in names:
        if name in ds:
            da = ds[name]
            for dim in list(da.dims):
                if dim not in ('time', 'latitude', 'longitude'):
                    if da.sizes[dim] != 1:
                        raise ValueError(f'Unexpected non-singleton dimension: {dim}')
                    da = da.isel({dim: 0}, drop=True)
            return da.transpose('time', 'latitude', 'longitude')
    raise KeyError(f'Expected {names}; found {list(ds.data_vars)}')


def require_times(da, expected):
    actual = pd.DatetimeIndex(da.time.values)
    if not actual.equals(pd.DatetimeIndex(expected)):
        raise ValueError('Missing, duplicate, unordered, or unexpected timestamps')


def validate(path, dates, signature):
    with xr.open_dataset(path) as ds:
        if ds.attrs.get('processing_signature') != signature:
            raise ValueError(f'{path}: settings differ; use another output directory')
        require_times(ds, dates)
        if ds.attrs.get('complete') != 'yes':
            raise ValueError(f'{path}: output is incomplete')
        for name in ('snowfall', 'wind_speed_p95'):
            if ds[name].dims != ('time', 'latitude', 'longitude'):
                raise ValueError(f'{path}: incorrect dimensions')
            # Read all compressed chunks before allowing source deletion.
            for i in range(len(dates)):
                ds[name].isel(time=i).load()


def process_month(client, year, month, area, root, signature):
    import netCDF4
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    dates = pd.date_range(start, end, inclusive='left', freq='D')
    label = start.strftime('%Y%m')
    final = root / f'era5_land_daily_{label}.nc'
    work = root / '_temporary' / label
    if final.exists():
        validate(final, dates, signature)
        if work.exists():
            shutil.rmtree(work)
        print(f'Skip completed {label}', flush=True)
        return
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / 'processing_signature.txt'
    if manifest.exists() and manifest.read_text() != signature:
        raise ValueError('Temporary downloads use different settings; use another output directory')
    manifest.write_text(signature)
    print(f'Processing {label}', flush=True)
    common = dict(area=area, data_format='netcdf', download_format='unarchived')
    # Days 2..end-of-month plus next month's first midnight give this month's totals.
    snow_paths = []
    snow_times = dates + pd.Timedelta(days=1)
    for period in snow_times.to_period('M').unique():
        subset = snow_times[snow_times.to_period('M') == period]
        path = work / f'snow_{period}.nc'
        request = dict(common, variable=['snowfall'], year=str(period.year),
                       month=f'{period.month:02d}', day=[f'{d.day:02d}' for d in subset],
                       time=['00:00'])
        download(client, request, path)
        snow_paths.append(path)
    wind_path = work / 'wind.nc'
    download(client, dict(common, variable=['10m_u_component_of_wind',
                        '10m_v_component_of_wind'], year=str(year), month=f'{month:02d}',
                        day=[f'{d.day:02d}' for d in dates],
                        time=[f'{h:02d}:00' for h in range(24)]), wind_path)
    temporary_output = root / f'.era5_land_daily_{label}.partial.nc'
    with ExitStack() as stack:
        snow = xr.concat([variable(open_download(stack, p), 'sf', 'snowfall')
                          for p in snow_paths], dim='time', join='exact').sortby('time')
        require_times(snow, snow_times)
        wind = open_download(stack, wind_path)
        u = variable(wind, 'u10', '10m_u_component_of_wind')
        v = variable(wind, 'v10', '10m_v_component_of_wind')
        hours = pd.date_range(start, end, inclusive='left', freq='h')
        require_times(u, hours)
        require_times(v, hours)
        for dim in ('latitude', 'longitude'):
            if not np.array_equal(snow[dim], u[dim]) or not np.array_equal(u[dim], v[dim]):
                raise ValueError(f'Snowfall/wind {dim} grids differ')
        with netCDF4.Dataset(temporary_output, 'w', format='NETCDF4') as out:
            out.setncatts(dict(Conventions='CF-1.8', source=DATASET,
                              processing_signature=signature, complete='no',
                              description='UTC daily snowfall total and p95 of 24 hourly wind speeds',
                              wind_percentile_method='linear; requires 24 finite hourly samples',
                              snowfall_method='next-day 00 UTC accumulation multiplied by 1000'))
            for dim, size in [('time', len(dates)), ('latitude', u.sizes['latitude']),
                              ('longitude', u.sizes['longitude'])]:
                out.createDimension(dim, size)
                coord = out.createVariable(dim, 'f8', (dim,))
                if dim == 'time':
                    coord.units = 'days since 1970-01-01 00:00:00'
                    coord.calendar = 'proleptic_gregorian'
                    coord.standard_name = 'time'
                    coord[:] = (dates - pd.Timestamp('1970-01-01')).days
                else:
                    coord[:] = u[dim].values
                    coord.standard_name = dim
                    coord.units = 'degrees_north' if dim == 'latitude' else 'degrees_east'
            fields = {}
            for name, units, long_name in [
                ('snowfall', 'mm', 'Daily snowfall total as liquid water equivalent'),
                ('wind_speed_p95', 'm s-1', 'Daily 95th percentile of hourly 10 metre wind speed')]:
                field = out.createVariable(name, 'f4', ('time', 'latitude', 'longitude'),
                    zlib=True, complevel=4, shuffle=True, fill_value=np.float32(np.nan),
                    chunksizes=(1, min(128, u.sizes['latitude']), min(256, u.sizes['longitude'])))
                field.units, field.long_name = units, long_name
                fields[name] = field
            for i, date in enumerate(dates):
                fields['snowfall'][i] = (snow.isel(time=i).values * 1000).astype('float32')
                selection = slice(i * 24, (i + 1) * 24)
                fields['wind_speed_p95'][i] = daily_wind_p95(
                    u.isel(time=selection).values, v.isel(time=selection).values)
                out.sync()
                print(f'  Saved {date.date()}', flush=True)
            out.complete = 'yes'
    validate(temporary_output, dates, signature)
    temporary_output.replace(final)
    shutil.rmtree(work)
    print(f'Finished {final}; temporary downloads removed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2023)
    parser.add_argument('--months', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--area', type=float, nargs=4, default=[73, -180, 45, 180],
                        metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'))
    parser.add_argument('--output', type=Path, default=Path('era5_land_daily'))
    args = parser.parse_args()
    n, w, s, e = args.area
    if not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        parser.error('Area must be NORTH WEST SOUTH EAST, without dateline crossing')
    if args.start_year > args.end_year or any(m < 1 or m > 12 for m in args.months):
        parser.error('Invalid year range or month')
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(dict(area=args.area, version=VERSION,
                                dataset=DATASET), sort_keys=True).encode()).hexdigest()
    import cdsapi
    client = cdsapi.Client()
    print(f'Output directory: {args.output}', flush=True)
    for year in range(args.start_year, args.end_year + 1):
        for month in sorted(set(args.months)):
            process_month(client, year, month, args.area, args.output, signature)


if __name__ == '__main__':
    main()
