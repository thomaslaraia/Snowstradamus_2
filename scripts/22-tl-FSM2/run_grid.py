#!/usr/bin/env python3
"""Independent FSM2 columns from monthly ERA5-Land files. See GUIDE.md."""
import time
import threading
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import dask
import numpy as np
import pandas as pd
import xarray as xr

MET = ['t2m', 'd2m', 'sf', 'tp', 'ssrd', 'strd', 'u10', 'v10', 'sp']
PIN = 'a76945e75b36628161f4679428ed72dce090b368'


def report(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


@contextmanager
def stage(name):
    report(name)
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            report(f'{name} | {time.monotonic() - start:.0f}s elapsed')

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def normalized(ds):
    if 'valid_time' in ds.dims:
        ds = ds.rename(valid_time='time')
    for d in ('time', 'latitude', 'longitude'):
        if d not in ds.dims:
            raise ValueError(f'Missing dimension {d}')
    return ds


def open_region(root):
    files = sorted(root.glob('????_??/driving_data.nc'))
    if not files:
        raise ValueError(f'No YYYY_MM/driving_data.nc files under {root}')
    with xr.open_dataset(files[0]) as f:
        f = normalized(f)
        static = f[['VAI', 'vegh']].load()
        for name in ('VAI', 'vegh'):
            if set(static[name].dims) != {'latitude', 'longitude'}:
                raise ValueError(f'{name} must be static and two-dimensional')
    for index, path in enumerate(files, 1):
        report(f'Checking month {index}/{len(files)}: {path.parent.name}')
        with xr.open_dataset(path) as f:
            f = normalized(f)
            for coord in ('latitude', 'longitude'):
                np.testing.assert_array_equal(f[coord], static[coord])
            for name in ('VAI', 'vegh'):
                np.testing.assert_allclose(f[name], static[name], rtol=0, atol=0, equal_nan=True)
            for name in MET:
                if set(f[name].dims) != {'time', 'latitude', 'longitude'}:
                    raise ValueError(f'{path}: unexpected dimensions for {name}: {f[name].dims}')
    report(f'Opening hourly forcing across {len(files)} months')
    ds = xr.open_mfdataset(files, preprocess=lambda d: normalized(d)[MET],
                          combine='nested', concat_dim='time', join='exact',
                          data_vars='minimal', coords='minimal', compat='equals',
                          chunks={})
    t = pd.DatetimeIndex(ds.time.values)
    if len(t) < 2 or t.hasnans or not np.all(np.diff(t.asi8) == 3600 * 10**9):
        ds.close()
        raise ValueError('Time must be strictly increasing, gap-free and hourly; do not bridge missing months')
    if np.any(t.minute != 0) or np.any(t.second != 0):
        raise ValueError('Expected whole UTC hours')
    return files, ds, static


def increments(a, times, mode, label):
    a = np.asarray(a, dtype='float64')
    if mode == 'cumulative':
        # 01 UTC begins a new forecast; 00 UTC ends the previous forecast.
        delta = np.diff(a, prepend=np.nan)
        out = np.where(times.hour == 1, a, delta)
    else:
        out = a.copy()
    # Allow only tiny round-off negatives, never disguise broken de-accumulation.
    tolerance = {'sf': 1e-7, 'tp': 1e-7, 'ssrd': 10.0, 'strd': 1.0}[label]
    known = out[np.isfinite(out)]
    if len(known) and known.min() < -tolerance:
        raise ValueError(f'{label}: negative increment {known.min()}; inspect accumulation convention')
    return np.maximum(out, 0)



def fill_ssrd(point, ds, iy, ix, mode):
    """Fill short cumulative-SSRD gaps only when neighbours support a plateau."""
    records = []
    if mode != "cumulative":
        return point, records
    t = pd.DatetimeIndex(point.time.values)
    original = np.asarray(point.ssrd.values, dtype=float)
    missing = np.flatnonzero(~np.isfinite(original))
    if not len(missing):
        return point, records
    filled = original.copy()
    y0, x0 = max(0, iy - 1), max(0, ix - 1)
    for gap in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1):
        a, b = int(gap[0]), int(gap[-1]) + 1  # b is first observation after gap
        def reject(reason):
            raise ValueError(f"SSRD gap at {t[a]} UTC ({len(gap)} h): {reason}")
        if len(gap) > 4 or b >= len(t):
            reject("requires at most four missing hours and a following observation")
        reset = t[a].hour == 1
        if (not reset and a == 0) or any(t[k].hour == 1 for k in range(a+1, b+1)):
            reject("no usable same-cycle bounds")
        start = a if reset else a - 1
        endpoint = original[b]
        if endpoint < 0:
            reject("negative cumulative endpoint")
        if not reset and abs(original[a-1] - endpoint) > 2:
            reject("target bounds differ by more than 2 J/m2")
        block = ds.ssrd.isel(latitude=slice(y0, iy+2), longitude=slice(x0, ix+2),
                             time=slice(start, b+1)).transpose("time", "latitude", "longitude")
        values = np.asarray(block.values, dtype=float)
        surrounding = np.ones(values.shape[1:], dtype=bool)
        surrounding[iy-y0, ix-x0] = False
        neighbours = values[:, surrounding]
        neighbours = neighbours[:, np.isfinite(neighbours).all(axis=0)]
        if neighbours.shape[1] < 3:
            reject("fewer than three complete immediate neighbours")
        variation = float(np.max(np.ptp(neighbours, axis=0)))
        if variation > 3:
            reject(f"neighbour cumulative range {variation:g} exceeds 3 J/m2")
        if reset and not neighbours[0].min()-2 <= endpoint <= neighbours[0].max()+2:
            reject("target endpoint outside neighbouring first-hour range (+/-2 J/m2)")
        filled[gap] = endpoint
        records.append(dict(start_utc=str(t[a]), end_utc=str(t[b-1]), hours=len(gap),
                            source_utc=str(t[b]), cumulative_J_m2=float(endpoint),
                            complete_neighbours=neighbours.shape[1], max_range_J_m2=variation))
    point = point.copy(deep=False)
    point["ssrd"] = point.ssrd.copy(data=filled)
    return point, records


def forcing(point, mode):
    t = pd.DatetimeIndex(point.time.values)
    a = {v: np.asarray(point[v].values, dtype='float64') for v in MET}
    if any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError('Missing/nonfinite meteorology; cell has not been simulated')
    if np.any((a['t2m'] < 180) | (a['t2m'] > 340)) or np.any((a['d2m'] < 170) | (a['d2m'] > 340)):
        raise ValueError('Temperature outside broad Kelvin range; inspect units')
    if np.any((a['sp'] < 20000) | (a['sp'] > 110000)):
        raise ValueError('Pressure must be in Pa')
    inc = {v: increments(a[v], t, mode, v) for v in ('sf', 'tp', 'ssrd', 'strd')}
    keep = np.isfinite(inc['sf'])
    # Only the first record can lack its preceding accumulation.
    if not keep[1:].all():
        raise ValueError('Unexpected unavailable increment')
    rain = inc['tp'] - inc['sf']
    if np.nanmin(rain) < -1e-7:
        raise ValueError('Hourly snowfall exceeds total precipitation beyond tolerance')
    td = a['d2m'] - 273.15
    tc = a['t2m'] - 273.15
    e = 611.2 * np.exp(17.67 * td / (td + 243.5))
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    q = 0.622 * e / (a['sp'] - 0.378 * e)
    rh = 100 * e / es  # Informational column; DRIV1D=2 uses q directly.
    # Use interval midpoint for solar geometry; output is relabelled to interval end.
    mid = t - pd.Timedelta(minutes=30)
    matrix = np.column_stack([mid.year, mid.month, mid.day,
                              mid.hour + mid.minute / 60,
                              inc['ssrd'] / 3600, inc['strd'] / 3600,
                              np.maximum(rain, 0) * 1000 / 3600,
                              inc['sf'] * 1000 / 3600,
                              a['t2m'], q, rh, np.hypot(a['u10'], a['v10']), a['sp']])
    matrix = matrix[keep]
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite converted forcing')
    return t[keep], matrix


def write_met(path, matrix):
    np.savetxt(path, matrix, fmt=['%d', '%d', '%d', '%.3f'] + ['%.9e'] * 9)


def namelist(vai, height, lat, lon, restart):
    return f'''&params
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
'''


def execute(exe, work, nlst, count):
    (work / 'namelist.txt').write_text(nlst)
    for name in ('stat.txt', 'flux.txt', 'subc.txt', 'dump.txt'):
        (work / name).unlink(missing_ok=True)
    with (work / 'namelist.txt').open() as f, (work / 'model.log').open('w') as log:
        start = time.monotonic()
        position = completed = 0
        with subprocess.Popen([str(exe)], stdin=f, stdout=log, stderr=subprocess.STDOUT, cwd=work) as proc:
            while True:
                try:
                    status = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    stats = work / 'stat.txt'
                    if stats.exists():
                        with stats.open('rb') as rows:
                            rows.seek(position)
                            completed += rows.read().count(b'\n')
                            position = rows.tell()
                    elapsed = time.monotonic() - start
                    eta = f'{elapsed * max(0, count - completed) / completed:.0f}s' if completed else 'pending'
                    report(f'FSM2: {completed}/{count} hourly rows written, pass ETA {eta}')
            if status:
                raise subprocess.CalledProcessError(status, str(exe))
    result = np.loadtxt(work / 'stat.txt', ndmin=2)
    if result.shape != (count, 13) or not np.isfinite(result).all():
        raise ValueError(f'Unexpected/nonfinite model output: {result.shape}; expected {(count, 13)}')
    if np.any(result[:, 4:7] < -1e-6):
        raise ValueError('Negative snow state in forest cell')
    if not (work / 'dump.txt').is_file():
        raise ValueError('Model did not write its restart state')
    return result


def signature(files, exe, args):
    payload = {'inputs': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
               'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'accumulation': args.accumulation, 'spinup_cycles': args.spinup_cycles,
               'minimum_height': 2.0, 'source_commit': PIN}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Region folder, e.g. /scratch/.../fsm2/S01')
    p.add_argument('--output', type=Path, required=True, help='New experiment output directory')
    p.add_argument('--exe', type=Path, default=Path(__file__).parent / 'FSM2_source/FSM2')
    p.add_argument('--accumulation', choices=['cumulative', 'hourly'], required=True)
    p.add_argument('--spinup-cycles', type=int, default=2)
    p.add_argument('--row-start', type=int, default=0)
    p.add_argument('--row-stop', type=int, default=None)
    p.add_argument('--limit', type=int, default=None, help='Number of eligible cells to attempt')
    args = p.parse_args()
    if args.spinup_cycles < 0 or (args.limit is not None and args.limit < 1):
        p.error('Use nonnegative spinup-cycles and positive limit')
    exe = args.exe.resolve()
    if not exe.is_file():
        p.error(f'Missing executable: {exe}; first run build_fsm2.sh')
    opts = exe.parent / 'src/OPTS.h'
    expected = {'DRIV1D': 2, 'SWPART': 1, 'ZOFFST': 1, 'PROFNC': 0, 'CANMOD': 1}
    options = dict((parts[1], int(parts[2])) for line in opts.read_text().splitlines()
                   if (parts := line.split()) and parts[0] == '#define')
    if any(options.get(k) != v for k, v in expected.items()):
        p.error('Executable source options differ from this starter configuration; rebuild')
    with stage('Preparing regional input'):
        files, ds, static = open_region(args.input.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    repair_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sig = signature(files, exe, args)
    stop = ds.sizes['latitude'] if args.row_stop is None else min(args.row_stop, ds.sizes['latitude'])
    if not 0 <= args.row_start < stop:
        p.error('Empty/invalid row range')
    region = static.isel(latitude=slice(args.row_start, stop))
    eligible = np.isfinite(region.VAI) & np.isfinite(region.vegh) & (region.VAI > 0) & (region.vegh > 2)
    total = int(eligible.sum())
    if args.limit is not None:
        total = min(total, args.limit)
    report(f'{total} eligible cells selected')
    newly_saved = 0
    compute_seconds = 0.
    done, failed = 0, []
    try:
        for iy in range(args.row_start, stop):
            for ix in range(ds.sizes['longitude']):
                vai = float(static.VAI.isel(latitude=iy, longitude=ix))
                h = float(static.vegh.isel(latitude=iy, longitude=ix))
                # hbas=2 m in this baseline. Low vegetation needs its own parameterization.
                if not (np.isfinite(vai) and np.isfinite(h) and vai > 0 and h > 2):
                    continue
                out = args.output / f'cell_y{iy:04d}_x{ix:04d}.nc'
                report(f'Cell {done + 1}/{total}: {out.name}')
                if out.exists():
                    with xr.open_dataset(out) as previous:
                        if previous.attrs.get('run_signature') != sig:
                            raise ValueError(f'{out}: configuration/input changed; use a new output directory')
                    print('Already complete:', out.name, flush=True)
                else:
                    work = Path(tempfile.mkdtemp(prefix=f'work_y{iy}_x{ix}_', dir=args.output))
                    try:
                        cell_start = time.monotonic()
                        with stage('Loading hourly weather for cell'), dask.config.set(scheduler='synchronous'):
                            point = ds.isel(latitude=iy, longitude=ix).load()
                        point, fills = fill_ssrd(point, ds, iy, ix, args.accumulation)
                        if fills:
                            report('SSRD fills: ' + json.dumps(fills))
                        times, matrix = forcing(point, args.accumulation)
                        lat, lon = float(point.latitude), float(point.longitude)
                        restart = 'none'
                        if args.spinup_cycles:
                            anniversary = times[0] + pd.DateOffset(years=1)
                            if times[-1] < anniversary - pd.Timedelta(hours=1):
                                raise ValueError('Spin-up requires at least 12 calendar months; use --spinup-cycles 0 only for a smoke test')
                            spin = matrix[times < anniversary]
                            for cycle in range(args.spinup_cycles):
                                report(f'Spin-up {cycle + 1}/{args.spinup_cycles}: {len(spin)} hours')
                                write_met(work / 'met.txt', spin)
                                execute(exe, work, namelist(vai, h, lat, lon, restart), len(spin))
                                shutil.copy2(work / 'dump.txt', work / 'start.txt')
                                restart = 'start.txt'
                        report(f'Production run: {len(matrix)} hours')
                        write_met(work / 'met.txt', matrix)
                        result = execute(exe, work, namelist(vai, h, lat, lon, restart), len(matrix))
                        # Model writes midpoint labels; validate them before assigning interval-end times.
                        expected_dates = matrix[:, :4]
                        np.testing.assert_allclose(result[:, :4], expected_dates, rtol=0, atol=0.001)
                        data = xr.Dataset({
                            'snow_depth': ('time', result[:, 4].astype('float32'), {'units': 'm'}),
                            'ground_snow_mass': ('time', result[:, 5].astype('float32'), {'units': 'kg m-2'}),
                            'canopy_snow_mass': ('time', result[:, 6].astype('float32'),
                                                 {'units': 'kg m-2', 'long_name': 'Total snow mass on vegetation per model ground area'})},
                            coords={'time': times, 'latitude': lat, 'longitude': lon},
                            attrs={'run_signature': sig, 'source_commit': PIN, 'VAI': vai, 'vegh_m': h,
                                   'spinup_cycles': args.spinup_cycles, 'accumulation': args.accumulation,
                                   'time_convention': 'UTC interval end; state after one-hour step; solar geometry at midpoint',
                                   'forcing_note': 'Instantaneous T, humidity, pressure and wind sampled at interval end',
                                   'model_options': opts.read_text()})
                        data.attrs['ssrd_fill_runner_sha256'] = repair_hash
                        data.attrs['ssrd_fill_method'] = 'neighbour_plateau_v1'
                        data.attrs['ssrd_fill_records'] = json.dumps(fills)
                        tmp = out.with_suffix('.partial.nc')
                        data.to_netcdf(tmp, encoding={n: {'zlib': True, 'complevel': 4} for n in data.data_vars})
                        tmp.replace(out)
                        shutil.rmtree(work)
                        newly_saved += 1
                        compute_seconds += time.monotonic() - cell_start
                        eta = (total - done - 1) * compute_seconds / newly_saved / 60
                        report(f'Saved {out.name} | {done + 1}/{total} considered | estimated remaining {eta:.1f} min')
                    except Exception as err:
                        failed.append((iy, ix, str(err)))
                        print(f'FAILED ({iy},{ix}): {err}; retained work: {work}', flush=True)
                done += 1
                if args.limit is not None and done >= args.limit:
                    break
            if args.limit is not None and done >= args.limit:
                break
    finally:
        ds.close()
    if failed:
        raise SystemExit(f'{len(failed)} cells failed; read the log. Successful cells are preserved.')
    print(f'Finished: {done} eligible cells considered', flush=True)


if __name__ == '__main__':
    main()
