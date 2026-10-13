#!/usr/bin/env python3
"""Summarise ERA5-Land and 2020 NASA ABoVE cover in a chosen area.

Install: pip install cdsapi xarray netCDF4 numpy rasterio earthaccess
CDS credentials: ~/.cdsapirc; NASA credentials: earthaccess login or ~/.netrc.

Example (three complete winters; change --area to your region):
  python seasonal_snow_site_climate.py --area 73 -180 45 180 \\
      --start-year 2019 --end-year 2021 --keep-tcc

For a single inclusive month range instead of OND/DJF/FMA, add for example
--months October April. This produces one season labelled OCT_APR. Winter-year
2019 then covers October 2018 through April 2019.

The default directory for downloads and output is
../../scratch/climate_site_selection/. Native ABoVE tiles already in its tcc/
folder are checked first and reused. Missing tiles are fetched automatically;
--keep-tcc saves those tiles in tcc/ for later runs. Without it, new tiles are
temporary.
Each tile is added to a copy of the partial NetCDF, which replaces the old
partial file only after the write finishes. Completed tile summaries are kept
until the final NetCDF is saved.
By default, season-year Y uses Oct-Dec Y-1 (OND), Dec Y-1-Feb Y (DJF), and
Feb-Apr Y (FMA). December and February intentionally occur twice.
Daily snowfall and precipitation use 00 UTC of the following day; shared
calendar-month requests contain only the timestamps required by the selected
periods.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter
from contextlib import ExitStack
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
from time import perf_counter
from urllib.parse import unquote, urlsplit
import warnings
import zipfile

import numpy as np
import xarray as xr

SEASON_MONTHS = {
    "OND": ((-1, 10), (-1, 11), (-1, 12)),
    "DJF": ((-1, 12), (0, 1), (0, 2)),
    "FMA": ((0, 2), (0, 3), (0, 4)),
}
INSTANT = ["2m_dewpoint_temperature", "2m_temperature",
           "10m_u_component_of_wind", "10m_v_component_of_wind"]
ALIASES = {"t2m": ("t2m", "2m_temperature"),
           "d2m": ("d2m", "2m_dewpoint_temperature"),
           "u10": ("u10", "10m_u_component_of_wind"),
           "v10": ("v10", "10m_v_component_of_wind"),
           "sf": ("sf", "snowfall"),
           "tp": ("tp", "total_precipitation")}
TCC_DOI = "10.3334/ORNLDAAC/2012"
TCC_RESOLUTION = 0.00025
MONTH_NUMBERS = {calendar.month_abbr[m].lower(): m for m in range(1, 13)}


def parse_month(value):
    if value.isdecimal() and 1 <= int(value) <= 12:
        return int(value)
    month = value.lower()
    if month in MONTH_NUMBERS:
        return MONTH_NUMBERS[month]
    names = {calendar.month_name[m].lower(): m for m in range(1, 13)}
    if month in names:
        return names[month]
    raise argparse.ArgumentTypeError(f"Invalid month {value!r}; use 1–12 or a month name")


def season_months(season, year):
    if season in SEASON_MONTHS:
        return [(year + offset, month) for offset, month in SEASON_MONTHS[season]]
    start_name, end_name = season.lower().split("_")
    start, end = MONTH_NUMBERS[start_name], MONTH_NUMBERS[end_name]
    # Winter-year Y has October–December in Y-1 and January–September in Y.
    end_year = year - (end >= 10)
    start_year = end_year - (start > end)
    months = []
    yy, mm = start_year, start
    while (yy, mm) <= (end_year, end):
        months.append((yy, mm))
        yy += mm == 12
        mm = mm % 12 + 1
    return months


def date_blocks(season, year, block_days):
    for yy, mm in season_months(season, year):
        ndays = calendar.monthrange(yy, mm)[1]
        for first in range(1, ndays + 1, block_days):
            last = min(ndays, first + block_days - 1)
            yield date(yy, mm, first), date(yy, mm, last)


def valid_response(path):
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            magic = f.read(8)
        return (zipfile.is_zipfile(path) or magic.startswith(b"CDF")
                or magic.startswith(b"\x89HDF"))
    except OSError:
        return False


def retrieve(client, variables, dates, times, area, path):
    """Retrieve one request; retain downloads on failure for a retry."""
    if path.exists():
        if valid_response(path):
            print(f"Reusing {path}", flush=True)
            return 0.0
        path.unlink()
    first = dates[0]
    assert all((d.year, d.month) == (first.year, first.month) for d in dates)
    request = {"variable": variables, "year": str(first.year),
               "month": f"{first.month:02d}",
               "day": [f"{d.day:02d}" for d in dates],
               "time": times, "data_format": "netcdf",
               "download_format": "zip", "area": area}
    part = path.with_name(path.name + ".partial")
    part.unlink(missing_ok=True)
    print(f"CDS {first.year}-{first.month:02d}: {len(dates)} days, "
          f"{len(times)} hours, {len(variables)} variables", flush=True)
    started = perf_counter()
    client.retrieve("reanalysis-era5-land", request, str(part))
    part.replace(path)
    seconds = perf_counter() - started
    print(f"[timing] CDS {path.name}: {seconds:.1f} s", flush=True)
    return seconds


def precipitation_request(work_dir, download_key, index, stamps, needed_dates):
    """Share 00 UTC dates by calendar month, honouring the old ZIP cache."""
    legacy = work_dir / f"{download_key}_precip{index}.zip"
    if valid_response(legacy):
        return legacy, stamps, False

    year, month = stamps[0].year, stamps[0].month
    path = work_dir / f"{year}_{month:02d}_precip_00utc.zip"
    return path, needed_dates[(year, month)], True


def precipitation_dates_by_month(sequence):
    """All 00 UTC stamps needed for the requested days, without duplicates."""
    monthly = {}
    for _, _, _, _, first, last, _ in sequence:
        for offset in range((last - first).days + 1):
            stamp = first + timedelta(days=offset + 1)
            monthly.setdefault((stamp.year, stamp.month), set()).add(stamp)
    return {year_month: sorted(stamps)
            for year_month, stamps in monthly.items()}


def open_response(path, stack):
    """Open all NetCDF members; CDS sometimes splits variables across files."""
    if zipfile.is_zipfile(path):
        folder = path.with_suffix("")
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist()
                       if m.filename.lower().endswith(".nc")]
            if not members:
                raise ValueError(f"No NetCDF inside {path}")
            paths = []
            for i, member in enumerate(members):
                dest = folder / f"{i}.nc"  # never trust paths within ZIP
                with archive.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                paths.append(dest)
    else:
        paths = [path]
    sources = [stack.enter_context(xr.open_dataset(p, engine="netcdf4"))
               for p in paths]
    return xr.merge(sources, join="exact", compat="no_conflicts")


def field(dataset, name, expected_times, lat, lon):
    candidates = ALIASES[name]
    found = next((dataset[n] for n in candidates if n in dataset), None)
    if found is None:
        raise KeyError(f"Missing {name} in {list(dataset.data_vars)}")
    time_dim = next((d for d in ("valid_time", "time") if d in found.dims), None)
    if time_dim is None:
        raise ValueError(f"No valid_time/time dimension for {name}")
    found = found.transpose(time_dim, "latitude", "longitude")
    actual = np.asarray(found[time_dim].values).astype("datetime64[h]")
    expected = np.array(expected_times, dtype="datetime64[h]")
    if not np.array_equal(actual, expected):
        raise ValueError(f"Unexpected timestamps for {name}: {actual[:3]} ... "
                         f"expected {expected[:3]} ...")
    if not np.array_equal(found.latitude.values, lat) or not np.array_equal(
            found.longitude.values, lon):
        raise ValueError(f"Coordinate mismatch for {name}")
    return found


def grid_of(ds):
    lat = np.asarray(ds.latitude.values)
    lon = np.asarray(ds.longitude.values)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise ValueError("A regular latitude/longitude ERA5 grid is required")
    if not np.all(np.diff(lat) < 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("Expected descending latitudes and ascending longitudes")
    return lat, lon


def new_state(ny, nx, ns, n_years):
    shape = (ns, ny, nx)
    return {"n": np.zeros(shape, dtype=np.int32),
            "t_sum": np.zeros(shape), "t_sq": np.zeros(shape),
            "rh_sum": np.zeros(shape), "rh_sq": np.zeros(shape),
            "w_sum": np.zeros(shape), "w_sq": np.zeros(shape),
            "sf_sum": np.zeros((ns, n_years, ny, nx)),
            "sf_days": np.zeros((ns, n_years, ny, nx), dtype=np.int16),
            "nsp_sum": np.zeros((ns, n_years, ny, nx)),
            "nsp_days": np.zeros((ns, n_years, ny, nx), dtype=np.int16)}


def add_hour(state, s, t, rh, w):
    valid = np.isfinite(t) & np.isfinite(rh) & np.isfinite(w)
    if not np.any(valid):
        return
    # A common valid mask keeps the mean, SD and wind p90 comparable.
    state["n"][s] += valid
    for prefix, array in (("t", t), ("rh", rh), ("w", w)):
        safe = np.where(valid, array, 0.0)
        state[prefix + "_sum"][s] += safe
        state[prefix + "_sq"][s] += safe * safe


def update_block(state, si, yi, instant, precipitation, timestamps, snow_dates,
                 lat, lon, wind_store, hour_offset):
    t = field(instant, "t2m", timestamps, lat, lon)
    td = field(instant, "d2m", timestamps, lat, lon)
    u = field(instant, "u10", timestamps, lat, lon)
    v = field(instant, "v10", timestamps, lat, lon)
    # Loading one UTC day at a time bounds peak memory even in long blocks.
    for k in range(0, len(timestamps), 24):
        tc = np.asarray(t.isel({t.dims[0]: slice(k, k + 24)}).values,
                        dtype=np.float32) - 273.15
        tdc = np.asarray(td.isel({td.dims[0]: slice(k, k + 24)}).values,
                         dtype=np.float32) - 273.15
        uc = np.asarray(u.isel({u.dims[0]: slice(k, k + 24)}).values,
                        dtype=np.float32)
        vc = np.asarray(v.isel({v.dims[0]: slice(k, k + 24)}).values,
                        dtype=np.float32)
        # RH relative to liquid water, estimated from T and dew point.
        with np.errstate(over="ignore", invalid="ignore"):
            rh = 100 * np.exp(17.625 * tdc / (243.04 + tdc)
                              - 17.625 * tc / (243.04 + tc))
        rh = np.clip(rh, 0, 100)
        wind = np.hypot(uc, vc)
        valid = np.isfinite(tc) & np.isfinite(rh) & np.isfinite(wind)
        wind_store[hour_offset + k:hour_offset + k + 24] = np.where(
            valid, wind, np.nan)
        for h in range(24):
            add_hour(state, si, tc[h], rh[h], wind[h])
        del tc, tdc, uc, vc, rh, wind

    sf = field(precipitation, "sf", snow_dates, lat, lon)
    tp = field(precipitation, "tp", snow_dates, lat, lon)
    for k in range(len(snow_dates)):
        snow = np.asarray(sf.isel({sf.dims[0]: k}).values, dtype=np.float32)
        total = np.asarray(tp.isel({tp.dims[0]: k}).values, dtype=np.float32)
        valid_snow = np.isfinite(snow) & (snow >= -1e-7)
        valid_both = valid_snow & np.isfinite(total) & (total >= -1e-7)
        snow = np.maximum(snow, 0)
        total = np.maximum(total, 0)
        state["sf_sum"][si, yi] += np.where(valid_snow, snow, 0) * 1000
        state["sf_days"][si, yi] += valid_snow
        state["nsp_sum"][si, yi] += np.where(
            valid_both, np.maximum(total - snow, 0), 0) * 1000
        state["nsp_days"][si, yi] += valid_both


def save_checkpoint(path, signature, lat, lon, state, completed):
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as f:
        np.savez_compressed(f, signature=signature, lat=lat, lon=lon,
                            completed=np.array(sorted(completed)), **state)
    temp.replace(path)


def load_checkpoint(path, signature):
    with np.load(path, allow_pickle=False) as saved:
        if saved["signature"].item() != signature:
            raise ValueError("Checkpoint options differ; choose another work directory")
        lat, lon = saved["lat"].copy(), saved["lon"].copy()
        done = set(saved["completed"].tolist())
        state = {name: saved[name].copy() for name in (
            "n", "t_sum", "t_sq", "rh_sum", "rh_sq", "w_sum", "w_sq",
            "sf_sum", "sf_days", "nsp_sum", "nsp_days")}
    return lat, lon, state, done


def tcc_bounds(area):
    """Include the whole edge cells of the 0.1-degree ERA5 grid."""
    n, w, s, e = area
    return (max(-180, w - 0.05), max(-90, s - 0.05),
            min(180, e + 0.05), min(90, n + 0.05))


def tcc_tile_bounds(path):
    if not path.is_file():
        return None
    from rasterio import open as rio_open
    from rasterio.errors import RasterioIOError
    try:
        with rio_open(path) as src:
            eps = 1e-6
            if (src.crs and src.crs.to_epsg() == 4326 and src.count == 1
                    and abs(src.res[0] - TCC_RESOLUTION) < eps
                    and abs(src.res[1] - TCC_RESOLUTION) < eps):
                return src.bounds
    except RasterioIOError:
        pass
    return None


def valid_tcc_tile(path):
    return tcc_tile_bounds(path) is not None


def cached_tcc_tiles(cache_dir, area):
    """Avoid Earthdata when a local inventory or tile footprints prove coverage."""
    manifest = cache_dir / "inventory.json"
    key = json.dumps(list(area))
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            names = entries.get(key)
            if (isinstance(names, list) and all(
                    isinstance(name, str) and Path(name).name == name
                    and valid_tcc_tile(cache_dir / name) for name in names)):
                return [cache_dir / name for name in names]
        except (OSError, ValueError, TypeError):
            pass

    # Without an inventory, cached footprints can prove coverage only if
    # their union fills the whole requested rectangle.
    w, s, e, n = tcc_bounds(area)
    tiles = []
    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*_tcc_2020.tif")):
            bounds = tcc_tile_bounds(path)
            if (bounds is not None and bounds.right > w and bounds.left < e
                    and bounds.top > s and bounds.bottom < n):
                tiles.append((path, bounds))
    if not tiles:
        return None
    edges = sorted({w, e} | {max(w, min(e, x)) for _, b in tiles
                             for x in (b.left, b.right)})
    eps = 1e-6
    for left, right in zip(edges, edges[1:]):
        if right - left <= eps:
            continue
        intervals = sorted((b.bottom, b.top) for _, b in tiles
                           if b.left <= left + eps and b.right >= right - eps)
        covered_to = s
        for bottom, top in intervals:
            if bottom > covered_to + eps:
                break
            covered_to = max(covered_to, top)
            if covered_to >= n - eps:
                break
        if covered_to < n - eps:
            return None
    return [path for path, _ in tiles]


def save_tcc_inventory(cache_dir, area, names):
    if not cache_dir.is_dir():
        return
    path = cache_dir / "inventory.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        entries = {}
    entries[json.dumps(list(area))] = sorted(set(names))
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(entries), encoding="utf-8")
    temp.replace(path)


def tcc_names(granule):
    return sorted({unquote(Path(urlsplit(link).path).name)
                   for link in granule.data_links()
                   if unquote(urlsplit(link).path).lower().endswith("_tcc_2020.tif")})


def tcc_granules(area):
    import earthaccess
    auth = earthaccess.login()  # environment, .netrc, then interactive
    if not auth.authenticated:
        raise RuntimeError("NASA Earthdata login failed")
    w, s, e, n = tcc_bounds(area)
    print("Searching NASA Earthdata for 2020 TCC tiles...", flush=True)
    granules = earthaccess.search_data(
        doi=TCC_DOI, bounding_box=(w, s, e, n),
        temporal=("2020-01-01", "2020-12-31"))
    return [g for g in granules if tcc_names(g)]


def tcc_tiles(args, area):
    """Reuse native tiles, fetching only missing granules for this area."""
    cache_dir = args.tcc_cache_dir
    cached_paths = cached_tcc_tiles(cache_dir, area)
    if cached_paths is not None:
        if not cached_paths and not args.allow_empty_tcc:
            raise RuntimeError("No 2020 TCC tiles found for this area")
        print(f"Using {len(cached_paths)} TCC tile(s) from {cache_dir}", flush=True)
        yield from cached_paths
        return

    import earthaccess
    matches = tcc_granules(area)
    if not matches:
        save_tcc_inventory(cache_dir, area, [])
        if args.allow_empty_tcc:
            return
        raise RuntimeError("No 2020 TCC tiles found via NASA search")
    temporary = args.work_dir / "tcc_temporary"
    used = set()
    try:
        for i, granule in enumerate(matches, 1):
            names = set(tcc_names(granule)) - used
            if not names:
                continue
            cached = {name for name in names if valid_tcc_tile(cache_dir / name)}
            for name in sorted(cached):
                used.add(name)
                print(f"Reusing TCC tile: {cache_dir / name}", flush=True)
                yield cache_dir / name
            missing = names - cached
            if not missing:
                continue
            print(f"NASA TCC granule {i}/{len(matches)}: fetching {len(missing)} tile(s)",
                  flush=True)
            temporary.mkdir(parents=True, exist_ok=True)
            files = {Path(p).name: Path(p) for p in earthaccess.download(
                [granule], str(temporary))}
            for name in sorted(missing):
                tile = files.get(name)
                if tile is None or not valid_tcc_tile(tile):
                    raise RuntimeError(f"NASA did not return a valid TCC tile: {name}")
                if args.keep_tcc:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    saved = cache_dir / name
                    tile.replace(saved)
                    used.add(name)
                    yield saved
                else:
                    used.add(name)
                    yield tile
                    tile.unlink(missing_ok=True)
        save_tcc_inventory(cache_dir, area, used)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def canopy_on_grid(args, area, lat, lon):
    from rasterio import open as rio_open
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.windows import Window, from_bounds
    from affine import Affine
    from rasterio.warp import reproject
    spacing_y = float(np.median(-np.diff(lat)))
    spacing_x = float(np.median(np.diff(lon)))
    transform = from_origin(float(lon[0] - spacing_x / 2),
                            float(lat[0] + spacing_y / 2), spacing_x, spacing_y)
    shape = (len(lat), len(lon))
    weighted = np.zeros(shape, dtype=np.float64)
    covered = np.zeros(shape, dtype=np.float64)
    seen = 0
    for path in tcc_tiles(args, area):
        with rio_open(path) as src:
            if not src.crs or src.count != 1:
                raise ValueError(f"Expected georeferenced, one-band TCC: {path}")
            # Skip local tiles outside the actual ERA5 extent.
            if (src.bounds.right <= lon[0] - spacing_x/2 or
                    src.bounds.left >= lon[-1] + spacing_x/2 or
                    src.bounds.top <= lat[-1] - spacing_y/2 or
                    src.bounds.bottom >= lat[0] + spacing_y/2):
                continue
            seen += 1
            # Read only the requested footprint, in small source windows.
            left, bottom, right, top = (
                float(lon[0] - spacing_x/2), float(lat[-1] - spacing_y/2),
                float(lon[-1] + spacing_x/2), float(lat[0] + spacing_y/2))
            intersection = (max(left, src.bounds.left),
                            max(bottom, src.bounds.bottom),
                            min(right, src.bounds.right),
                            min(top, src.bounds.top))
            raw = from_bounds(*intersection, transform=src.transform)
            col0 = max(0, int(np.floor(raw.col_off)))
            row0 = max(0, int(np.floor(raw.row_off)))
            col1 = min(src.width, int(np.ceil(raw.col_off + raw.width)))
            row1 = min(src.height, int(np.ceil(raw.row_off + raw.height)))
            for row in range(row0, row1, 4096):
                for col in range(col0, col1, 4096):
                    window = Window(col, row, min(4096, col1-col),
                                    min(4096, row1-row))
                    data = src.read(1, window=window)
                    valid = data <= 100
                    if not np.any(valid):
                        continue
                    data[~valid] = 255
                    # Pad with missing pixels by one output cell. GDAL's
                    # average warp can skip a partial edge cell if its center
                    # falls outside a narrow source window.
                    pad_x = int(np.ceil(spacing_x / src.res[0])) + 2
                    pad_y = int(np.ceil(spacing_y / src.res[1])) + 2
                    source_transform = (src.window_transform(window)
                                        * Affine.translation(-pad_x, -pad_y))
                    mask = np.pad(valid.astype(np.float32),
                                  ((pad_y, pad_y), (pad_x, pad_x)))
                    data = np.pad(data, ((pad_y, pad_y), (pad_x, pad_x)),
                                  constant_values=255)
                    mean = np.full(shape, np.nan, dtype=np.float32)
                    fraction = np.zeros(shape, dtype=np.float32)
                    reproject(data, mean, src_transform=source_transform,
                              src_crs=src.crs, src_nodata=255,
                              dst_transform=transform, dst_crs=CRS.from_epsg(4326),
                              dst_nodata=np.nan, resampling=Resampling.average)
                    reproject(mask, fraction, src_transform=source_transform,
                              src_crs=src.crs, dst_transform=transform,
                              dst_crs=CRS.from_epsg(4326), dst_nodata=0,
                              resampling=Resampling.average)
                    fraction = np.clip(fraction, 0, 1)
                    weighted += np.where(np.isfinite(mean), mean, 0) * fraction
                    covered += fraction
        print(f"Coarsened {path.name}", flush=True)
    if seen == 0:
        if args.allow_empty_tcc:
            return (np.full(shape, np.nan, dtype=np.float32),
                    np.zeros(shape, dtype=np.float32))
        raise ValueError("No 2020 TCC tiles intersect the ERA5 grid")
    with np.errstate(invalid="ignore", divide="ignore"):
        canopy = np.where(covered > 0, weighted / covered, np.nan)
    return canopy.astype(np.float32), np.clip(covered, 0, 1).astype(np.float32)


def exact_wind_p90(wind_store, season_hours, ny, nx):
    """Calculate linear-interpolated hourly p90 in small pixel batches."""
    started = perf_counter()
    result = np.full((len(season_hours), ny * nx), np.nan, dtype=np.float32)
    offset = 0
    for si, hours in enumerate(season_hours):
        values = wind_store[offset:offset + hours].reshape(hours, ny * nx)
        for start in range(0, ny * nx, 128):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                result[si, start:start + 128] = np.nanquantile(
                    values[:, start:start + 128], 0.9, axis=0)
        offset += hours
    print(f"[timing] exact wind p90: {perf_counter() - started:.1f} s",
          flush=True)
    return result.reshape(len(season_hours), ny, nx)


def make_output(state, seasons, years, lat, lon, canopy, canopy_fraction, area,
                wind_store, season_hours):
    n = state["n"]
    coords = {"season": seasons, "latitude": lat, "longitude": lon}
    ds = xr.Dataset(coords=coords)
    def put(name, data, units, description, dims=("season", "latitude", "longitude")):
        ds[name] = xr.DataArray(np.asarray(data, dtype=np.float32), dims=dims)
        ds[name].attrs.update(units=units, long_name=description)
    with np.errstate(invalid="ignore", divide="ignore"):
        for key, prefix, units in (("temperature", "t", "degree_Celsius"),
                                   ("relative_humidity", "rh", "%"),
                                   ("wind_speed", "w", "m s-1")):
            avg = np.where(n > 0, state[prefix + "_sum"] / n, np.nan)
            sd = np.where(n > 0, np.sqrt(np.maximum(
                state[prefix + "_sq"] / np.maximum(n, 1) - avg**2, 0)), np.nan)
            put(key + "_mean", avg, units, f"Mean hourly {key.replace('_', ' ')}")
            put(key + "_sd", sd, units, f"Population SD of hourly {key.replace('_', ' ')}")
    put("wind_speed_p90", exact_wind_p90(
        wind_store, season_hours, len(lat), len(lon)),
        "m s-1", "Exact linear-interpolated 90th percentile of hourly 10 m wind speed")
    ds["valid_hour_count"] = xr.DataArray(n, dims=("season", "latitude", "longitude"))
    ds["valid_hour_count"].attrs["long_name"] = "Hours valid for all meteorological metrics"
    # Reject incomplete season-years for each pixel; 00 UTC is the prior day.
    expected = np.array([[sum(calendar.monthrange(yy, mm)[1]
                              for yy, mm in season_months(s, y))
                          for y in years] for s in seasons])
    def seasonal_totals(prefix):
        complete = state[prefix + "_days"] == expected[:, :, None, None]
        totals = np.where(complete, state[prefix + "_sum"], np.nan)
        count = complete.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(totals, axis=1) / np.where(count > 0, count, np.nan)
            sd = np.sqrt(np.nansum(np.where(
                complete, (totals - mean[:, None])**2, 0), axis=1)
                / np.where(count > 0, count, np.nan))
        return mean, sd, count

    snow_mean, snow_sd, snow_count = seasonal_totals("sf")
    put("snowfall_mean", snow_mean, "mm water equivalent",
        "Mean total snowfall per complete season-year")
    put("snowfall_sd", snow_sd, "mm water equivalent",
        "Population SD of total snowfall between complete season-years")
    ds["complete_snowfall_seasons"] = xr.DataArray(
        snow_count.astype(np.int16), dims=("season", "latitude", "longitude"))
    nsp_mean, nsp_sd, nsp_count = seasonal_totals("nsp")
    put("non_snow_precipitation_mean", nsp_mean, "mm water equivalent",
        "Mean total non-snow precipitation per complete season-year")
    put("non_snow_precipitation_sd", nsp_sd, "mm water equivalent",
        "Population SD of non-snow precipitation totals between complete season-years")
    ds["complete_non_snow_precipitation_seasons"] = xr.DataArray(
        nsp_count.astype(np.int16), dims=("season", "latitude", "longitude"))
    put("tree_canopy_cover_2020", canopy, "%",
        "NASA ABoVE mean valid 30 m tree canopy cover, 2020",
        dims=("latitude", "longitude"))
    put("tree_canopy_coverage_fraction", canopy_fraction, "1",
        "Approximate fraction of ERA5 cell with valid 2020 ABoVE data",
        dims=("latitude", "longitude"))
    def year_label(ym):
        yy, mm = ym
        return f"{calendar.month_abbr[mm]} {'Y-1' if yy == 1999 else 'Y'}"
    season_definition = "Winter-year Y: " + "; ".join(
        f"{s}={year_label(season_months(s, 2000)[0])} through "
        f"{year_label(season_months(s, 2000)[-1])}" for s in seasons) + "; all UTC"
    ds.attrs.update(source="ERA5-Land hourly; NASA ABoVE TCC 2020 (ORNLDAAC/2012)",
                    region_nwse=json.dumps(area), season_years=f"{years[0]}-{years[-1]}",
                    season_definition=season_definition,
                    humidity_definition="RH relative to liquid water from 2m temperature/dewpoint (Magnus)",
                    snowfall_definition="00 UTC cumulative snowfall assigned to previous UTC day",
                    non_snow_precipitation_definition="Sum of max(daily total precipitation - daily snowfall, 0); 00 UTC values assigned to previous UTC day; mm water equivalent",
                    wind_p90_definition="numpy nanquantile(0.9) of all valid hourly wind speed",
                    canopy_aggregation="Mean of valid pixels in 0.1-degree cells; missing cells retained")
    return ds


def run(args):
    tile_started = perf_counter()
    area = list(args.area)
    n, w, s, e = area
    if not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        raise ValueError("Expected --area NORTH WEST SOUTH EAST without a dateline crossing")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.out}; use --overwrite")
    years = list(range(args.start_year, args.end_year + 1))
    if not years:
        raise ValueError("End year must be at least start year")
    signature = json.dumps(["overlap-seasons-v2-precipitation", area, years, args.seasons, args.block_days])
    sequence = []
    season_hours = []
    for si, season in enumerate(args.seasons):
        hours = 0
        for yi, year in enumerate(years):
            for first, last in date_blocks(season, year, args.block_days):
                sequence.append((si, yi, season, year, first, last, hours))
                hours += ((last - first).days + 1) * 24
        season_hours.append(hours)
    season_starts = np.cumsum([0, *season_hours[:-1]])
    wind_path = args.work_dir / "hourly_wind_for_exact_p90.float32"
    checkpoint = args.work_dir / "climate_checkpoint.npz"
    if checkpoint.exists():
        lat, lon, state, done = load_checkpoint(checkpoint, signature)
        print(f"Resuming from checkpoint: {len(done)} completed blocks", flush=True)
    else:
        lat = lon = state = None
        done = set()
    wind_store = None
    if state is not None:
        expected_size = sum(season_hours) * len(lat) * len(lon) * 4
        if not wind_path.exists() or wind_path.stat().st_size != expected_size:
            raise ValueError("Hourly wind scratch file is missing or incomplete; "
                             "use a new --work-dir to restart")
        wind_store = np.memmap(wind_path, mode="r+", dtype=np.float32,
                               shape=(sum(season_hours), len(lat), len(lon)))
    needed_precip_dates = precipitation_dates_by_month(sequence)
    remaining_precip_uses = Counter()
    remaining_instant_uses = Counter()
    for _, _, season, year, first, last, _ in sequence:
        key = f"{season}_{year}_{first:%Y%m%d}_{last:%Y%m%d}"
        if key not in done:
            remaining_instant_uses[(first, last)] += 1
            first_stamp = first + timedelta(days=1)
            last_stamp = last + timedelta(days=1)
            for year_month in {(first_stamp.year, first_stamp.month),
                               (last_stamp.year, last_stamp.month)}:
                remaining_precip_uses[year_month] += 1
    import cdsapi
    client = cdsapi.Client()
    try:
        for si, yi, season, year, first, last, local_hour in sequence:
            hour_offset = int(season_starts[si] + local_hour)
            key = f"{season}_{year}_{first:%Y%m%d}_{last:%Y%m%d}"
            if key in done:
                continue
            block_started = perf_counter()
            # Cache by actual dates so the overlapping seasons share one CDS download.
            download_key = f"{first:%Y%m%d}_{last:%Y%m%d}"
            days = [first + timedelta(days=j)
                    for j in range((last - first).days + 1)]
            # All 24 instantaneous hours in the named UTC days.
            hourly_path = args.work_dir / f"{download_key}_instant.zip"
            download_seconds = retrieve(
                client, INSTANT, days,
                [f"{hour:02d}:00" for hour in range(24)], area, hourly_path)
            # The following day at 00 UTC has that day's precipitation totals.
            snow_stamps = [d + timedelta(days=1) for d in days]
            groups = {}
            for d in snow_stamps:
                groups.setdefault((d.year, d.month), []).append(d)
            precip_inputs = []
            for j, dates in enumerate(groups.values()):
                path, request_dates, shared = precipitation_request(
                    args.work_dir, download_key, j, dates, needed_precip_dates)
                download_seconds += retrieve(
                    client, ["snowfall", "total_precipitation"],
                    request_dates, ["00:00"], area, path)
                precip_inputs.append((path, dates, shared))
            input_started = perf_counter()
            with ExitStack() as stack:
                met = open_response(hourly_path, stack)
                precip_sources = []
                for path, dates, _ in precip_inputs:
                    source = open_response(path, stack)
                    sf_var = next(source[name] for name in ALIASES["sf"]
                                  if name in source)
                    time_dim = next(d for d in ("valid_time", "time")
                                    if d in sf_var.dims)
                    precip_sources.append(source.sel({
                        time_dim: np.array(dates, dtype="datetime64[ns]")}))
                if len(precip_sources) > 1:
                    sf_var = next(precip_sources[0][name] for name in ALIASES["sf"]
                                  if name in precip_sources[0])
                    time_dim = next(d for d in ("valid_time", "time")
                                    if d in sf_var.dims)
                    precipitation = xr.concat(precip_sources, dim=time_dim,
                                              data_vars="all", coords="minimal",
                                              compat="override")
                else:
                    precipitation = precip_sources[0]
                input_seconds = perf_counter() - input_started
                this_lat, this_lon = grid_of(met)
                if state is None:
                    if this_lat.size * this_lon.size > args.max_cells:
                        raise ValueError(f"Area has {this_lat.size * this_lon.size:,} "
                                         f"grid cells (> {args.max_cells:,}); use "
                                         "a smaller --area or increase --max-cells")
                    lat, lon = this_lat, this_lon
                    state = new_state(len(lat), len(lon), len(args.seasons), len(years))
                    wind_store = np.memmap(
                        wind_path, mode="w+", dtype=np.float32,
                        shape=(sum(season_hours), len(lat), len(lon)))
                if not np.array_equal(lat, this_lat) or not np.array_equal(lon, this_lon):
                    raise ValueError("CDS returned a different grid across requests")
                timestamps = [np.datetime64(d) + np.timedelta64(h, 'h')
                              for d in days for h in range(24)]
                compute_started = perf_counter()
                update_block(state, si, yi, met, precipitation, timestamps,
                             [np.datetime64(d) for d in snow_stamps], lat, lon,
                             wind_store, hour_offset)
                compute_seconds = perf_counter() - compute_started
            checkpoint_started = perf_counter()
            wind_store.flush()
            done.add(key)
            save_checkpoint(checkpoint, signature, lat, lon, state, done)
            checkpoint_seconds = perf_counter() - checkpoint_started
            remaining_instant_uses[(first, last)] -= 1
            for year_month in groups:
                remaining_precip_uses[year_month] -= 1
            if args.delete_raw and remaining_instant_uses[(first, last)] == 0:
                hourly_path.unlink(missing_ok=True)
            shutil.rmtree(hourly_path.with_suffix(""), ignore_errors=True)
            for path, dates, shared in precip_inputs:
                year_month = (dates[0].year, dates[0].month)
                if args.delete_raw:
                    if not shared or remaining_precip_uses[year_month] == 0:
                        path.unlink(missing_ok=True)
                shutil.rmtree(path.with_suffix(""), ignore_errors=True)
            if args.delete_raw:
                for year_month in groups:
                    if remaining_precip_uses[year_month] == 0:
                        year, month = year_month
                        (args.work_dir / f"{year}_{month:02d}_precip_00utc.zip")\
                            .unlink(missing_ok=True)
            print(f"Finished {key}; checkpoint written", flush=True)
            print(f"[timing] {key}: CDS={download_seconds:.1f} s, "
                  f"open/extract={input_seconds:.1f} s, "
                  f"compute={compute_seconds:.1f} s, "
                  f"checkpoint={checkpoint_seconds:.1f} s, "
                  f"total={perf_counter() - block_started:.1f} s", flush=True)
        canopy_started = perf_counter()
        canopy, fraction = canopy_on_grid(args, area, lat, lon)
        print(f"[timing] canopy tile {area}: "
              f"{perf_counter() - canopy_started:.1f} s", flush=True)
        summary_started = perf_counter()
        ds = make_output(state, args.seasons, years, lat, lon, canopy, fraction,
                         area, wind_store, season_hours)
        print(f"[timing] derived summary tile {area}: "
              f"{perf_counter() - summary_started:.1f} s", flush=True)
        temp = args.out.with_name(args.out.name + ".partial")
        encoding = {name: {"zlib": True, "complevel": 4}
                    for name in ds.data_vars}
        write_started = perf_counter()
        ds.to_netcdf(temp, engine="netcdf4", encoding=encoding)
        temp.replace(args.out)
        print(f"[timing] tile NetCDF write: "
              f"{perf_counter() - write_started:.1f} s", flush=True)
        checkpoint.unlink()
        del wind_store
        wind_path.unlink()
        print(f"Saved {args.out}", flush=True)
        print(f"[timing] tile total: {perf_counter() - tile_started:.1f} s",
              flush=True)
    finally:
        # Failed downloads and a valid checkpoint remain for a resumed run.
        pass


def run_area_mosaic(args):
    """Commit one tile at a time without modifying the previous partial file."""
    from hashlib import sha256
    from netCDF4 import Dataset as NetCDF

    north, west, south, east = args.area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("Expected --area NORTH WEST SOUTH EAST without a dateline crossing")
    if any(abs(value * 10 - round(value * 10)) > 1e-6 for value in args.area):
        raise ValueError("--area bounds must fall on ERA5-Land's 0.1-degree grid")
    lat = np.round(np.arange(north, south - 0.001, -0.1), 2)
    lon = np.round(np.arange(west, east + 0.001, 0.1), 2)
    max_columns = args.max_tile_cells // len(lat)
    if max_columns < 2:
        raise ValueError("--max-tile-cells is too small for this latitude range")
    interval = max_columns - 1  # One shared longitude at each tile seam.
    tiles = [(j, min(j + interval, len(lon) - 1))
             for j in range(0, len(lon) - 1, interval)]
    signature = json.dumps(["area-v6-precipitation", args.area, args.start_year,
                            args.end_year, args.seasons, args.block_days,
                            args.max_tile_cells])
    run_id = sha256(signature.encode()).hexdigest()[:12]
    root = args.work_dir / "work" / run_id
    root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    candidate = partial.with_name(partial.name + ".next")
    progress_path = root / "progress.json"

    def same_mosaic(dataset):
        return (dataset.attrs.get("region_nwse") == json.dumps(args.area)
                and dataset.attrs.get("season_years")
                == f"{args.start_year}-{args.end_year}"
                and dataset.attrs.get("run_signature", signature) == signature
                and np.array_equal(dataset.latitude.values, lat)
                and np.array_equal(dataset.longitude.values, lon)
                and list(dataset.season.values) == args.seasons)

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.out}; use --overwrite")
    # A timeout can leave this copy half-written. It was never committed.
    candidate.unlink(missing_ok=True)
    if args.overwrite:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        # Raw monthly ZIPs remain available for another calculation.
        for i in range(len(tiles)):
            folder = root / f"tile_{i:03d}"
            for name in ("summary.nc", "climate_checkpoint.npz",
                         "hourly_wind_for_exact_p90.float32"):
                (folder / name).unlink(missing_ok=True)
    progress = {"signature": signature, "completed": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["signature"] != signature:
            raise ValueError("Area checkpoint options differ from this run")
        completed = progress["completed"]
        if (not isinstance(completed, list) or any(
                not isinstance(i, int) or i < 0 or i >= len(tiles)
                for i in completed)):
            raise ValueError("Area checkpoint has invalid completed tile numbers")
    else:
        completed = []

    if partial.exists():
        try:
            with xr.open_dataset(partial, engine="netcdf4") as previous:
                if not same_mosaic(previous):
                    raise ValueError("Interrupted mosaic belongs to different options")
                if "run_signature" not in previous.attrs:
                    # Older versions wrote directly into this file. Read the
                    # committed slices once before trusting them on restart.
                    for i in completed:
                        start, end = tiles[i]
                        previous.isel(longitude=slice(start, end + 1)).load()
        except (OSError, RuntimeError) as exc:
            missing = [i for i in completed if not
                       (root / f"tile_{i:03d}" / "summary.nc").exists()]
            if missing:
                raise ValueError(
                    f"Cannot read {partial}; completed tiles {missing} were "
                    "written by the older script without retained summaries") from exc
            if not completed and not (root / "tile_000" / "summary.nc").exists():
                raise ValueError(f"Untracked damaged partial output: {partial}") from exc
            print(f"Discarding damaged partial mosaic: {partial}", flush=True)
            partial.unlink()
            progress = {"signature": signature, "completed": []}
            progress_path.unlink(missing_ok=True)
    elif completed:
        missing = [i for i in completed if not
                   (root / f"tile_{i:03d}" / "summary.nc").exists()]
        if missing:
            raise ValueError(
                f"Partial output is missing; completed tiles {missing} have "
                "no retained summary")
        progress = {"signature": signature, "completed": []}
        progress_path.unlink(missing_ok=True)

    if partial.exists() and not progress_path.exists():
        # A first-tile write may have committed just before its progress file.
        # Its summary still exists, so it can be applied again below.
        if not (root / "tile_000" / "summary.nc").exists():
            raise ValueError(f"Untracked partial output: {partial}")
        partial.unlink()

    args.tcc_cache_dir = args.work_dir / "tcc"
    print(f"Area {args.area}: {len(tiles)} spatial tile(s), "
          f"monthly ERA5 requests, raw downloads in {root}", flush=True)
    for i, (start, end) in enumerate(tiles):
        if i in progress["completed"]:
            continue
        tile_started = perf_counter()
        folder = root / f"tile_{i:03d}"
        staged = folder / "summary.nc"
        if not staged.exists():
            child = argparse.Namespace(**vars(args))
            child.area = [north, float(lon[start]), south, float(lon[end])]
            child.work_dir = folder
            child.out = staged
            child.overwrite = False
            child.max_cells = args.max_tile_cells
            child.allow_empty_tcc = True
            print(f"Tile {i + 1}/{len(tiles)}: {child.area}", flush=True)
            run(child)

        mosaic_started = perf_counter()
        with xr.open_dataset(staged, engine="netcdf4") as tile:
            tile_lat = np.asarray(tile.latitude.values)
            tile_lon = np.asarray(tile.longitude.values)
            if (not np.allclose(tile_lat, lat, atol=1e-5) or
                    not np.allclose(tile_lon, lon[start:end + 1], atol=1e-5)):
                raise ValueError(f"Unexpected ERA5-Land grid in tile {i}")
            if partial.exists():
                shutil.copyfile(partial, candidate)
            else:
                xr.Dataset(coords={"season": args.seasons,
                                   "latitude": lat, "longitude": lon},
                           attrs=dict(tile.attrs, region_nwse=json.dumps(args.area),
                                      processing="Spatial tiles, exact hourly summaries",
                                      run_signature=signature))\
                    .to_netcdf(candidate, engine="netcdf4")
            with NetCDF(candidate, "a") as dest:
                for name, array in tile.data_vars.items():
                    if name not in dest.variables:
                        limits = ((1, 64, 100) if array.ndim == 3 else (64, 100))
                        chunks = tuple(min(limit, len(dest.dimensions[dimension]))
                                       for limit, dimension in zip(limits, array.dims))
                        fill = np.nan if np.issubdtype(array.dtype, np.floating) else -1
                        variable = dest.createVariable(
                            name, array.dtype, array.dims, zlib=True,
                            complevel=4, chunksizes=chunks, fill_value=fill)
                        variable.setncatts(dict(array.attrs))
                    if array.ndim == 3:
                        dest[name][:, :, start:end + 1] = array.values
                    else:
                        dest[name][:, start:end + 1] = array.values
                dest.sync()

        candidate.replace(partial)
        progress["completed"].append(i)
        temp_progress = progress_path.with_suffix(".json.partial")
        temp_progress.write_text(json.dumps(progress), encoding="utf-8")
        temp_progress.replace(progress_path)
        print(f"Added tile {i + 1}/{len(tiles)} to {partial}", flush=True)
        print(f"[timing] mosaic write tile {i + 1}/{len(tiles)}: "
              f"{perf_counter() - mosaic_started:.1f} s", flush=True)
        print(f"[timing] tile {i + 1}/{len(tiles)} including processing and "
              f"mosaic write: {perf_counter() - tile_started:.1f} s", flush=True)

    partial.replace(args.out)
    progress_path.unlink(missing_ok=True)
    for i in range(len(tiles)):
        (root / f"tile_{i:03d}" / "summary.nc").unlink(missing_ok=True)
    print(f"Saved {args.out}", flush=True)


def main():
    started = perf_counter()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--area", type=float, nargs=4, required=True,
                   metavar=("N", "W", "S", "E"),
                   help="Exact requested region; internal tiling is automatic")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=2021)
    window = p.add_mutually_exclusive_group()
    window.add_argument("--seasons", nargs="+", choices=list(SEASON_MONTHS),
                        help="Named season windows; default OND DJF FMA")
    window.add_argument("--months", nargs=2, type=parse_month,
                        metavar=("START", "END"),
                        help="One inclusive month range, e.g. October April; "
                             "accepts names or 1–12")
    p.add_argument("--block-days", type=int, default=31,
                   help="Days per ERA5 request (default 31: one request per month)")
    p.add_argument("--max-tile-cells", type=int, default=30000,
                   help="Approximate spatial cells per CDS request (default 30000)")
    p.add_argument("--keep-tcc", action="store_true",
                   help="Keep fetched native ABoVE GeoTIFF tiles in WORK_DIR/tcc/; "
                        "existing tiles there are always reused")
    p.add_argument("--work-dir", type=Path,
                   default=Path("../../scratch/climate_site_selection/"),
                   help="Directory for reusable raw ZIPs and processing state")
    p.add_argument("--out", type=Path, default=None,
                   help="Final NetCDF (default WORK_DIR/seasonal_climate.nc)")
    p.add_argument("--delete-raw", action="store_true",
                   help="Delete downloaded ZIPs after their last needed block; "
                        "default is to retain them")
    p.add_argument("--overwrite", action="store_true",
                   help="Recalculate all tiles, reusing retained ERA5 ZIPs")
    args = p.parse_args()
    if args.months:
        args.seasons = ["_".join(calendar.month_abbr[m].upper()
                                 for m in args.months)]
    elif args.seasons is None:
        args.seasons = list(SEASON_MONTHS)
    if args.out is None:
        args.out = args.work_dir / "seasonal_climate.nc"
    if args.block_days < 1 or args.max_tile_cells < 2:
        p.error("--block-days must be positive and --max-tile-cells at least 2")
    if len(set(args.seasons)) != len(args.seasons):
        p.error("Each season can appear only once")
    if args.start_year > args.end_year:
        p.error("--end-year must be at least --start-year")
    run_area_mosaic(args)
    print(f"[timing] entire run: {perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()

