"""ERA5-Land hourly flux preparation for FSM2."""

import numpy as np
import pandas as pd
import xarray as xr


FORMAT = "hourly_mean_flux_v1"

# Output units, multiplier applied before division by 3600, and tolerance
# for small negative hourly increments in the original accumulation units.
FIELDS = {
    "sf":   ("kg m-2 s-1", 1000.0, 1e-7, "Snowfall mass flux"),
    "tp":   ("kg m-2 s-1", 1000.0, 1e-7, "Total precipitation mass flux"),
    "ssrd": ("W m-2",         1.0, 10.0, "Downward surface shortwave radiation flux"),
    "strd": ("W m-2",         1.0,  1.0, "Downward surface longwave radiation flux"),
}


def time_name(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


def normalized(ds):
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


def plateau_fill(values, times, y, x, a, b):
    """Evaluate one missing run, [a, b). Return its audit record or a reason."""
    if b - a > 4:
        return None, "gap exceeds four hours"
    if b == len(times):
        return None, "no following observation"

    reset = times[a].hour == 1
    if (a == 0 and not reset) or np.any(times[a + 1:b + 1].hour == 1):
        return None, "gap lacks bounds within one forecast cycle"

    start = a if reset else a - 1
    endpoint = float(values[b, y, x])
    if not np.isfinite(endpoint) or endpoint < 0:
        return None, "invalid following accumulation"

    if not reset:
        left = float(values[a - 1, y, x])
        if not np.isfinite(left) or abs(left - endpoint) > 2:
            return None, "bounding accumulations differ by more than 2 J/m²"

    y0, x0 = max(0, y - 1), max(0, x - 1)
    block = values[start:b + 1, y0:y + 2, x0:x + 2]
    neighbour_mask = np.ones(block.shape[1:], dtype=bool)
    neighbour_mask[y - y0, x - x0] = False
    neighbours = block[:, neighbour_mask]
    neighbours = neighbours[:, np.isfinite(neighbours).all(axis=0)]

    if neighbours.shape[1] < 3:
        return None, "fewer than three complete neighbours"

    variation = float(np.ptp(neighbours, axis=0).max())
    if variation > 3:
        return None, "neighbour accumulation range exceeds 3 J/m²"

    if reset:
        lower = neighbours[0].min() - 2
        upper = neighbours[0].max() + 2
        if not lower <= endpoint <= upper:
            return None, "endpoint outside neighbouring first-hour range"

    record = {
        "start_utc": str(times[a]),
        "end_utc": str(times[b - 1]),
        "hours": int(b - a),
        "source_utc": str(times[b]),
        "cumulative_J_m2": endpoint,
        "complete_neighbours": int(neighbours.shape[1]),
        "max_range_J_m2": variation,
    }
    return record, None


def repair_ssrd(raw, start, stop):
    """Repair gaps relevant to this month's rates using original neighbours."""
    times = pd.DatetimeIndex(raw.time.values)
    original = raw.ssrd.transpose("time", "latitude", "longitude").values
    repaired = original.copy()
    audit = {}

    # start-1 is needed for the first difference in this month's file.
    candidates = (
        (~np.isfinite(original[start - 1:stop])).any(axis=0)
        & np.isfinite(original).any(axis=0)
    )

    for y, x in zip(*np.where(candidates)):
        missing = np.flatnonzero(~np.isfinite(original[:, y, x]))
        gaps = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
        for gap in gaps:
            a, b = int(gap[0]), int(gap[-1]) + 1
            if b <= start - 1 or a >= stop:
                continue

            record, reason = plateau_fill(original, times, y, x, a, b)
            if record is None:
                print(f"Unfilled SSRD ({y},{x}) at {times[a]}: {reason}", flush=True)
                continue

            repaired[a:b, y, x] = record["cumulative_J_m2"]
            if start <= a < stop:
                key = f"cell_y{y:04d}_x{x:04d}"
                audit.setdefault(key, []).append(record)

    return raw.ssrd.copy(data=repaired), audit


def prepare_fluxes(weather, previous, following=None):
    """Return one month of hour-ending mean fluxes and its SSRD fill audit."""
    original_time = time_name(weather)
    fields = list(FIELDS)

    # Six previous hours supply the preceding value and context for a
    # possible four-hour SSRD gap ending at the previous month's last hour.
    before = normalized(previous)[fields].isel(time=slice(-6, None))
    month = normalized(weather)
    pieces = [before, month[fields]]
    if following is not None:
        pieces.append(normalized(following)[fields].isel(time=slice(0, 1)))

    for piece in pieces:
        for name in fields:
            allowed = (
                {"m", "m of water equivalent"} if name in ("sf", "tp")
                else {"J m**-2", "J m^-2", "J m-2"}
            )
            units = piece[name].attrs.get("units")
            if units not in allowed:
                raise ValueError(f"Unexpected raw units for {name}: {units!r}")

    raw = xr.concat(pieces, dim="time", join="exact")
    raw = raw.transpose("time", "latitude", "longitude")
    times = pd.DatetimeIndex(raw.time.values)
    if not np.all(np.diff(times.values) == np.timedelta64(1, "h")):
        raise ValueError("Raw input is not continuous across the month boundary")

    start = before.sizes["time"]
    stop = start + month.sizes["time"]
    raw["ssrd"], audit = repair_ssrd(raw, start, stop)

    result = month.copy()
    for name, (units, multiplier, tolerance, long_name) in FIELDS.items():
        cumulative = raw[name].astype("float64")
        amount = xr.where(
            raw.time.dt.hour == 1,
            cumulative,
            cumulative - cumulative.shift(time=1),
        ).isel(time=slice(start, stop))

        # Larger negative increments and nonfinite values stay missing.
        amount = amount.where(np.isfinite(amount) & (amount >= -tolerance))
        result[name] = amount.clip(min=0) * multiplier / 3600.0
        result[name].attrs = {
            "long_name": long_name,
            "units": units,
            "cell_methods": "time: mean",
        }

    if original_time != "time":
        result = result.rename({"time": original_time})

    return result, audit
