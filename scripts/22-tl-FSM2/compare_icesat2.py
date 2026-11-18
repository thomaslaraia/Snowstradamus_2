#!/usr/bin/env python3
"""Match grouped ATL08 observations to one assembled FSM2 member (UTC nearest hour)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROBS = ["prob_no_snow", "prob_ground_snow", "prob_ground_canopy_snow"]


def compare(observations, lookup, model, ground_threshold=0.5, canopy_threshold=0.5):
    out = observations.copy()
    # Prefer actual timestamps; otherwise approximate using granule start time.
    if "observation_time" in out:
        out["observation_time"] = pd.to_datetime(out.observation_time, utc=True)
    else:
        stamp = out["resource"].str.extract(r"ATL0[38]_(\d{14})", expand=False)
        out["observation_time"] = pd.to_datetime(stamp, format="%Y%m%d%H%M%S", utc=True)
    if out.observation_time.isna().any():
        raise ValueError("Cannot recover acquisition time from every resource name")
    out["model_time"] = out.observation_time.dt.round("h").dt.tz_localize(None)

    query = {
        "Eg_strong": xr.DataArray(out.Eg_strong.to_numpy(), dims="observation"),
        "Ev_strong": xr.DataArray(out.Ev_strong.to_numpy(), dims="observation"),
    }
    out[PROBS] = np.column_stack([lookup[p].interp(query).values for p in PROBS])
    valid_probs = np.isfinite(out[PROBS]).all(axis=1)
    out["icesat_class"] = pd.array(
        np.where(valid_probs, out[PROBS].to_numpy().argmax(axis=1), np.nan), dtype="Int8",
    )

    inside = np.ones(len(out), dtype=bool)
    indices = {}
    for dim, col in [("latitude", "lat"), ("longitude", "lon")]:
        values = model[dim].values
        if len(values) < 2:
            raise ValueError("Comparison requires at least two grid coordinates per axis")
        spacing = np.abs(np.diff(values))
        if not np.allclose(spacing, spacing[0]):
            raise ValueError(f"Expected regular {dim} grid")
        idx = model.indexes[dim].get_indexer(out[col], method="nearest")
        inside &= np.isfinite(out[col]) & (
            np.abs(out[col] - values[idx]) <= spacing[0] / 2 + 1e-8
        )
        indices[dim] = idx
        out["model_" + dim] = values[idx]

    time_index = model.indexes["time"].get_indexer(out.model_time)
    inside &= time_index >= 0
    indices["time"] = np.maximum(time_index, 0)
    model_vars = ["ground_snow_mass", "canopy_snow_mass", "ground_snow_cover_fraction",]
    selection = model[model_vars].isel(
        {dim: xr.DataArray(idx, dims="observation") for dim, idx in indices.items()}
    ).load()
    for name in selection.data_vars:
        out[name] = np.where(inside, selection[name].values, np.nan)

    valid_model = np.isfinite(out[model_vars]).all(axis=1)
    ground = out.ground_snow_cover_fraction >= ground_threshold
    classes = np.where(ground, 1 + (out.canopy_snow_mass > canopy_threshold), 0)
    out["model_class"] = pd.array(np.where(valid_model, classes, np.nan), dtype="Int8")
    matched = valid_probs & valid_model
    out["matched"] = matched

    # Retain all groups; give each cell/overpass total comparison weight 1.
    keys = ["resource", "model_time", "model_latitude", "model_longitude"]
    count = out.loc[matched].groupby(keys)["resource"].transform("size")
    out["comparison_weight"] = 0.
    out.loc[matched, "comparison_weight"] = 1. / count
    out["ground_threshold_fraction"] = ground_threshold
    out["canopy_threshold_kg_m2"] = canopy_threshold
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--lookup", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ground-threshold", type=float, default=0.5)
    p.add_argument("--canopy-threshold", type=float, default=0.5)
    a = p.parse_args()
    if not 0 <= a.ground_threshold <= 1:
        p.error("--ground-threshold must be a fraction between 0 and 1")
    if not np.isfinite(a.canopy_threshold) or a.canopy_threshold < 0:
        p.error("--canopy-threshold must be finite and nonnegative")

    with xr.open_dataset(a.lookup) as lookup, xr.open_dataset(a.model) as model:
        result = compare(
            pd.read_parquet(a.observations), lookup, model,
            a.ground_threshold, a.canopy_threshold,
        )
    result["model_file"] = str(a.model.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(".partial.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(a.output)

    matched = result[result.matched]
    if len(matched):
        agreement = np.average(
            (matched.model_class == matched.icesat_class).astype(float),
            weights=matched.comparison_weight,
        )
        print(
            f"Matched {len(matched):,}/{len(result):,}; "
            f"cell/overpass-weighted class agreement: {agreement:.1%}"
        )
    else:
        print("No matched observations with valid radiometry and model output")
    print(a.output)
