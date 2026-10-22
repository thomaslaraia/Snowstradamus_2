#!/usr/bin/env python3
"""Download and retain native NASA ABoVE 2020 canopy-cover GeoTIFFs.

Install: pip install earthaccess rasterio
Run from the same directory as your climate script to reuse its relative paths:
    python -u download_canopy_cover.py --area 73 -180 45 180

Completed files are retained across restarts; an interrupted tile is downloaded
again. Failed tiles are retried in later passes, indefinitely by default.
The original climate script can run concurrently. Its shared inventory.json
and tcc_temporary folder are not modified by this downloader.
"""

import argparse
import fcntl
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit

TCC_DOI = "10.3334/ORNLDAAC/2012"
TCC_RESOLUTION = 0.00025
LOG = logging.getLogger("canopy")


def valid_tile(path):
    """Check the native grid and read all raster blocks to detect truncation."""
    if not path.is_file():
        return False
    import rasterio
    try:
        with rasterio.open(path) as src:
            if not (src.crs and src.crs.to_epsg() == 4326 and src.count == 1
                    and abs(src.res[0] - TCC_RESOLUTION) < 1e-6
                    and abs(src.res[1] - TCC_RESOLUTION) < 1e-6):
                return False
            for _, window in src.block_windows(1):
                src.read(1, window=window)  # Bounded memory, even for large tiles.
        return True
    except (OSError, rasterio.errors.RasterioError):
        return False


def tile_names(granule):
    names = set()
    for link in granule.data_links():
        name = unquote(Path(urlsplit(link).path).name)
        if name.lower().endswith("_tcc_2020.tif"):
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Unexpected filename in NASA search result")
            names.add(name)
    return names


def discover(area):
    import earthaccess
    n, w, s, e = area
    bounds = (max(-180, w - 0.05), max(-90, s - 0.05),
              min(180, e + 0.05), min(90, n + 0.05))
    LOG.info("Searching NASA for 2020 canopy tiles within %s", bounds)
    granules = earthaccess.search_data(
        doi=TCC_DOI, bounding_box=bounds,
        temporal=("2020-01-01", "2020-12-31"))
    return [(g, names) for g in granules if (names := tile_names(g))]


def run(args):
    import earthaccess
    cache = args.tcc_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    LOG.info("Saving completed tiles to %s", cache)
    # This lock prevents two copies of this standalone downloader sharing a cache.
    with (cache / ".canopy_downloader.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another canopy downloader is using this folder")

        auth = earthaccess.login()  # Uses the same Earthdata login as your script.
        if not auth.authenticated:
            raise RuntimeError("NASA Earthdata login failed")

        attempt = 0
        while True:
            attempt += 1
            try:
                matches = discover(args.area)
                break
            except Exception as exc:
                LOG.warning("Search attempt %d failed: %s", attempt, exc)
                if args.max_passes and attempt >= args.max_passes:
                    raise
                time.sleep(args.retry_delay)
        if not matches:
            raise RuntimeError("NASA returned no 2020 canopy tiles for this area")

        expected = set().union(*(names for _, names in matches))
        LOG.info("Found %d unique tile(s); checking existing files", len(expected))
        completed = set()
        for name in sorted(expected):
            if valid_tile(cache / name):
                completed.add(name)
                LOG.info("Reusing %s [%d/%d]", name, len(completed), len(expected))

        pass_number = 0
        while completed != expected:
            pass_number += 1
            LOG.info("Pass %d: %d tile(s) remaining", pass_number,
                     len(expected - completed))
            for granule, names in matches:
                missing = names - completed
                if not missing:
                    continue
                # The climate process may have completed these since our search.
                for name in sorted(missing):
                    if valid_tile(cache / name):
                        completed.add(name)
                        LOG.info("Reusing %s", name)
                missing = names - completed
                if not missing:
                    continue
                started = time.monotonic()
                try:
                    # Same filesystem permits atomic publication of each full tile.
                    # Unique directories avoid the climate script's temporary files.
                    with tempfile.TemporaryDirectory(
                            prefix=".canopy_download_", dir=cache) as folder:
                        LOG.info("Downloading %s", ", ".join(sorted(missing)))
                        paths = earthaccess.download([granule], folder)
                        files = {Path(p).name: Path(p) for p in paths}
                        for name in sorted(missing):
                            tile = files.get(name)
                            if tile is None or not valid_tile(tile):
                                LOG.warning("Missing or unreadable tile: %s", name)
                                continue
                            with tile.open("rb") as handle:
                                os.fsync(handle.fileno())
                            tile.replace(cache / name)  # Appears only when complete.
                            completed.add(name)
                            LOG.info("Saved %s [%d/%d], %.1f s", name,
                                     len(completed), len(expected),
                                     time.monotonic() - started)
                except Exception as exc:
                    LOG.warning("Download failed; will retry: %s", exc)

            if completed != expected:
                if args.max_passes and pass_number >= args.max_passes:
                    LOG.error("Still missing: %s", ", ".join(sorted(expected - completed)))
                    return 1
                LOG.info("Waiting %.0f seconds before retrying %d missing tile(s)",
                         args.retry_delay, len(expected - completed))
                time.sleep(args.retry_delay)

        LOG.info("Complete: all %d tiles found by NASA are available in %s",
                 len(expected), cache)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", type=float, nargs=4, required=True,
                        metavar=("N", "W", "S", "E"))
    parser.add_argument("--tcc-dir", type=Path,
                        default=Path("../../scratch/climate_site_selection/tcc"))
    parser.add_argument("--retry-delay", type=float, default=60,
                        help="Seconds between retry passes (default: 60)")
    parser.add_argument("--max-passes", type=int, default=0,
                        help="Maximum search attempts and download passes; 0 retries indefinitely")
    args = parser.parse_args()
    n, w, s, e = args.area
    if not (all(math.isfinite(v) for v in args.area)
            and -90 <= s < n <= 90 and -180 <= w < e <= 180):
        parser.error("Use N W S E bounds with N > S and E > W; no dateline crossing")
    if not math.isfinite(args.retry_delay) or args.retry_delay <= 0 or args.max_passes < 0:
        parser.error("Retry delay must be positive and max passes nonnegative")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.info("Stopped. Restart with the same command to reuse completed files.")
        return 130
    except Exception as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
