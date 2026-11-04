#!/usr/bin/env python3
"""Process remote overpasses and publish one Parquet file per candidate."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import re
import pyarrow.parquet as pq
import pandas as pd
import yaml
import remote_batch_core as remote


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temp, path)


def publish_candidate(parts, target):
    """Combine checkpoint files without loading the whole candidate into RAM."""
    temporary = target.with_suffix('.tmp')
    writer = None
    try:
        for part in sorted(parts.glob('*.parquet')):
            with pq.ParquetFile(part) as source:
                for batch in source.iter_batches(batch_size=100_000):
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, batch.schema, compression='zstd')
                    writer.write_batch(batch)
        if writer is not None:
            writer.close()
            writer = None
        else:
            pd.DataFrame(columns=['site_id', 'resource']).astype('string').to_parquet(
                temporary, index=False, engine='pyarrow')
        os.replace(temporary, target)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sites', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--config', type=Path, default=Path('../config.yaml'))
    ap.add_argument('--start', default='2018-10-01T00:00:00Z')
    ap.add_argument('--end', default=None, help='Default: UTC now; only available release-007 resources are discovered')
    ap.add_argument('--length', type=float, default=30)
    ap.add_argument('--box-side-km', type=float, default=None,
                    help='Full side length in km; otherwise use YAML small_box_half_side')
    ap.add_argument('--site-indices', type=int, nargs='+')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = yaml.safe_load(args.config.read_text())
    cfg['parallel_blocks'].update(rebinned=args.length, DW=False,
        trim_atmospheric=True, sat_flag=True)
    # Whole-beam atmospheric thresholds no longer affect processing or its signature.
    cfg['parallel_blocks'].pop('msw_flag_threshold', None)
    cfg['parallel_blocks'].pop('layer_flag_threshold', None)
    if args.box_side_km is not None:
        if args.box_side_km <= 0:
            raise ValueError('--box-side-km must be positive')
        cfg['parallel_blocks']['small_box_half_side'] = args.box_side_km / 2
    sites = remote.load_sites(args.sites)
    if args.site_indices is not None:
        sites = sites.iloc[args.site_indices]
    start = pd.Timestamp(args.start)
    start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    end = pd.Timestamp(args.end or datetime.now(timezone.utc))
    end = end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    if start >= end:
        raise ValueError('Start must precede end')
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signature = dict(config=cfg,
        core_sha256=hashlib.sha256(Path(remote.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logging.info('Candidate Parquet files: %s', root)
    from sliderule import sliderule, earthdata
    sliderule.init(verbose=False, rethrow=True)
    provider = remote.public_provider()
    failures = 0
    for site in sites.itertuples(index=False):
        site_id = str(site.site_id)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', site_id):
            raise ValueError(f'Unsafe site_id for filename: {site_id!r}')
        state = root/'_checkpoints'/site_id
        for sub in ['data', 'status', 'errors', 'discovery']:
            (state/sub).mkdir(parents=True, exist_ok=True)
        manifest = state/'manifest.json'
        current = dict(signature, bounds=[site.west, site.south, site.east, site.north])
        if (manifest.exists() and json.loads(manifest.read_text()) != current
                and (any((state/'status').glob('*.json')) or any((state/'data').glob('*.parquet')))):
            raise ValueError(f'{site_id}: settings/source/bounds changed; use a fresh output directory '
                             'or remove this candidate checkpoint folder and Parquet to reprocess')
        atomic_json(manifest, current)
        candidate_target = root/f'{site_id}.parquet'
        # Recover publication after an interruption; checkpoints remain authoritative.
        if any((state/'data').glob('*.parquet')):
            publish_candidate(state/'data', candidate_target)
        # CMR only needs the footprint boundary; retain the buffered/dense
        # polygon inside process_resource for photon retrieval.
        polygon = [dict(lon=x, lat=y) for x, y in [
            (site.west, site.south), (site.east, site.south),
            (site.east, site.north), (site.west, site.north),
            (site.west, site.south)]]
        discovery_ok = 0
        site_failures = 0
        cursor = start
        while cursor < end:
            stop = min(cursor.normalize().replace(day=1) + pd.offsets.MonthBegin(1), end)
            tag = f'{cursor:%Y%m%dT%H%M%S}_{stop:%Y%m%dT%H%M%S}'
            try:
                # Re-query each month on restart to pick up newly available data.
                found = earthdata.cmr(short_name='ATL03', version='007', polygon=polygon,
                    time_start=cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    time_end=stop.strftime("%Y-%m-%dT%H:%M:%SZ"))
                discovery_ok += 1
                logging.info("%s %s: discovered %d resources", site_id, cursor, len(found))
                latest = {}
                for resource in sorted(set(found)):
                    info = remote.granule_info(resource)
                    when = info['time'].tz_localize('UTC')
                    if start <= when < end:
                        latest[resource.rsplit('_', 1)[0]] = resource
                atomic_json(state/'discovery'/f'{tag}.json', dict(site_id=site.site_id,
                    start=cursor, end=stop, resources=list(latest.values())))
            except Exception:
                failures += 1
                site_failures += 1
                logging.exception('Discovery failed for %s %s; retry by resubmitting', site.site_id, cursor)
                cursor = stop
                continue
            for resource in sorted(latest.values()):
                # Acquisition key replaces earlier revisions instead of duplicating them.
                acquisition = resource.rsplit('_', 1)[0]
                token = acquisition
                done = state/'status'/f'{token}.json'
                target = state/'data'/f'{token}.parquet'
                if done.exists():
                    old = json.loads(done.read_text())
                    if old['resource'] == resource and (old['rows'] == 0 or target.exists()):
                        continue
                logging.info('%s %s', site.site_id, resource)
                try:
                    run = remote.process_resource(site, resource, cfg, provider, graph_detail=0)
                    # A missing database acquisition can be ingestion lag, so don't checkpoint it.
                    if run.diagnostics.empty or run.diagnostics.status.eq('no_atl08').any():
                        raise RuntimeError('Missing ATL08 beam records; leave acquisition pending for retry')
                    frame = run.results.copy()
                    if len(frame):
                        frame['file_index'] = int(hashlib.sha256(token.encode()).hexdigest()[:13], 16)  # Stable numeric ID, exact in float64 too.
                        frame['ATL03'] = resource
                        frame['ATL08'] = pd.Series(pd.NA, index=frame.index, dtype='string')
                        frame['atl08_source'] = 'icesat2db ATL08 release 007'
                        # Normalize exploded object columns for a consistent directory-wide schema.
                        strings = {'folder','date','beam_str','resource','site_id','ATL03','ATL08',
                                   'atl08_source','atl08_segment_id'}
                        for col in frame:
                            frame[col] = frame[col].astype('string') if col in strings else pd.to_numeric(frame[col], errors='raise').astype('float64')
                        temporary = target.with_suffix('.tmp')
                        frame.to_parquet(temporary, index=False, engine='pyarrow', compression='zstd')
                        os.replace(temporary, target)
                    elif target.exists():
                        target.unlink()  # New revision supersedes the previous output.
                    atomic_json(done, dict(site_id=site.site_id, resource=resource, rows=len(frame),
                        completed=datetime.now(timezone.utc), diagnostics=run.diagnostics.to_dict('records')))
                    logging.info('Saved %d rows', len(frame))
                    del run, frame
                    gc.collect()
                except Exception as exc:
                    failures += 1
                    site_failures += 1
                    logging.exception('Failed; remains pending for next submission')
                    atomic_json(state/'errors'/f'{token}.json', dict(resource=resource,
                        site_id=site.site_id, error=str(exc), time=datetime.now(timezone.utc)))
            publish_candidate(state/'data', candidate_target)
            cursor = stop
        if not discovery_ok:
            logging.error('%s: every discovery query failed; no new data retrieved', site_id)
        elif site_failures:
            logging.error('%s: partial result at %s; failed requests remain pending', site_id, candidate_target)
        else:
            logging.info('Candidate saved: %s', candidate_target)
    if failures:
        raise RuntimeError(f'{failures} requests failed; rerun to retry. Existing checkpoints are retained.')
    logging.info('Finished. Load candidate Parquet files from %s', root)

if __name__ == '__main__':
    main()
