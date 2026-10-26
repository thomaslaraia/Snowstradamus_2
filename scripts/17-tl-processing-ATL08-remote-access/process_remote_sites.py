#!/usr/bin/env python3
"""Stream site/overpass regression rows to resumable Parquet partitions."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import pandas as pd
import yaml
import remote_batch_core as remote


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sites', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--config', type=Path, default=Path('../config.yaml'))
    ap.add_argument('--start', default='2018-10-01T00:00:00Z')
    ap.add_argument('--end', default=None, help='Default: UTC now; only available release-007 resources are discovered')
    ap.add_argument('--length', type=float, default=30)
    ap.add_argument('--msw-threshold', type=float, default=1.0)
    ap.add_argument('--layer-threshold', type=float, default=1.0)
    ap.add_argument('--site-indices', type=int, nargs='+')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = yaml.safe_load(args.config.read_text())
    cfg['parallel_blocks'].update(rebinned=args.length, DW=False,
        msw_flag_threshold=args.msw_threshold, layer_flag_threshold=args.layer_threshold)
    sites = remote.load_sites(args.sites)
    if args.site_indices is not None:
        sites = sites.iloc[args.site_indices]
    start = pd.Timestamp(args.start)
    start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    end = pd.Timestamp(args.end or datetime.now(timezone.utc))
    end = end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    if start >= end:
        raise ValueError('Start must precede end')
    # Config and source changes get separate output trees; extending dates resumes the same tree.
    signature = dict(config=cfg, sites=sites[['site_id','west','south','east','north']].to_dict('records'),
        core_sha256=hashlib.sha256(Path(remote.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()[:16]
    root = args.output / key
    for sub in ['data', 'status', 'errors', 'discovery']:
        (root/sub).mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Prevent two jobs writing this dataset.
    atomic_json(root/'manifest.json', signature)
    logging.info('Dataset: %s', root)
    from sliderule import sliderule, earthdata
    sliderule.init(verbose=False, rethrow=True)
    provider = remote.public_provider()
    for site in sites.itertuples(index=False):
        polygon, _ = remote.query_polygon(site, max(300, 2*args.length))
        cursor = start
        while cursor < end:
            stop = min(cursor.normalize().replace(day=1) + pd.offsets.MonthBegin(1), end)
            tag = hashlib.sha256(f'{site.site_id}|{cursor}|{stop}'.encode()).hexdigest()[:24]
            try:
                # Re-query each month on restart to pick up newly available data.
                found = earthdata.cmr(short_name='ATL03', version='007', polygon=polygon,
                    time_start=cursor.isoformat(), time_end=stop.isoformat())
                latest = {}
                for resource in sorted(set(found)):
                    info = remote.granule_info(resource)
                    when = info['time'].tz_localize('UTC')
                    if start <= when < end:
                        latest[resource.rsplit('_', 1)[0]] = resource
                atomic_json(root/'discovery'/f'{tag}.json', dict(site_id=site.site_id,
                    start=cursor, end=stop, resources=list(latest.values())))
            except Exception:
                logging.exception('Discovery failed for %s %s; retry by resubmitting', site.site_id, cursor)
                cursor = stop
                continue
            for resource in sorted(latest.values()):
                # Acquisition key replaces earlier revisions instead of duplicating them.
                acquisition = resource.rsplit('_', 1)[0]
                token = hashlib.sha256(f'{site.site_id}|{acquisition}'.encode()).hexdigest()[:24]
                done = root/'status'/f'{token}.json'
                target = root/'data'/f'{token}.parquet'
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
                        frame['file_index'] = int(token[:13], 16)  # Stable numeric ID, exact in float64 too.
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
                    logging.exception('Failed; remains pending for next submission')
                    atomic_json(root/'errors'/f'{token}.json', dict(resource=resource,
                        site_id=site.site_id, error=str(exc), time=datetime.now(timezone.utc)))
            cursor = stop
    logging.info('Finished. Load Parquet files from %s', root/'data')

if __name__ == '__main__':
    main()
