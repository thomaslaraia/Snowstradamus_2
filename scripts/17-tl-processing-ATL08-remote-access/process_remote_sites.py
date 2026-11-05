#!/usr/bin/env python3
"""Process remote overpasses and publish one Parquet file per candidate."""
import argparse
from contextlib import redirect_stdout
import sys
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import re
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import yaml
import remote_batch_core as remote


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temp, path)


def load_file_indices(status_dir):
    """Assign a completion-order index to legacy checkpoints."""
    records = [(path, json.loads(path.read_text())) for path in status_dir.glob('*.json')]
    records.sort(key=lambda item: (item[1]['completed'], item[0].stem))
    next_index = max((record.get('file_index', 0) for _, record in records), default=0) + 1
    indices = {}
    for path, record in records:
        if 'file_index' not in record:
            record['file_index'] = next_index
            next_index += 1
            atomic_json(path, record)
        indices[path.stem] = record['file_index']
    return indices, next_index


def publish_candidate(parts, target, indices):
    """Combine checkpoint files without loading the whole candidate into RAM."""
    temporary = target.with_suffix('.tmp')
    writer = None
    try:
        for part in sorted(parts.glob('*.parquet')):
            if part.stem not in indices:
                continue  # The overpass was interrupted before its completion record.
            with pq.ParquetFile(part) as source:
                for batch in source.iter_batches(batch_size=100_000):
                    column = batch.schema.get_field_index('file_index')
                    if column >= 0:
                        values = pa.repeat(pa.scalar(float(indices[part.stem])), batch.num_rows)
                        batch = batch.set_column(column, 'file_index', values)
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


def print_beam_summary(site_id, file_index, resource, frame):
    date = remote.granule_info(resource)['time'].date()
    if frame.empty:
        print(f'{site_id} #{file_index} {date} | no segments saved', flush=True)
    else:
        segments = frame.drop_duplicates(['beam', 'bin_id'])
        for beam, group in segments.groupby('beam', sort=True):
            print(
                f'{site_id} #{file_index} {date} | beam {int(beam)} | '
                f'{len(group):,} segments | '
                f'mean msw={group["msw_flag"].mean():.2f} '
                f'layer={group["layer_flag"].mean():.2f}',
                flush=True,
            )


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
    ap.add_argument('--resume-from-runner-sha256',
                    help='Accept this prior runner hash if the config, core and bounds match')
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
        # Accept checkpoints from earlier runners with matching processing settings.
        compatible_legacy = dict(current, runner_sha256=
            '7ffb362215f44d868eeaeb41ad4b348ac413ff0049c5e7c9260e35fadd0ba2a0')
        compatible_numbered = dict(current, runner_sha256='8583426327c74008290bc5a94f02dfa2bfa9d9bba2082c6724ac56d2ab405556')
        compatible_partial_beams = dict(current, runner_sha256='a1d60a5126c2bdae8afa45dfcbb96b93580990f772ef7eef7b9d0239786b5de6')
        compatible_beam_log = dict(current, runner_sha256='35a6a5002c3911d1787e987b1f55bcfe333211e21eb539ee59bc54f080f8ab2c')
        compatible_replay_log = dict(current, runner_sha256='8fb0bce0fc6fe9b890f24e8cdea566644dea5d4bc48111e4238e39ca1b2b6a51')
        compatible = [current, compatible_legacy, compatible_numbered,
                      compatible_partial_beams, compatible_beam_log, compatible_replay_log]
        compatible.append(dict(current, runner_sha256='798ee903b1832f126359709bc3b47634db0e80ac509d417d83c17fb6979a06f4'))
        if args.resume_from_runner_sha256:
            compatible.append(dict(current, runner_sha256=args.resume_from_runner_sha256))
        # The core change only replaces the filename in a diagnostic.
        compatible.append(dict(current, runner_sha256='c3c0b3ea1944cb73573eb1498806daa3e5116633835ed101756351cb3aa7985b'))
        if current['core_sha256'] == 'baf51349253fa300211fb7c936b853916e29b64badd026d4b66d5b881bf37033':
            compatible.extend(dict(entry, core_sha256='5452402d8c4c616fd4b4a4ca59fd4eb104244c03145c4b7c82c3ab7df716bee0')
                              for entry in list(compatible))
        if manifest.exists() and (any((state/'status').glob('*.json'))
                                  or any((state/'data').glob('*.parquet'))):
            saved_manifest = json.loads(manifest.read_text())
            if saved_manifest not in compatible:
                changed = [key for key in sorted(current.keys() | saved_manifest.keys())
                           if current.get(key) != saved_manifest.get(key)]
                message = f'{site_id}: checkpoint mismatch in {", ".join(changed)}.'
                if changed == ['runner_sha256'] and saved_manifest.get('runner_sha256'):
                    message += (
                        ' If your prior runner differs only in logging or numbering, '
                        'resume with --resume-from-runner-sha256 '
                        + saved_manifest['runner_sha256']
                    )
                raise ValueError(message)
        indices, next_index = load_file_indices(state/'status')
        atomic_json(manifest, current)
        logging.info('%s: %d overpasses already processed', site_id, len(indices))
        if indices:
            print(f'{site_id} | previously completed overpasses', flush=True)
        for token, file_index in sorted(indices.items(), key=lambda item: item[1]):
            record = json.loads((state/'status'/f'{token}.json').read_text())
            part = state/'data'/f'{token}.parquet'
            if record['rows'] == 0:
                saved = pd.DataFrame()
            elif part.exists():
                saved = pd.read_parquet(part, columns=['beam', 'bin_id', 'msw_flag', 'layer_flag'])
            else:
                continue  # The normal processing loop retries missing output files.
            print_beam_summary(site_id, file_index, record['resource'], saved)
            del saved
        candidate_target = root/f'{site_id}.parquet'
        # Recover publication after an interruption; checkpoints remain authoritative.
        if any((state/'data').glob('*.parquet')):
            publish_candidate(state/'data', candidate_target, indices)
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
                file_index = indices.get(token, next_index)
                logging.info('%s overpass #%d: %s', site_id, file_index, resource)
                try:
                    # Keep the stdout log for saved results and send detailed diagnostics to stderr.
                    with redirect_stdout(sys.stderr):
                        run = remote.process_resource(site, resource, cfg, provider, graph_detail=0)
                    # A beam can miss the site entirely. Keep results from the beams that returned records.
                    if run.diagnostics.empty or run.diagnostics.status.eq('no_atl08').all():
                        raise RuntimeError('No ATL08 records for any requested beam; retry on resubmission')
                    frame = run.results.copy()
                    if len(frame):
                        frame['file_index'] = file_index
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
                    atomic_json(done, dict(site_id=site.site_id, resource=resource, rows=len(frame), file_index=file_index,
                        completed=datetime.now(timezone.utc), diagnostics=run.diagnostics.to_dict('records')))
                    if token not in indices:
                        indices[token] = file_index
                        next_index += 1
                    logging.info('%s: %d overpasses processed (%d rows for #%d)',
                                 site_id, len(indices), len(frame), file_index)
                    print_beam_summary(site_id, file_index, resource, frame)
                    del run, frame
                    gc.collect()
                except Exception as exc:
                    failures += 1
                    site_failures += 1
                    logging.exception('Failed; remains pending for next submission')
                    atomic_json(state/'errors'/f'{token}.json', dict(resource=resource,
                        site_id=site.site_id, error=str(exc), time=datetime.now(timezone.utc)))
            publish_candidate(state/'data', candidate_target, indices)
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

