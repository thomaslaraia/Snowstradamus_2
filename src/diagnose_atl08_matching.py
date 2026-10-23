"""Explain ATL03-bin/icesat2db matching failures without changing the pipeline.

Put this file beside remote_parallel_blocks.py (or on the same Python path).
In a notebook:
    from diagnose_atl08_matching import diagnose_sites, plot_matching
    for report in diagnose_sites(selected.iloc[[0]], START_TIME, END_TIME,
                                  length=30, max_resources=3, cfg=remote_cfg):
        display(report.summary)
        display(report.bin_reasons)
        display(report.by_class)
        display(report.field_comparison)
        plot_matching(report)

All outputs stay in memory. No regression or atmospheric filtering is applied.
The default source check reads a small selection of 1D ATL08 land-segment fields
with SlideRule h5p, not the complete HDF5 file. Disable with compare_source=False.
Source-check errors are reported as errors, never as evidence of missing data.
Exact ATL08 revision used internally by atl03x is not guaranteed; the separately
checked ATL08 resource is reported explicitly. No geographic nearest-neighbour
replacement or time-tolerance change is applied to the production results.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.remote_parallel_blocks as pipeline

TIME_TOLERANCE_S = 1e-6  # Exactly the current production match tolerance.
SOURCE_FIELDS = {
    'segment_id_beg': 'segment_id_beg', 'segment_id_end': 'segment_id_end',
    'delta_time_beg': 'delta_time_beg', 'delta_time_end': 'delta_time_end',
    'longitude': 'longitude', 'latitude': 'latitude',
    'h_te_best_fit': 'terrain/h_te_best_fit',
    'h_te_uncertainty': 'terrain/h_te_uncertainty',
    'h_te_median': 'terrain/h_te_median',
    'h_canopy': 'canopy/h_canopy',
    'h_canopy_uncertainty': 'canopy/h_canopy_uncertainty',
    'urban_flag': 'urban_flag', 'segment_watermask': 'segment_watermask',
    'msw_flag': 'msw_flag', 'layer_flag': 'layer_flag',
}
REASON_ORDER = ['matched', 'no_parents', 'before_first_parent',
                'after_last_parent', 'segment_gap', 'invalid_parent_time',
                'before_parent_time', 'after_parent_time']


@dataclass
class MatchingReport:
    summary: pd.DataFrame
    bin_reasons: pd.DataFrame
    by_class: pd.DataFrame
    bins: pd.DataFrame
    parent_comparison: pd.DataFrame
    field_comparison: pd.DataFrame
    query_audit: pd.DataFrame
    examples: pd.DataFrame


def _inside(table, bounds):
    west, south, east, north = bounds
    return (table.longitude.between(west, east) & table.latitude.between(south, north))


def _percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else np.nan


def _tag(table, site, resource, beam=None):
    table = table.copy()
    table['site_id'], table['resource'] = str(site.site_id), resource
    if beam is not None:
        table['beam'] = beam
    return table


def _concat(tables):
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def fetch_database_with_audit(provider, geometry, resource):
    """Reproduce production query and identity filters, retaining their counts."""
    info = pipeline.granule_info(resource)
    available = set(provider.get_available_variables().index)
    missing = set(pipeline.DB_FIELDS) - available
    if missing:
        raise ValueError(f'Missing database fields: {sorted(missing)}')
    fields = pipeline.DB_FIELDS + (['source_granule'] if 'source_granule' in available else [])
    day = info['time'].normalize()
    t = provider.get_data(variables=fields, geometry=geometry,
        start_time=str(day), end_time=str(day+pd.Timedelta(days=2)),
        query_type='bounding_box', return_type='dataframe')
    stages = []
    if t is None or len(t) == 0:
        return pd.DataFrame(), pd.DataFrame([dict(stage='database_query', rows=0)])
    t = t.copy()
    stages.append(dict(stage='database_query', rows=len(t)))
    packed = np.array([int(v) for v in t.segment_id], dtype=np.int64)
    rgt, cycle, beam = packed >> 43, (packed >> 35) & 255, (packed >> 32) & 7
    starts = packed & ((1 << 32)-1)
    if ((beam > 5).any() or not np.array_equal(starts, t.segment_id_beg.to_numpy())
            or not np.array_equal(rgt, t.rgt.to_numpy())):
        raise ValueError('Database packed IDs disagree with the identity fields.')
    t['beam'] = np.array(pipeline.BEAMS)[beam]
    t['db_segment_id'] = packed
    stages.append(dict(stage='correct_rgt', rows=int((rgt == info['rgt']).sum())))
    t = t[(rgt == info['rgt']) & (cycle == info['cycle'])].copy()
    stages.append(dict(stage='correct_rgt_and_cycle', rows=len(t)))
    if 'source_granule' in t:
        prefix = resource.replace('ATL03_', 'ATL08_').rsplit('_', 1)[0]+'_'
        t = t[t.source_granule.astype(str).str.startswith(prefix)].copy()
        stages.append(dict(stage='source_filename_filter', rows=len(t)))
    if t.duplicated(['beam', 'segment_id_beg']).any():
        raise ValueError('Duplicate database parents; resolve revisions before matching.')
    for field in pipeline.ANCILLARY:
        t[field] = pd.to_numeric(t[field], errors='raise')
        t.loc[t[field].abs() >= 1e30, field] = np.nan
    for name in pipeline.BEAMS:
        stages.append(dict(stage=f'parents_{name}', rows=int(t.beam.eq(name).sum())))
    return t, pd.DataFrame(stages)


def resolve_source_resource(resource):
    """Discover an ATL08 companion rather than assume its revision matches ATL03."""
    from sliderule import earthdata
    info = pipeline.granule_info(resource)
    prefix = resource.replace('ATL03_', 'ATL08_').rsplit('_', 1)[0]+'_'
    day = info['time'].normalize()
    found = earthdata.cmr(short_name='ATL08', version='007',
        time_start=day.strftime('%Y-%m-%dT%H:%M:%SZ'),
        time_end=(day+pd.Timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        name_filter=prefix+'*')
    found = sorted({x for x in found if x.startswith(prefix) and x.endswith('.h5')})
    if not found:
        raise RuntimeError(f'No ATL08 companion found by CMR for {resource}')
    # Explicitly report the selected latest companion and number of candidates.
    return found[-1], len(found)


def fetch_source_parents(resource08, beam):
    """Read only the listed land-segment columns, including invalid source rows."""
    from sliderule import h5
    paths = {name: f'/{beam}/land_segments/{relative}' for name, relative in SOURCE_FIELDS.items()}
    raw = h5.h5p([{'dataset': path} for path in paths.values()], resource08, asset='icesat2')
    if raw is None or not all(path in raw for path in paths.values()):
        absent = [name for name, path in paths.items() if raw is None or path not in raw]
        raise RuntimeError(f'Incomplete remote ATL08 read: {absent}')
    arrays = {name: np.asarray(raw[path]).reshape(-1) for name, path in paths.items()}
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError('Source ATL08 fields have inconsistent lengths.')
    result = pd.DataFrame(arrays)
    if result.empty:
        raise RuntimeError('Remote ATL08 read returned no parent rows; coverage not established.')
    result['beam'], result['source_resource'] = beam, resource08
    return result


def match_stages(photons, parents):
    """Audit exact production rules, exposing the reason for every failed photon."""
    n = len(photons)
    out = pd.DataFrame(index=np.arange(n))
    out['reason'] = 'no_parents'
    out['id_match'], out['strict_match'] = False, False
    for col in ['parent_start', 'parent_end', 'parent_time_start', 'parent_time_end',
                'time_outside_s', 'nearest_segment_gap']:
        out[col] = np.nan
    if parents.empty:
        return out
    p = parents.sort_values('segment_id_beg').reset_index(drop=True)
    starts, ends = p.segment_id_beg.to_numpy(), p.segment_id_end.to_numpy()
    if (not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or (starts > ends).any() or (starts[1:] <= ends[:-1]).any()):
        raise ValueError('Invalid or overlapping parent segment-ID intervals.')
    ids, times = photons.segment_id.to_numpy(), photons.delta_time.to_numpy()
    loc = np.searchsorted(starts, ids, side='right')-1
    safe = np.maximum(loc, 0)
    inside = (loc >= 0) & (ids <= ends[safe])
    begin, end = p.delta_time_beg.to_numpy()[safe], p.delta_time_end.to_numpy()[safe]
    time_valid = np.isfinite(begin) & np.isfinite(end) & (abs(begin) < 1e30) & (abs(end) < 1e30) & (begin <= end)
    # strict_match is the exact original predicate, including its fill handling.
    strict = inside & (times >= begin-TIME_TOLERANCE_S) & (times <= end+TIME_TOLERANCE_S)
    reason = np.full(n, 'segment_gap', dtype=object)
    reason[ids < starts[0]], reason[ids > ends[-1]] = 'before_first_parent', 'after_last_parent'
    reason[inside & ~time_valid] = 'invalid_parent_time'
    reason[inside & time_valid & (times < begin-TIME_TOLERANCE_S)] = 'before_parent_time'
    reason[inside & time_valid & (times > end+TIME_TOLERANCE_S)] = 'after_parent_time'
    reason[strict] = 'matched'
    out['reason'], out['id_match'], out['strict_match'] = reason, inside, strict
    for name, values in [('parent_start', starts[safe]), ('parent_end', ends[safe]),
                         ('parent_time_start', begin), ('parent_time_end', end)]:
        out[name] = np.where(inside, values, np.nan)
    out['time_outside_s'] = np.where(inside & time_valid,
        np.maximum(np.maximum(begin-times, times-end), 0.), np.nan)
    before_gap = np.where(loc >= 0, np.maximum(ids-ends[safe], 0), np.inf)
    next_loc = np.minimum(loc+1, len(starts)-1)
    after_gap = np.where(loc+1 < len(starts), np.maximum(starts[next_loc]-ids, 0), np.inf)
    out['nearest_segment_gap'] = np.where(inside, 0., np.minimum(before_gap, after_gap))
    return out


def compare_parent_fields(db, source, min_id, max_id, query_bounds):
    """Compare parent membership and values, preserving source fill values."""
    src = source[(source.segment_id_end >= min_id) & (source.segment_id_beg <= max_id)].copy()
    db = db[(db.segment_id_end >= min_id) & (db.segment_id_beg <= max_id)].copy() if len(db) else db.copy()
    if db.empty:
        db = pd.DataFrame(columns=['segment_id_beg', *[c for c in SOURCE_FIELDS if c != 'segment_id_beg']])
    src['source_center_in_query'] = _inside(src, query_bounds)
    src = src.rename(columns={c:c+'_source' for c in src if c not in ['segment_id_beg','source_center_in_query']})
    db = db.rename(columns={c:c+'_db' for c in db if c != 'segment_id_beg'})
    joined = src.merge(db, on='segment_id_beg', how='outer',
                       indicator='membership', validate='one_to_one')
    joined['membership'] = joined.membership.astype(str).replace(
        {'left_only':'source_only', 'right_only':'database_only', 'both':'both'})
    for field, label in [('h_te_best_fit','source_invalid_terrain_height'),
                          ('h_canopy','source_invalid_canopy_height')]:
        if field+'_source' in joined:
            values = pd.to_numeric(joined[field+'_source'],errors='coerce')
            joined[label] = (~np.isfinite(values) | (values.abs()>=1e30) | (values<=-999)) & joined.membership.ne('database_only')
    stats = []
    shared = joined[joined.membership.eq('both')]
    for field in ['segment_id_end', 'delta_time_beg', 'delta_time_end',
                  'longitude', 'latitude', 'h_te_best_fit', 'msw_flag', 'layer_flag']:
        a, b = field+'_source', field+'_db'
        if a not in joined or b not in joined:
            continue
        old = pd.to_numeric(shared[a], errors='coerce')
        new = pd.to_numeric(shared[b], errors='coerce')
        valid = np.isfinite(old) & np.isfinite(new) & (old.abs()<1e30) & (new.abs()<1e30)
        difference = new[valid]-old[valid]
        stats.append(dict(field=field, compared=int(valid.sum()),
            exact_mismatches=int(difference.ne(0).sum()),
            median_db_minus_source=difference.median(),
            max_absolute_difference=difference.abs().max()))
    return joined, pd.DataFrame(stats)


def audit_beam(photons, parents, site, length=30, *, source=None, cfg=None,
               query_bounds=None, max_examples=5):
    """Offline diagnostic core; no service calls and no production data mutation."""
    if length <= 0 or not np.isfinite(length):
        raise ValueError('length must be positive.')
    f = photons.reset_index(drop=True).copy()
    if f.empty or f[['resource', 'beam']].drop_duplicates().shape[0] != 1:
        raise ValueError('Supply photons from one nonempty resource/beam.')
    resource, beam = f.resource.iloc[0], f.beam.iloc[0]
    if len(parents) and (parents.beam.nunique() != 1 or parents.beam.iloc[0] != beam):
        raise ValueError('Database parent beam differs from photon beam.')
    if source is not None and len(source) and (source.beam.nunique() != 1 or source.beam.iloc[0] != beam):
        raise ValueError('Source parent beam differs from photon beam.')
    if not np.isfinite(f[['x_atc', 'segment_id', 'delta_time']]).all().all():
        raise ValueError('Invalid photon distance, ID or raw time.')
    bounds = (site.west, site.south, site.east, site.north)
    query_bounds = bounds if query_bounds is None else query_bounds
    f['bin_id'] = np.floor(f.x_atc/length).astype('int64')
    f['signal'] = f.classification.isin([1,2,3])
    f['distance_to_center'] = abs(f.x_atc-(f.bin_id+.5)*length)
    staged = match_stages(f, parents)
    f = pd.concat([f, staged.add_prefix('db_')], axis=1)
    group = f.groupby('bin_id', sort=True)
    bins = group.agg(longitude=('longitude','mean'), latitude=('latitude','mean'),
        n_photons=('classification','size'), n_signal=('signal','sum'),
        id_matched_photons=('db_id_match','sum'), strict_matched_photons=('db_strict_match','sum'))
    representatives = group.distance_to_center.idxmin().to_numpy()
    rep = f.loc[representatives].set_index('bin_id')
    bins['reason'] = rep.db_reason
    bins['matched'] = rep.db_strict_match
    bins['id_match'] = rep.db_id_match
    bins['representative_class'] = rep.classification
    bins['representative_segment_id'] = rep.segment_id
    bins['representative_delta_time'] = rep.delta_time
    bins['time_outside_s'] = rep.db_time_outside_s
    bins['nearest_segment_gap'] = rep.db_nearest_segment_gap
    bins['parent_time_start'], bins['parent_time_end'] = rep.db_parent_time_start, rep.db_parent_time_end
    bins['edge_bin'] = bins.index.isin([f.bin_id.min(), f.bin_id.max()])
    bins['length_m'], bins['x_mid_m'] = length, (bins.index+.5)*length
    bins = bins[_inside(bins, bounds)].copy()  # Exactly production's mean-location crop.
    in_bins = f.bin_id.isin(bins.index)
    subset = f[in_bins]
    signal_bins = bins.n_signal.gt(0)
    reasons = bins.groupby('reason', sort=False).agg(bins=('matched','size'),
        signal_bins=('n_signal', lambda x: int(x.gt(0).sum()))).reset_index()
    reasons['percent_all_bins'] = 100*reasons.bins/len(bins) if len(bins) else np.nan
    reasons['percent_signal_bins'] = 100*reasons.signal_bins/signal_bins.sum() if signal_bins.any() else np.nan
    by_class = subset.groupby('classification', sort=True).agg(photons=('ph_index','size'),
        id_matched=('db_id_match','sum'), strict_matched=('db_strict_match','sum')).reset_index()
    by_class['class_name'] = by_class.classification.map({-1:'unclassified',0:'noise',1:'ground',2:'canopy',3:'top_of_canopy',4:'unclassified'})
    by_class['strict_match_pct'] = 100*by_class.strict_matched/by_class.photons
    summary = dict(status='ok', photons_returned=len(f), photons_in_analysis_bins=len(subset),
        database_parents=len(parents), bins=len(bins), matched_bins=int(bins.matched.sum()),
        matched_pct=_percent(bins.matched.sum(),len(bins)), signal_bins=int(signal_bins.sum()),
        signal_matched_bins=int((signal_bins & bins.matched).sum()),
        signal_matched_pct=_percent((signal_bins & bins.matched).sum(),signal_bins.sum()),
        no_signal_bins=int((~signal_bins).sum()),
        representative_time_rejected_bins=int((bins.id_match & ~bins.matched).sum()),
        id_only_matched_bins=int(bins.id_match.sum()),
        representative_miss_with_other_photon_match=int((~bins.matched & bins.strict_matched_photons.gt(0)).sum()),
        nonedge_bins=int((~bins.edge_bin).sum()),
        nonedge_matched_pct=_percent((~bins.edge_bin & bins.matched).sum(),(~bins.edge_bin).sum()),
        max_representative_time_outside_s=bins.time_outside_s.max())
    p_roi = parents[_inside(parents,bounds)] if len(parents) else parents
    for field, threshold in [('msw_flag','msw_flag_threshold'), ('layer_flag','layer_flag_threshold')]:
        summary[field+'_mean'] = p_roi[field].mean() if len(p_roi) else np.nan
        if cfg is not None:
            summary[threshold] = cfg['parallel_blocks'][threshold]
    parent_report, fields_report = pd.DataFrame(), pd.DataFrame()
    if source is not None:
        src_stage = match_stages(f, source)
        src_rep = src_stage.loc[representatives].copy()
        src_rep.index = group.size().index
        bins['source_id_match'] = src_rep.id_match
        bins['source_strict_match'] = src_rep.strict_match
        bins['source_parent_start'] = src_rep.parent_start
        db_starts = parents.segment_id_beg.to_numpy() if len(parents) else []
        bins['source_parent_missing_from_db'] = bins.source_id_match & ~bins.source_parent_start.isin(db_starts)
        parent_report, fields_report = compare_parent_fields(parents,source,
            int(f.segment_id.min()),int(f.segment_id.max()),query_bounds)
        summary.update(source_id_matched_bins=int(bins.source_id_match.sum()),
            source_strict_matched_bins=int(bins.source_strict_match.sum()),
            source_parent_missing_from_db_bins=int(bins.source_parent_missing_from_db.sum()),
            source_parents_in_photon_span=int(parent_report.membership.isin(['source_only','both']).sum()),
            source_parents_missing_from_db=int(parent_report.membership.eq('source_only').sum()),
            source_parents_in_query_missing_from_db=int((parent_report.membership.eq('source_only') &
                parent_report.source_center_in_query.eq(True)).sum()))
        missing_in_query = parent_report.membership.eq('source_only') & parent_report.source_center_in_query.eq(True)
        for flag in ['source_invalid_terrain_height','source_invalid_canopy_height']:
            if flag in parent_report:
                summary['missing_db_parents_with_'+flag] = int((missing_in_query & parent_report[flag]).sum())
        by_class['source_id_matched'] = src_stage.loc[in_bins, 'id_match'].groupby(subset.classification).sum().reindex(by_class.classification).to_numpy()
    for name in ['msw_flag','layer_flag']:
        mean = summary[name+'_mean']
        if cfg is not None:
            summary[name+'_passes_beam_check'] = bool(np.isfinite(mean) and mean <= summary[name+'_threshold'])
    examples = bins[~bins.matched].groupby('reason',sort=False,group_keys=False).head(max_examples).reset_index()
    return MatchingReport(
        _tag(pd.DataFrame([summary]),site,resource,beam),
        _tag(reasons,site,resource,beam), _tag(by_class,site,resource,beam),
        _tag(bins.reset_index(),site,resource,beam),
        _tag(parent_report,site,resource,beam), _tag(fields_report,site,resource,beam),
        pd.DataFrame(), _tag(examples,site,resource,beam))


def diagnose_sites(sites, start_time, end_time, *, length=30, resources=None,
                   max_resources=3, beams=pipeline.BEAMS, buffer_m=300,
                   compare_source=True, cfg=None, provider=None,
                   continue_on_error=True):
    """Yield one compact report per site/overpass; discard photons after each beam.

    Unlike production, this queries photons even when the database has no parents.
    Failed retrievals stay visible as error rows with no invented zero counts.
    Explicit resources override discovery and its date filter.
    """
    from sliderule import sliderule, earthdata
    sliderule.init(verbose=False,rethrow=True)
    sites = pipeline.load_sites(sites)
    provider = pipeline.public_provider() if provider is None else provider
    for site in sites.itertuples(index=False):
        polygon, geometry = pipeline.query_polygon(site,max(buffer_m,2*length))
        found = list(resources) if resources is not None else earthdata.cmr(
            short_name='ATL03',version='007',polygon=polygon,time_start=start_time,time_end=end_time)
        acquisitions = {}
        for resource in sorted(set(found)):
            pipeline.granule_info(resource)
            acquisitions[resource.rsplit('_',1)[0]] = resource
        chosen = sorted(acquisitions.values())
        if max_resources is not None:
            chosen = chosen[:max_resources]
        print(f'{site.site_id}: diagnosing {len(chosen)} overpasses',flush=True)
        for resource in chosen:
            print(f'  {resource}',flush=True)
            tables = {name:[] for name in MatchingReport.__dataclass_fields__}
            try:
                parents, query = fetch_database_with_audit(provider,geometry,resource)
                tables['query_audit'].append(_tag(query,site,resource))
            except Exception as exc:
                if not continue_on_error:
                    raise
                tables['summary'].append(_tag(pd.DataFrame([dict(status='database_query_error',error=str(exc))]),site,resource))
                yield MatchingReport(**{k:_concat(v) for k,v in tables.items()})
                continue
            source_resource, source_candidates, source_error = None,0,None
            if compare_source:
                try:
                    source_resource,source_candidates = resolve_source_resource(resource)
                    print(f'    Source check: {source_resource}',flush=True)
                except Exception as exc:
                    source_error = str(exc)
            for beam in beams:
                print(f'    {beam}',flush=True)
                try:
                    photons = pipeline.fetch_photons(resource,beam,polygon)
                    if photons.empty:
                        tables['summary'].append(_tag(pd.DataFrame([dict(status='no_photons_returned')]),site,resource,beam))
                        continue
                    source, beam_source_error = None,source_error
                    if source_resource is not None:
                        try:
                            source = fetch_source_parents(source_resource,beam)
                        except Exception as exc:
                            beam_source_error = str(exc)
                    p = parents[parents.beam.eq(beam)] if len(parents) else parents
                    report = audit_beam(photons,p,site,length,source=source,cfg=cfg,
                                        query_bounds=tuple(geometry.total_bounds))
                    report.summary['source_check_status'] = ('disabled' if not compare_source else
                        'failed' if beam_source_error is not None else 'ok')
                    report.summary['source_resource'] = source_resource
                    report.summary['source_candidates'] = source_candidates
                    report.summary['source_check_error'] = beam_source_error
                    for name in tables:
                        table = getattr(report,name)
                        if not table.empty:
                            tables[name].append(table)
                    del photons,source,report
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    tables['summary'].append(_tag(pd.DataFrame([dict(status='beam_diagnostic_error',error=str(exc))]),site,resource,beam))
                    print(f'      Diagnostic failed: {exc}',flush=True)
            yield MatchingReport(**{k:_concat(v) for k,v in tables.items()})


def plot_matching(report):
    """One row per beam: coverage and classed-signal support along the track."""
    if report.bins.empty:
        print('No bin-level diagnostic data. Inspect report.summary for errors.')
        return
    names = list(report.bins.beam.unique())
    fig,axes = plt.subplots(len(names),1,figsize=(12,2.5*len(names)),squeeze=False,layout='constrained')
    for ax,beam in zip(axes[:,0],names):
        t = report.bins[report.bins.beam.eq(beam)].sort_values('x_mid_m')
        x = (t.x_mid_m-t.x_mid_m.min())/1000
        for mask,level,label,color in [
            (t.matched,2,'Matched','#277a49'),
            (~t.matched & t.id_match,1,'Parent ID found; time rejected','#d98618'),
            (~t.id_match,0,'No database parent ID range','#b74343')]:
            ax.scatter(x[mask],np.full(mask.sum(),level),s=7,c=color,label=label)
        ax.scatter(x[t.n_signal.gt(0)],np.full(t.n_signal.gt(0).sum(),-1),s=7,c='#315a99',label='Contains class 1–3 photons')
        ax.set(title=beam,xlabel='Along track from first displayed bin (km)',
               yticks=[-1,0,1,2],yticklabels=['Signal','No parent','Time rejected','Matched'],ylim=(-1.5,2.5))
    axes[0,0].legend(loc='upper center',bbox_to_anchor=(.5,1.65),ncol=2,fontsize=9)
    first = report.bins.iloc[0]
    fig.suptitle(f'{first.site_id}: {first.resource}',fontweight='bold')
    plt.show()
