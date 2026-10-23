"""Remote replacement for parallel_blocks_v2's data-loading and rebinning path.

Use run_sites(selected, cfg, start_time, end_time) in a notebook. Each yielded
RunResult contains regression rows, joined bins and a small status table.
No HDF5 downloads or output writes. CLI writes CSV only if --output is supplied.
See the README for resolution differences and an example using your existing cfg.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import copy
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box as shapely_box
from shapely.geometry.polygon import orient
from pyproj import CRS, Transformer
from scipy.optimize import least_squares
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

BEAMS = ('gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r')
# Preserve Thomas's original beam numbering, rather than equating it to spot.
SPOT_TO_BEAM = {5: 1, 6: 2, 3: 3, 4: 4, 1: 5, 2: 6}
ANCILLARY = ['msw_flag', 'night_flag', 'asr', 'canopy_openness', 'snr',
             'segment_cover', 'segment_landcover', 'h_te_best_fit',
             'terrain_slope', 'cloud_flag_atm', 'layer_flag', 'sat_flag']
DB_FIELDS = ANCILLARY + ['segment_id', 'segment_id_beg', 'segment_id_end',
                        'delta_time_beg', 'delta_time_end', 'rgt']
QUALITY_VALUES = [0, 1, 2, 3, 4, 5, 10, 11, 12, 14, 15, 20, 21, 22, 24, 25]
cmap = plt.get_cmap('viridis')
cmap3 = plt.get_cmap('tab20')


@dataclass
class RunResult:
    site_id: str
    resource: str
    results: pd.DataFrame
    segments: pd.DataFrame
    diagnostics: pd.DataFrame


def granule_info(resource):
    m = re.fullmatch(r'ATL03_(\d{14})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})\.h5', resource)
    if m is None:
        raise ValueError(f'Expected a canonical ATL03 resource name: {resource}')
    time, rgt, cycle, region, release, revision = m.groups()
    if release != '007':
        raise ValueError('The public database used here is ATL08 release 007.')
    return dict(time=pd.Timestamp(datetime.strptime(time, '%Y%m%d%H%M%S')),
                rgt=int(rgt), cycle=int(cycle), region=int(region), release=release)


def load_sites(sites):
    """Accept selected_sites.csv, selected_sites.geojson, or selected DataFrame."""
    if isinstance(sites, (str, Path)):
        sites = (pd.read_csv(sites) if Path(sites).suffix.lower() == '.csv'
                 else gpd.read_file(sites))
    sites = sites.copy()
    if isinstance(sites, gpd.GeoDataFrame):
        if sites.crs is None:
            raise ValueError('Site geometry needs a CRS.')
        sites = sites.to_crs(4326)
        for name in ['west', 'south', 'east', 'north']:
            if name not in sites:
                # Per-site bounds, not the bounding box of the whole selection.
                sites[name] = sites.geometry.bounds[['minx', 'miny', 'maxx', 'maxy'][
                    ['west', 'south', 'east', 'north'].index(name)]]
    required = {'site_id', 'west', 'south', 'east', 'north'}
    if not required.issubset(sites):
        raise ValueError(f'Site table needs {sorted(required)}.')
    if sites.site_id.duplicated().any():
        raise ValueError('site_id must be unique.')
    for row in sites.itertuples():
        if not (-180 <= row.west < row.east <= 180 and -90 < row.south < row.north < 90):
            raise ValueError('This draft requires non-dateline-crossing site rectangles.')
    return sites


def query_polygon(site, buffer_m=300):
    """Conservative buffered bounds; final crop uses the exact selected bounds."""
    latitude = max(abs(site.south), abs(site.north))
    dlat = buffer_m / 110000
    dlon = buffer_m / (110000 * np.cos(np.radians(latitude + dlat)))
    bounds = (site.west-dlon, site.south-dlat, site.east+dlon, site.north+dlat)
    if bounds[0] <= -180 or bounds[2] >= 180 or bounds[1] <= -90 or bounds[3] >= 90:
        raise ValueError('Buffered site crosses the dateline or pole.')
    shape = orient(shapely_box(*bounds), sign=1)
    # Densify edges to limit the difference between geodesic and lon/lat boxes.
    coords = list(shape.exterior.coords)
    points = []
    for a, b in zip(coords[:-1], coords[1:]):
        for fraction in np.linspace(0, 1, 20, endpoint=False):
            points.append({'lon': a[0]+fraction*(b[0]-a[0]),
                           'lat': a[1]+fraction*(b[1]-a[1])})
    return points + [points[0]], gpd.GeoDataFrame(geometry=[shape], crs=4326)


def public_provider():
    import icesat2db as isdb
    return isdb.IceSat2Provider(storage_type='s3',
        s3_bucket='dog-ext.icesat2db.icesat2-atl08-v007', url='https://s3.gfz.de')


def fetch_atl08(provider, geometry, resource):
    """Decode the database's packed identity; never join across tracks/beams."""
    info = granule_info(resource)
    available = set(provider.get_available_variables().index)
    missing = set(DB_FIELDS) - available
    if missing:
        raise ValueError(f'ATL08 database is missing required fields: {sorted(missing)}')
    fields = DB_FIELDS + (['source_granule'] if 'source_granule' in available else [])
    day = info['time'].normalize()
    table = provider.get_data(variables=fields, geometry=geometry,
        start_time=str(day), end_time=str(day + pd.Timedelta(days=2)),
        query_type='bounding_box', return_type='dataframe')
    if table is None or len(table) == 0:
        return pd.DataFrame()
    table = table.copy()
    # The public database uses pack_segment_id: rgt<<43 | cycle<<35 |
    # beam<<32 | segment_id_beg. Keep int64 precision (never convert via float).
    packed = np.array([int(v) for v in table.segment_id], dtype=np.int64)
    rgt, cycle, beam = packed >> 43, (packed >> 35) & 255, (packed >> 32) & 7
    beginning = packed & ((1 << 32) - 1)
    if ((beam > 5).any() or not np.array_equal(beginning, table.segment_id_beg.to_numpy())
            or not np.array_equal(rgt, table.rgt.to_numpy())):
        raise ValueError('Unexpected icesat2db ID format; refusing an uncertain join.')
    table['beam'] = np.array(BEAMS)[beam]
    table['db_segment_id'] = packed
    table = table[(rgt == info['rgt']) & (cycle == info['cycle'])].copy()
    if 'source_granule' in table:
        prefix = resource.replace('ATL03_', 'ATL08_').rsplit('_', 1)[0] + '_'
        table = table[table.source_granule.astype(str).str.startswith(prefix)].copy()
    # Duplicate parent records (e.g. revisions) must not be resolved arbitrarily.
    if table.duplicated(['beam', 'segment_id_beg']).any():
        raise ValueError('Multiple ATL08 records for the same parent segment.')
    for field in ANCILLARY:
        table[field] = pd.to_numeric(table[field], errors='raise')
        table.loc[table[field].abs() >= 1e30, field] = np.nan
    return table


def fetch_photons(resource, beam, polygon, quality_values=QUALITY_VALUES):
    from sliderule import sliderule
    parms = dict(poly=polygon, beams=[beam], srt=0, cnf=list(range(-2, 5)),
        quality_ph=list(quality_values), podppd=255,
        atl08_class=['atl08_noise', 'atl08_ground', 'atl08_canopy',
                     'atl08_top_of_canopy', 'atl08_unclassified'],
        len=100., res=100., ats=0., cnt=1, pass_invalid=True,
        atl03_ph_fields=['delta_time', 'h_ph', 'pce_mframe_cnt', 'ph_id_pulse'])
    raw = sliderule.run('atl03x', parms, resources=[resource])
    if raw is None:
        raise RuntimeError(f'SlideRule returned None for {resource}/{beam}.')
    if len(raw) == 0:
        return pd.DataFrame()
    f = pd.DataFrame(raw.drop(columns='geometry', errors='ignore')).copy()
    if 'geometry' in raw:
        f['longitude'], f['latitude'] = raw.geometry.x.to_numpy(), raw.geometry.y.to_numpy()
    f = f.rename(columns={'atl08_class': 'classification'})
    needed = ['x_atc', 'classification', 'h_ph', 'delta_time', 'segment_id',
              'ph_index', 'spot', 'longitude', 'latitude']
    if not set(needed).issubset(f):
        raise ValueError(f'Missing SlideRule fields: {sorted(set(needed)-set(f))}')
    if not np.isfinite(f[['x_atc', 'delta_time', 'segment_id']]).all().all():
        raise ValueError('Invalid photon coordinate, time or segment ID.')
    f['resource'], f['beam'] = resource, beam
    return f.drop_duplicates('ph_index').reset_index(drop=True)


def rebin_and_attach(photons, parents, length, shot_columns=('delta_time',)):
    """Bin native photon distances, then attach the representative photon's parent.

    Bins span [bin_id*length, (bin_id+1)*length). The photon nearest the bin's
    geometric centre chooses the parent; the join uses ATL03 segment ID within
    ATL08 [segment_id_beg, segment_id_end], with an additional time-range check.
    Missing coverage is retained as atl08_matched=False, never nearest-filled.
    Only occupied bins are returned; empty bins cannot enter the regression.
    """
    if not np.isfinite(length) or length <= 0:
        raise ValueError('length must be positive.')
    if photons.empty:
        return pd.DataFrame()
    if photons[['resource', 'beam']].drop_duplicates().shape[0] != 1:
        raise ValueError('Rebin one resource and beam at a time.')
    if not shot_columns or photons[list(shot_columns)].isna().any().any():
        raise ValueError('Valid shot identifiers are required.')
    f = photons.copy()
    f['bin_id'] = np.floor(f.x_atc / length).astype('int64')
    p = parents.sort_values('segment_id_beg').reset_index(drop=True)
    parent_index = np.full(len(f), -1, dtype=int)
    if len(p):
        if p.beam.nunique() != 1 or p.beam.iloc[0] != f.beam.iloc[0]:
            raise ValueError('Parent beam does not match photons.')
        if (p.segment_id_beg.to_numpy()[1:] <= p.segment_id_end.to_numpy()[:-1]).any():
            raise ValueError('Overlapping ATL08 parent ranges.')
        ids = f.segment_id.to_numpy()
        loc = np.searchsorted(p.segment_id_beg, ids, side='right')-1
        safe = np.maximum(loc, 0)
        valid = ((loc >= 0) & (ids <= p.segment_id_end.to_numpy()[safe]) &
                 (f.delta_time.to_numpy() >= p.delta_time_beg.to_numpy()[safe]-1e-6) &
                 (f.delta_time.to_numpy() <= p.delta_time_end.to_numpy()[safe]+1e-6))
        parent_index[valid] = loc[valid]
    f['_parent'] = parent_index
    rows = []
    low, high = f.bin_id.min(), f.bin_id.max()
    for b, g in f.groupby('bin_id', sort=True):
        signal = g[g.classification.isin([1, 2, 3])]
        ground = g[g.classification.eq(1)]
        nshots = len(signal.drop_duplicates(list(shot_columns)))
        ng, nc, nt = [int(g.classification.eq(c).sum()) for c in (1, 2, 3)]
        representative = g.iloc[np.argmin(abs(g.x_atc.to_numpy()-(b+.5)*length))]
        parent = int(representative['_parent'])
        row = dict(resource=g.resource.iloc[0], beam=g.beam.iloc[0], spot=int(g.spot.iloc[0]),
            bin_id=int(b), length_m=length, coordinate_start=b*length,
            coordinate_end=(b+1)*length, longitude=g.longitude.mean(), latitude=g.latitude.mean(),
            n_te_photons=ng, n_ca_photons=nc, n_toc_photons=nt,
            n_classed_shots=nshots, n_seg_ph=len(g),
            photon_rate_te=ng/nshots if nshots else np.nan,
            photon_rate_can_nr=(nc+nt)/nshots if nshots else np.nan,
            h_te_std=ground.h_ph.std(ddof=0), h_te_median=ground.h_ph.median(),
            edge_bin=b in (low, high), atl08_matched=parent >= 0,
            atl08_parent_count=g.loc[g._parent.ge(0), '_parent'].nunique(),
            atl08_matched_photon_fraction=g._parent.ge(0).mean(),
            atl08_parent_fraction=g._parent.eq(parent).mean() if parent >= 0 else np.nan,
            ancillary_resolution_m=100, canopy_openness_resolution_m=100)
        for name in ANCILLARY:
            row[name] = p.iloc[parent][name] if parent >= 0 else np.nan
        row['atl08_segment_id'] = str(p.iloc[parent].db_segment_id) if parent >= 0 else None
        row['atl08_segment_id_beg'] = int(p.iloc[parent].segment_id_beg) if parent >= 0 else None
        rows.append(row)
    return pd.DataFrame(rows)


def process_resource(site, resource, cfg, provider, *, buffer_m=300,
                     beams=BEAMS, shot_columns=('delta_time',), quality_values=QUALITY_VALUES,
                     graph_detail=0, dw_sampler=None):
    """Process one overpass. Requires sliderule.init before calling directly."""
    cfg = copy.deepcopy(cfg)
    c = cfg['parallel_blocks']
    length = c['rebinned']
    if length <= 0:
        raise ValueError('Set cfg["parallel_blocks"]["rebinned"] to 15, 30 or another positive length.')
    if c.get('DW') and dw_sampler is None:
        raise ValueError('DW=True needs dw_sampler(site, segments); otherwise set DW=False to use ATL08 landcover.')
    polygon, geometry = query_polygon(site, max(buffer_m, 2*length))
    parents = fetch_atl08(provider, geometry, resource)
    beam_infos, joined, diagnostics = {}, [], []
    bounds = (site.west, site.south, site.east, site.north)
    for beam in beams:
        if parents.empty:
            diagnostics.append(dict(beam=beam, status='no_atl08'))
            continue
        p = parents[parents.beam == beam].copy()
        if p.empty:
            diagnostics.append(dict(beam=beam, status='no_atl08'))
            continue
        f = fetch_photons(resource, beam, polygon, quality_values)
        if f.empty:
            diagnostics.append(dict(beam=beam, status='no_photons'))
            continue
        spots = f.spot.unique()
        if len(spots) != 1 or spots[0] not in SPOT_TO_BEAM:
            raise ValueError('Invalid or changing spacecraft orientation within a beam.')
        i = SPOT_TO_BEAM[int(spots[0])]-1
        bins = rebin_and_attach(f, p, length, shot_columns)
        bins = _filter_to_bounds(bins, 'longitude', 'latitude', bounds)
        matched = bins.atl08_matched & ~bins.edge_bin
        p_roi = _filter_to_bounds(p, 'longitude', 'latitude', bounds)
        atmosphere_ok = (len(p_roi) > 0 and
            p_roi.msw_flag.mean() <= c['msw_flag_threshold'] and
            p_roi.layer_flag.mean() <= c['layer_flag_threshold'])
        keep = matched & (bins.photon_rate_te < 16) & (bins.photon_rate_can_nr < 16)
        if c.get('DW'):
            bins['DW'] = np.asarray(dw_sampler(site, bins))
        if c['landcover'] == 'forest':
            keep &= (bins.DW.eq(1) if c.get('DW') else bins.segment_landcover.isin(
                [111,112,113,114,115,116,121,122,123,124,125,126]))
        elif c['landcover'] == 'all':
            keep &= (bins.DW.notna() & bins.DW.ne(0) if c.get('DW') else
                bins.segment_landcover.notna() & ~bins.segment_landcover.isin([60,40,100,50,70,80,200,0]))
        elif c['landcover'] is not None:
            raise ValueError('landcover must be forest, all or None.')
        if c['altitude'] is not None:
            keep &= abs(bins.h_te_best_fit-c['altitude']) <= c['alt_thresh']
        if c['trim_atmospheric']:
            keep &= bins.layer_flag.eq(0) | bins.msw_flag.eq(0)  # original OR
        if c['sat_flag']:
            keep &= bins.sat_flag.eq(0)
        bins['used_for_processing'] = keep & atmosphere_ok
        bins['pipeline_beam'] = i+1
        bins['site_id'] = str(site.site_id)
        joined.append(bins)
        diagnostics.append(dict(beam=beam, status='ok' if atmosphere_ok else 'atmosphere_rejected',
            photons=len(f), bins=len(bins), matched_bins=int(bins.atl08_matched.sum()),
            accepted_bins=int(bins.used_for_processing.sum())))
        selected = bins[bins.used_for_processing].copy()
        if selected.empty:
            continue
        if i in beam_infos:
            raise ValueError('Two beams mapped to the same pipeline beam number.')
        f = _filter_to_bounds(f, 'longitude', 'latitude', bounds)
        f = f.rename(columns={'longitude': 'lon_ph', 'latitude': 'lat_ph'})
        f['classification'] = f.classification.replace(4, -1)
        beam_infos[i] = dict(gt=beam, beam_name=f'Beam {i+1}',
            atl03=SimpleNamespace(df=f), atl08=SimpleNamespace(df=selected),
            roi_bounds=bounds, center=((site.west+site.east)/2, (site.south+site.north)/2))
    result = fit_prepared(beam_infos, cfg, str(site.site_id), resource,
                          coords=((site.west+site.east)/2, (site.south+site.north)/2),
                          graph_detail=graph_detail)
    result['resource'] = resource
    result['site_id'] = str(site.site_id)
    result['length_m'] = length
    result['canopy_openness_resolution_m'] = 100
    return RunResult(str(site.site_id), resource, result,
        pd.concat(joined, ignore_index=True) if joined else pd.DataFrame(),
        pd.DataFrame(diagnostics))


def run_sites(sites, cfg, start_time, end_time, *, resources=None, max_resources=None,
              provider=None, **kwargs):
    """Yield one result at a time; dates are ISO UTC, end_time is an instant.

    Explicit resources override CMR discovery (useful for the first known pass).
    Do not accumulate yielded segments for large surveys unless you need them.
    Exceptions stop the run with site/resource context; nothing is silently skipped.
    """
    from sliderule import sliderule, earthdata
    sites = load_sites(sites)
    required = {'rebinned', 'small_box_half_side', 'altitude', 'alt_thresh',
        'insufficient_data_threshold', 'msw_flag_threshold', 'layer_flag_threshold',
        'landcover', 'trim_atmospheric', 'sat_flag', 'divide_array',
        'outlier_removal', 'slope_init', 'slope_lb', 'slope_ub', 'loss', 'f_scale', 'w'}
    missing = required - set(cfg.get('parallel_blocks', {}))
    if missing:
        raise ValueError(f'Existing parallel_blocks config is missing: {sorted(missing)}')
    if pd.Timestamp(start_time) >= pd.Timestamp(end_time):
        raise ValueError('start_time must be before end_time.')
    sliderule.init(verbose=False, rethrow=True)
    provider = public_provider() if provider is None else provider
    for site in sites.itertuples(index=False):
        polygon, _ = query_polygon(site, kwargs.get('buffer_m', 300))
        found = list(resources) if resources is not None else earthdata.cmr(
            short_name='ATL03', version='007', polygon=polygon,
            time_start=start_time, time_end=end_time)
        # One revision per acquisition; latest revision from the discovered set.
        acquisitions = {}
        for resource in sorted(set(found)):
            granule_info(resource)
            acquisitions[resource.rsplit('_', 1)[0]] = resource
        chosen = sorted(acquisitions.values())
        if max_resources is not None:
            chosen = chosen[:max_resources]
        print(f'{site.site_id}: {len(chosen)} overpasses')
        for resource in chosen:
            print(f'  {resource}', flush=True)
            try:
                yield process_resource(site, resource, cfg, provider, **kwargs)
            except Exception as exc:
                raise RuntimeError(f'{site.site_id} / {resource}: {exc}') from exc


# Regression and spatial-block functions adapted below from the supplied
# parallel_blocks_v2.py. No src.imports, PhoREAL, or local HDF5 dependency.

def parallel_odr(dataset, intercepts, maxes, cfg, model=None, res=None):
    """Original weighted parallel ODR, with explicit handling of tiny beams."""
    model = parallel_model if model is None else model
    res = parallel_residuals if res is None else res
    c = cfg['parallel_blocks']
    beam_columns = [col for col in dataset if col.startswith('Beam')]
    full = []
    removal = c['outlier_removal']
    if removal is not None and not (0 < removal <= .5 or removal >= 2):
        raise ValueError('outlier_removal must be None, (0, .5], or >=2 (LOF neighbours).')
    for beam in beam_columns:
        data = dataset[dataset[beam]].copy()
        data['Outlier'] = 1
        if removal is not None:
            xy = data[['Eg', 'Ev']]
            if removal < 1 and len(data) >= 4 and np.linalg.matrix_rank(np.cov(xy.T)) == 2:
                data['Outlier'] = EllipticEnvelope(contamination=removal,
                    random_state=42).fit_predict(xy)
            elif removal >= 2 and len(data) >= 3:
                data['Outlier'] = LocalOutlierFactor(n_neighbors=min(int(removal), len(data)-1),
                    contamination='auto').fit_predict(xy)
        full.append(data)
    full = pd.concat(full, ignore_index=True)
    kept = full[full.Outlier.eq(1)].drop(columns='Outlier').copy()
    if any(not kept[col].any() for col in beam_columns):
        raise ValueError('Outlier removal left a beam without observations.')
    X = kept.drop(columns=['Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'])
    Y = kept[['Ev']]
    lower = np.array([c['slope_lb']] + [0]*len(beam_columns), dtype=float)
    upper = np.array([c['slope_ub']] + list(maxes), dtype=float)
    initial = np.array([c['slope_init']] + list(intercepts), dtype=float)
    initial = np.clip(initial, lower+1e-8, upper-1e-8)
    fit = least_squares(res, initial, args=(X, Y, cfg, model), bounds=(lower, upper),
        loss=c['loss'], f_scale=c['f_scale'])
    if not fit.success:
        raise RuntimeError(f'Parallel regression failed: {fit.message}')
    return fit.x, kept, full


def divide_arrays(X, Y, n=2):
    '''
    This function takes arrays X and Y and splits it into partitions of equal size in increasing magnitude in x
    It returns the first and last partitions, used for estimating an initial guess for slope.
    '''
    combined = list(zip(X, Y))
    combined.sort(key=lambda tup: tup[0])
    midpoint = max(1, len(combined) // max(2, int(n)))
    
    lower_half = combined[:midpoint]
    upper_half = combined[-midpoint:]

    lower_X, lower_Y = zip(*lower_half)
    upper_X, upper_Y = zip(*upper_half)
    
    return lower_X, lower_Y, upper_X, upper_Y


def intercept_from_slope_and_point(slope, point):
    x1, y1 = point
    intercept = y1 - slope * x1
    return intercept


def find_slope_and_intercept(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def starting_intercept(X,Y,n):
    '''
    This function computes a starting guess intercept and slope given a distribution of ICESat-2 Eg Ev data.
    '''
    if len(Y) == 1:
        slope = -.3
        intercept = intercept_from_slope_and_point(slope, (list(X)[0],list(Y)[0]))
    else:
        lower_X, lower_Y, upper_X, upper_Y = divide_arrays(X, Y, n)

        y1 = np.median(lower_Y)
        y2 = np.median(upper_Y)
        x1 = np.median(lower_X)
        x2 = np.median(upper_X)

        if x1 == x2:
            slope = -.3
            intercept = intercept_from_slope_and_point(slope, (x1,y1))
        else:
            slope, intercept = find_slope_and_intercept(x1, y1, x2, y2)
            if slope > -0.01:
                slope = -0.01
            if slope < -100:
                slope = -100
            intercept = intercept_from_slope_and_point(slope, (np.mean([x1,x2]),np.mean([y1,y2])))

    return intercept, slope


def parse_filename_datetime(filename):
    '''
    Retrieve the datetime from the ATL03/08 filename.
    '''
    filename_only = filename.split('/')[-1]
    atl03_index = filename_only.find('ATL03_')
    atl08_index = filename_only.find('ATL08_')

    # Determining the split index based on which string appears first or if neither is found
    split_index = min(filter(lambda x: x >= 0, [atl03_index, atl08_index]))

    # Extracting yyyymmddhhmmss part
    date_str = filename_only[split_index + 6:split_index + 20]

    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj


def datetime_to_title(datetime_obj):
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')


def datetime_to_date(datetime_obj):
    return datetime_obj.strftime('%d/%m/%Y')


def make_box(coords, width_km=25, height_km=25):
    '''
    Function to quickly make a GeoDataFrame polygon of given half-width and half-length
    '''
    # Convert width and height from kilometers to degrees
    km_per_degree_lat = 111  # Kilometers per degree of latitude
    km_per_degree_lon = 111 * np.cos(np.radians(coords[1]))  # Kilometers per degree of longitude at given latitude

    # Convert the input width and height from kilometers to degrees
    width_deg = width_km / km_per_degree_lon
    height_deg = height_km / km_per_degree_lat

    # Create the bounding box using converted degrees
    polygon = gpd.GeoDataFrame(
        geometry=[
            shapely_box(
                coords[0] - width_deg, coords[1] - height_deg, 
                coords[0] + width_deg, coords[1] + height_deg
            )
        ], 
        crs="EPSG:4326"
    )
    
    return polygon


def model(params, x):
    return params[0]*x + params[1]


def parallel_model(params, x):
    common_slope, *parallel = params
    # Get all columns starting with 'Beam'
    beam_columns = [col for col in x.columns if col.startswith('Beam')]
    return common_slope*x['Eg'] + np.dot(x[beam_columns], parallel)


def parallel_residuals(params, x, y, cfg, model=parallel_model):
    w=cfg['parallel_blocks']['w']
    
    common_slope = params[0]
    model_output = model(params, x)
    residuals = (y.T.values[0] - model_output) / np.sqrt(1 + common_slope**2)

    beam_columns = [col for col in x.columns if col.startswith('Beam')]

    weights = []
    for i, col in enumerate(beam_columns):
        beam_number = int(col.split('Beam')[-1])  # Extract beam number
        weight = w[0] if beam_number % 2 != 0 else w[1]  # Weight: 1 if odd, 1/4 if even
        weights.append(weight)
    weighted_residuals = residuals.copy()*np.dot(x[beam_columns], weights)

    prior_penalty = 0

    residuals_and_penalty = np.append(weighted_residuals, prior_penalty)
    return residuals_and_penalty


def plot(df, ax):
    """
    Plot function for the ground tracks.
    
    df - Dataframe generated by get_atl03_struct
    ax - Axis on which to plot the relevant figure
    """
    class_dict = {-1: {'color':cmap(0.98),
                       'name':'Unclassified'},
                   0: {'color':cmap(0.2),
                       'name':'Noise'},
                   1: {'color':cmap(0.8),
                       'name':'Ground'},
                   2: {'color':cmap(0.4),
                       'name':'Canopy'},
                   3: {'color':cmap(0.6),
                       'name':'Top of canopy'}}

    ymin = None
    ymax = None
    
    if 'classification' in df.columns:
        for c in np.unique(df.classification):
            mask = df.classification==c
            ax.scatter(df[mask].lat_ph,
                       df[mask].h_ph,
                       color=class_dict[c]['color'],
                       label=class_dict[c]['name'],
                       s = 3)

            ax.legend(loc='best')
            if c != -1:
                if ymin == None:
                    ymin = min(df[mask].h_ph)
                    ymax = max(df[mask].h_ph)
                else:
                    ymin = min(ymin,min(df[mask].h_ph))
                    ymax = max(ymax,max(df[mask].h_ph))
        ax.set_ylim(ymin-0.02*(ymax-ymin),ymax+0.02*(ymax-ymin))
            
    else:
            ax.scatter(df.lat_ph,
                      df.h_ph,
                      s = 3)
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Elevation (m)')
    return


def plot_parallel(
    coefs, colors, title_date, X, Y, xx, yy,
    beam=None, file_index=None, graph_detail=1, atl03s=None,
    canopy_frac=None, terrain_frac=None, coords=None):
    """
    Combined plotting function for pvpg_parallel.

    Parameters
    ----------
    coefs : array-like
        Optimized parameters. coefs[0] is the shared slope, and coefs[1+i]
        is the intercept term for the i-th available beam.
    colors : list
        Beam indices minus one, used for consistent beam colouring.
    title_date : str
        Datetime string for the ICESat-2 overpass.
    X, Y : list of arrays
        Eg and Ev data for each successfully read beam, including outliers
    xx, yy : list-like
        Additional plotted data by beam index, not including outliers
    beam : list, optional
        List of beam numbers to show, e.g. [3, 4]. If None, all available beams are shown.
    file_index : int, optional
        File index to include in the title.
    graph_detail : int, optional
        1 = just the main scatter/regression plot
        2 = just the groundtrack subplots
        3 = groundtrack subplots + main scatter/regression plot
    atl03s : list, optional
        Required if graph_detail == 2. ATL03 objects for groundtrack plotting.
    canopy_frac : list, optional
        Canopy fraction by beam index, used in subplot titles when graph_detail == 2 or 3.
    terrain_frac : list, optional
        Terrain fraction by beam index, used in subplot titles when graph_detail == 2 or 3.
    """

    beam_names = [f"Beam {i}" for i in range(1, 7)]

    if graph_detail == 1:
        fig = plt.figure(figsize=(10, 6))
        ax_main = fig.add_subplot(111)
        axes = None
    elif graph_detail != 0:
        fig = plt.figure(figsize=(10, 12))
        if graph_detail == 2:
            ax1 = fig.add_subplot(331)
            ax2 = fig.add_subplot(332)
            ax3 = fig.add_subplot(334)
            ax4 = fig.add_subplot(335)
            ax5 = fig.add_subplot(337)
            ax6 = fig.add_subplot(338)
            ax_main = fig.add_subplot(133)
        else:
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
            ax_main = None
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    else:
        return

    # Title
    if file_index:
        fig.suptitle(
            title_date + ' - N = ' + str(file_index),
            fontsize=16 if graph_detail == 2 else 18,
        )
    else:
        fig.suptitle(
            title_date,
            fontsize=16 if graph_detail == 2 else 18,
        )

    # Groundtrack Displays
    if graph_detail in (2, 3):
        if atl03s is None:
            raise ValueError("atl03s must be provided when graph_detail is 2 or 3")
        for i, c, atl03 in zip(np.arange(len(colors)), colors, atl03s):
            if (canopy_frac is not None) and (terrain_frac is not None):
                axes[c].set_title(
                    f"{beam_names[c]} - TF = {round(terrain_frac[c], 2)}, CF = {round(canopy_frac[c], 2)}"
                )
            elif canopy_frac:
                axes[c].set_title(f"{beam_names[c]} - CF = {round(canopy_frac[c], 2)}")
            elif terrain_frac:
                axes[c].set_title(f"{beam_names[c]} - TF = {round(terrain_frac[c], 2)}")
            else:
                axes[c].set_title(f"{beam_names[c]}")

            plot(atl03, axes[c])

    # pv/pg Plot
    if graph_detail in (1, 2):
        for i, c in enumerate(colors):
            if (beam is not None) and ((c + 1) not in beam):
                continue

            ax_main.scatter(X[i], Y[i], s=7 if graph_detail == 1 else 5,
                           color=cmap3(2 * c + 1), marker='o')
            ax_main.scatter(xx[c], yy[c], s=7 if graph_detail == 1 else 5,
                           color=cmap3(2 * c), marker='o')

            ax_main.plot(
                np.array([0, 12]),
                model([coefs[0], coefs[1 + i]], np.array([0, 12])),
                label=f"Beam {int(c + 1)}",
                color=cmap3(2 * c),
                linestyle='--',
                zorder=3,
                linewidth=2 if graph_detail == 1 else 1.5
            )

            ax_main.annotate(
            r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
            xy=(.14, .967) if graph_detail == 1 else (.35, .98),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=14 if graph_detail == 1 else 8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="white"
            )
        )

        if graph_detail == 1:
            ax_main.set_xlabel('Eg (returns/shot)', fontsize=14)
            ax_main.set_ylabel('Ev (returns/shot)', fontsize=14)
            ax_main.set_xlim(0, 9)
            ax_main.set_ylim(0, 9)
            ax_main.tick_params(axis='both', labelsize=16)
            ax_main.legend(loc='best', fontsize=16)
        else:
            ax_main.set_title("Ev/Eg Rates", fontsize=8)
            ax_main.set_xlabel('Eg (returns/shot)')
            ax_main.set_ylabel('Ev (returns/shot)')
            ax_main.set_xlim(0, 8)
            ax_main.set_ylim(0, 40)
            ax_main.legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return


def _new_box(variable_names):
    return {
        'Eg': [],
        'Ev': [],
        'beam_str': [],
        'beam': [],
        'data_quantity': [],
        'dataset': [],
        'plotX': [],
        'plotY': [],
        'atl03s': [],
        'colors': [],
        'slope_init': [],
        'slope_weight': [],
        'intercepts': [],
        'maxes': [],
        'var_dict': {var: [] for var in variable_names},
    }


def _midpoint_from_frames(atl08_df, atl03_df):
    if atl08_df is not None and len(atl08_df) > 0:
        lon_min, lon_max = atl08_df['longitude'].min(), atl08_df['longitude'].max()
        lat_min, lat_max = atl08_df['latitude'].min(), atl08_df['latitude'].max()
    elif atl03_df is not None and len(atl03_df) > 0:
        lon_min, lon_max = atl03_df['lon_ph'].min(), atl03_df['lon_ph'].max()
        lat_min, lat_max = atl03_df['lat_ph'].min(), atl03_df['lat_ph'].max()
    else:
        return None
    return ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)


def _filter_to_bounds(df, lon_col, lat_col, bounds):
    if bounds is None:
        return df
    min_lon, min_lat, max_lon, max_lat = bounds
    return df[
        (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon) &
        (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat)
    ].copy()


def _compute_small_box_centers(atl08_df, atl03_df, small_box, lat_bounds=None):
    if atl08_df is not None and len(atl08_df) > 0:
        lat_series = atl08_df['latitude']
        lon_series = atl08_df['longitude']
        lon_col = 'longitude'
        lat_col = 'latitude'
        source_df = atl08_df
    elif atl03_df is not None and len(atl03_df) > 0:
        lat_series = atl03_df['lat_ph']
        lon_series = atl03_df['lon_ph']
        lon_col = 'lon_ph'
        lat_col = 'lat_ph'
        source_df = atl03_df
    else:
        return []

    if lat_bounds is None:
        lat_min, lat_max = lat_series.min(), lat_series.max()
    else:
        lat_min, lat_max = lat_bounds

    if pd.isna(lat_min) or pd.isna(lat_max):
        return []

    small_box_lat = small_box / 111.0
    lats = np.arange(lat_min + small_box_lat, lat_max + small_box_lat, small_box_lat * 2)
    if len(lats) <= 1:
        lats = np.array([(lat_min + lat_max) / 2])

    centers = []
    for lat in lats:
        lat_mask = (source_df[lat_col] >= lat - small_box_lat) & (source_df[lat_col] <= lat + small_box_lat)
        temp = source_df.loc[lat_mask]
        if len(temp) == 0:
            continue
        lon = temp[lon_col].mean()
        centers.append((lat, lon))
    return centers


def _reference_beam_index(available_indices):
    # Prefer Beam 3, then Beam 1, then Beam 5, then anything available.
    for idx in [2, 0, 4]:
        if idx in available_indices:
            return idx
    return available_indices[0] if available_indices else None


def fit_prepared(beam_infos, cfg, foldername, resource, coords=None,
                 file_index=None, graph_detail=0, beam_focus=None,
                 model=parallel_model, res=parallel_residuals, odr=parallel_odr):
    """Apply the supplied script's block grouping and output layout to remote frames."""
    small_box = cfg['parallel_blocks']['small_box_half_side']
    if small_box is not None and (small_box <= 0 or 2 < small_box < 4):
        raise ValueError('small_box_half_side must be None, (0,2], or >=4 km.')
    has_small_boxes = small_box is not None
    shared_all_beams = small_box is None or small_box >= 4
    has_roi = True
    altitude = cfg['parallel_blocks']['altitude']
    threshold = cfg['parallel_blocks']['insufficient_data_threshold']
    beam_names = [f"Beam {i}" for i in range(1,7)]
    mid_date = parse_filename_datetime(resource)
    title_date, table_date = datetime_to_title(mid_date), datetime_to_date(mid_date)
    variable_names = [
        'msw_flag', 'night_flag', 'asr', 'canopy_openness', 'snr',
        'segment_cover', 'segment_landcover', 'h_te_best_fit', 'h_te_std',
        'terrain_slope', 'longitude', 'latitude', 'cloud_flag_atm', 'layer_flag',
        'bin_id', 'atl08_segment_id', 'atl08_parent_count', 'atl08_parent_fraction',
        'atl08_matched_photon_fraction'
    ]
    if cfg['parallel_blocks'].get('DW'):
        variable_names.append('DW')
    available_beams = [i for i in range(6) if i in beam_infos and beam_infos[i]['atl08'].df.shape[0] > 0]

    columns_list = ['folder', 'date', 'lon', 'lat', 'pvpg', 'pv', 'pg', 'Eg', 'Ev',
                    'data_quantity', 'altitude', 'pv_ratio_mean', 'pv_ratio_max', 'beam', 'beam_str',
                    'outlier'] + variable_names

    if len(available_beams) == 0:
        return pd.DataFrame(columns=columns_list)

    # ---------------------------
    # Build box centers and beam->box mapping
    # ---------------------------
    box_centers = []
    beam_to_boxes = {}

    # Case 1:
    # small_box is None -> one big box, no later spatial filtering
    if not has_small_boxes:
        ref_idx = _reference_beam_index(available_beams)

        if coords is not None and has_roi:
            lon0, lat0 = coords
        else:
            ref_center = beam_infos[ref_idx]['center']
            if ref_center:
                ref_center = _midpoint_from_frames(
                    beam_infos[ref_idx]['atl08'].df,
                    beam_infos[ref_idx]['atl03'].df
                )
            lon0, lat0 = ref_center

        box_centers = [(lat0, lon0)]
        for i in available_beams:
            beam_to_boxes[i] = [0]

    # Case 2:
    # small_box >= 4 -> shared boxes for all beams based on Beam 3, else Beam 1, else Beam 5
    elif shared_all_beams:
        ref_idx = _reference_beam_index(available_beams)
        ref_info = beam_infos[ref_idx]

        box_centers = _compute_small_box_centers(
            ref_info['atl08'].df,
            ref_info['atl03'].df,
            small_box,
            lat_bounds=None
        )

        for i in available_beams:
            beam_to_boxes[i] = list(range(len(box_centers)))

    # Case 3:
    # small_box is a small number (< 2, or exactly 2 if you want to allow that)
    # keep pairwise strong/weak grouping as before
    else:
        for pair in [(0, 1), (2, 3), (4, 5)]:
            pair_available = [idx for idx in pair if idx in available_beams]
            if len(pair_available) == 0:
                continue

            ref_idx = pair[0] if pair[0] in pair_available else pair_available[0]
            ref_info = beam_infos[ref_idx]

            # If an ROI was used, preserve the old behavior of stepping through that lat range.
            lat_bounds = None
            if ref_info['roi_bounds']:
                lat_bounds = (ref_info['roi_bounds'][1], ref_info['roi_bounds'][3])

            pair_centers = _compute_small_box_centers(
                ref_info['atl08'].df,
                ref_info['atl03'].df,
                small_box,
                lat_bounds=lat_bounds
            )

            start = len(box_centers)
            box_centers.extend(pair_centers)
            pair_box_indices = list(range(start, start + len(pair_centers)))

            for idx in pair_available:
                beam_to_boxes[idx] = pair_box_indices

    if len(box_centers) == 0:
        return pd.DataFrame(columns=columns_list)

    boxes = [_new_box(variable_names) for _ in box_centers]

    # ---------------------------
    # Fill each box with beam data
    # ---------------------------
    for i in available_beams:
        info = beam_infos[i]

        for n, box_index in enumerate(beam_to_boxes.get(i, [])):
            # Cases with no later filtering: whole dataset is one box
            if not has_small_boxes:
                atl03_temp = info['atl03'].df.copy()
                atl08_temp = info['atl08'].df.copy()

            # Cases with later filtering: small boxes
            else:
                lat, lon = box_centers[box_index]
                sub_bounds = make_box((lon, lat), small_box, small_box).total_bounds

                atl03_temp = _filter_to_bounds(info['atl03'].df, 'lon_ph', 'lat_ph', sub_bounds)
                atl08_temp = _filter_to_bounds(info['atl08'].df, 'longitude', 'latitude', sub_bounds)

            if atl08_temp.shape[0] < threshold:
                print(f'Beam {i + 1}, box {n} in {foldername} file {file_index} has insufficient data.')
                continue

            X = atl08_temp.photon_rate_te.copy()
            Y = atl08_temp.photon_rate_can_nr.copy()
            if i + 1 == 3:
                X = X / 0.85
                Y = Y / 0.85

            layer_flag = atl08_temp.layer_flag
            msw_flag = atl08_temp.msw_flag
            cloud_flag_atm = atl08_temp.cloud_flag_atm

            box = boxes[box_index]

            box['plotX'].append(X)
            box['plotY'].append(Y)
            box['atl03s'].append(atl03_temp)
            box['colors'].append(i)

            box['Eg'].append(X)
            box['Ev'].append(Y)
            box['data_quantity'].append([len(X) for _ in range(len(X))])

            for var in variable_names:
                box['var_dict'][var].append(atl08_temp[var])

            if i % 2 == 0:
                box['beam_str'].append(['strong' for _ in range(len(atl08_temp['n_ca_photons']))])
            else:
                box['beam_str'].append(['weak' for _ in range(len(atl08_temp['n_ca_photons']))])

            box['beam'].append([i + 1 for _ in range(len(atl08_temp['n_ca_photons']))])

            for x, y, lf, mf, cfa in zip(X, Y, layer_flag, msw_flag, cloud_flag_atm):
                box['dataset'].append([x, y, beam_names[i], lf, mf, cfa])

            intercept, slope = starting_intercept(X, Y, cfg['parallel_blocks']['divide_array'])
            box['slope_init'].append(min(max(slope, -100 + 1e-3), -1/100 - 1e-3))
            box['slope_weight'].append(len(Y))
            box['intercepts'].append(min(intercept, 16))
            box['maxes'].append(16)

    # ---------------------------
    # Fit each box
    # ---------------------------
    rows = []

    for box_index, (lat, lon) in enumerate(box_centers):
        box = boxes[box_index]
        if len(box['dataset']) == 0:
            continue

        slope_weights = np.asarray(box['slope_weight'], dtype=float)
        slope_weights = slope_weights / slope_weights.sum()

        df = pd.DataFrame(
            box['dataset'],
            columns=['Eg', 'Ev', 'gt', 'layer_flag', 'msw_flag', 'cloud_flag_atm']
        )
        df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')

        coefs, xy, full_xy = odr(
            df_encoded,
            intercepts=box['intercepts'],
            maxes=box['maxes'],
            cfg=cfg,
            model=model,
            res=res
        )

        xx = [[] for _ in range(6)]
        yy = [[] for _ in range(6)]
        beams_in_play = []

        for beam_num in range(1, 7):
            col = f'Beam {beam_num}'
            if col in xy.columns:
                xx[beam_num - 1] = xy[xy[col] == True]['Eg']
                yy[beam_num - 1] = xy[xy[col] == True]['Ev']
                beams_in_play.append(beam_num)

        box_graph_detail = graph_detail if len(box['colors']) > 0 else 0

        plot_parallel(
            coefs=coefs,
            colors=box['colors'],
            title_date=title_date,
            X=box['plotX'],
            Y=box['plotY'],
            xx=xx,
            yy=yy,
            beam=beam_focus,
            file_index=file_index,
            graph_detail=box_graph_detail,
            atl03s=box['atl03s'],
            canopy_frac=None,
            terrain_frac=None,
            coords=(lat, lon)
        )

        # Use np.nan, not None, so np.isnan works later.
        for missing_beam in [i for i in range(1, 7) if i not in beams_in_play]:
            coefs = np.insert(coefs, missing_beam, np.nan)

        if np.all(np.isnan([coefs[1], coefs[3], coefs[5]])):
            y_strong = np.nan
            y_strong_max = np.nan
        else:
            y_strong = np.nanmean([coefs[1], coefs[3], coefs[5]])
            y_strong_max = np.nanmax([coefs[1], coefs[3], coefs[5]])

        if np.all(np.isnan([coefs[2], coefs[4], coefs[6]])):
            y_weak = np.nan
            y_weak_max = np.nan
        else:
            y_weak = np.nanmean([coefs[2], coefs[4], coefs[6]])
            y_weak_max = np.nanmax([coefs[2], coefs[4], coefs[6]])

        if np.any(np.isnan([y_strong, y_weak])):
            pv_ratio_mean = np.nan
            pv_ratio_max = np.nan
        else:
            pv_ratio_mean = y_strong / y_weak
            pv_ratio_max = y_strong_max / y_weak_max

        y_intercept_dict = {k: coefs[k] for k in range(1, 7)}
        x_intercept_dict = {
            k: (-coefs[k] / coefs[0]) if not np.isnan(coefs[k]) else np.nan
            for k in range(1, 7)
        }

        for j in range(len(box['Eg'])):
            beam_num = box['beam'][j][0] if len(box['beam'][j]) > 0 else None

            outlier_list = []
            if beam_num is not None and f'Beam {beam_num}' in full_xy.columns and 'Outlier' in full_xy.columns:
                outlier_list = full_xy.loc[
                    full_xy[f'Beam {beam_num}'] == True,
                    'Outlier'
                ].tolist()
            if len(outlier_list) == 0:
                outlier_list = [1 for _ in range(len(box['Eg'][j]))]

            row_data = [
                foldername, table_date, lon, lat, -coefs[0],
                [y_intercept_dict[x] for x in box['beam'][j]],
                [x_intercept_dict[x] for x in box['beam'][j]],
                list(box['Eg'][j]), list(box['Ev'][j]),
                box['data_quantity'][j], altitude, pv_ratio_mean, pv_ratio_max,
                box['beam'][j], box['beam_str'][j], outlier_list
            ]

            for var in variable_names:
                row_data.append(list(box['var_dict'][var][j]))

            rows.append(row_data)

    BIG_DF = pd.DataFrame(rows, columns=columns_list)
    if BIG_DF.empty:
        return BIG_DF

    explode_cols = [c for c in BIG_DF.columns if BIG_DF[c].apply(lambda x: isinstance(x, list)).any()]
    return BIG_DF.explode(explode_cols, ignore_index=True)


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', required=True, help='selected_sites.csv or selected_sites.geojson')
    parser.add_argument('--config', required=True, help='Your existing processing YAML (parallel_blocks section)')
    parser.add_argument('--start', required=True, help='ISO UTC date/time')
    parser.add_argument('--end', required=True, help='ISO UTC end instant; use next day to include a whole day')
    parser.add_argument('--site', help='Optional site_id to run first')
    parser.add_argument('--length', type=float, help='Override rebinned length in metres')
    parser.add_argument('--max-resources', type=int, help='Limit overpasses per site for a first test')
    parser.add_argument('--resource', action='append', help='Explicit ATL03 filename; bypass discovery')
    parser.add_argument('--plot', action='store_true', help='Show Eg/Ev regression plots')
    parser.add_argument('--output', type=Path, help='Optional regression CSV; existing files are not replaced')
    args = parser.parse_args()
    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)
    if args.length is not None:
        cfg['parallel_blocks']['rebinned'] = args.length
    sites = load_sites(args.sites)
    if args.site is not None:
        sites = sites[sites.site_id.astype(str).eq(args.site)]
    if sites.empty:
        parser.error('No selected sites.')
    if args.output is not None and args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    wrote_header = False
    for run in run_sites(sites, cfg, args.start, args.end, resources=args.resource,
                         max_resources=args.max_resources, graph_detail=1 if args.plot else 0):
        print(run.diagnostics.to_string(index=False))
        print(f'{len(run.results)} regression rows')
        if args.output is not None and not run.results.empty:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            run.results.to_csv(args.output, mode='a' if wrote_header else 'x',
                               header=not wrote_header, index=False)
            wrote_header = True
    if args.output is not None:
        print(f'Saved {args.output}' if wrote_header else 'No regression rows; no output file created.')


if __name__ == '__main__':
    main()
