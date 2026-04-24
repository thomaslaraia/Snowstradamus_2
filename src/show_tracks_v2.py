from src.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
                        
from src.track_pairs import *
from src.DW import *
from shapely.geometry import Point, box as shapely_box

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import sys

PHOREAL_PATH = Path(__file__).resolve().parents[2] / "src" / "PhoREAL"
if str(PHOREAL_PATH) not in sys.path:
    sys.path.insert(1, str(PHOREAL_PATH))
# sys.path.insert(1,'/home/s1803229/src/PhoREAL')
# sys.path.insert(1,'C:/Users/s1803229/Documents/src/PhoREAL')

from phoreal.reader import get_atl03_struct, get_atl08_struct
from phoreal.binner import rebin_atl08

import folium
import matplotlib.colors as colors
import matplotlib.cm as cm

import contextily as ctx
from shapely.geometry import Point, box
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

def plot_static_map(df, c='Eg', cmap='viridis', vmin=0, vmax=6,
                             save='no',
                             pad_fraction=0.08):

    df = df.dropna(subset=['latitude', 'longitude', c]).copy()
    df = df[np.isfinite(df[c])]

    if df.empty:
        raise ValueError("No valid data points available to plot.")

    if vmin is None:
        vmin = df[c].min()
    if vmax is None:
        vmax = df[c].max()

    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    lon_pad = max(lon_span * pad_fraction, 0.01)
    lat_pad = max(lat_span * pad_fraction, 0.01)

    extent = [
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad
    ]

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs="EPSG:4326"
    )
    gdf_web = gdf.to_crs(epsg=3857)

    extent_pts = gpd.GeoSeries(
        [Point(extent[0], extent[2]), Point(extent[1], extent[3])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    extent_web = extent_pts.total_bounds

    fig, ax = plt.subplots(figsize=(10, 8))

    gdf_web.plot(
        ax=ax,
        column=c,
        cmap=cmap,
        markersize=3,
        legend=True,
        vmin=vmin,
        vmax=vmax
    )

    cbar = ax.get_figure().get_axes()[-1]
    cbar.set_ylabel(c, rotation=270, labelpad=15, fontsize=13)
    cbar.tick_params(labelsize=13)

    ax.set_xlim(extent_web[0], extent_web[2])
    ax.set_ylim(extent_web[1], extent_web[3])
    ax.set_aspect('equal', adjustable='box')

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def format_lon(x, _):
        lon, _ = transformer.transform(x, 0)
        return f"{lon:.3f}°"

    def format_lat(y, _):
        _, lat = transformer.transform(0, y)
        return f"{lat:.3f}°"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    plt.tight_layout()

    if save != 'no':
        plt.savefig(save, dpi=600)

    plt.show()

def show_tracks(dirpath, atl03paths, atl08paths, cfg, gtx=None):
    """
    Shows the groundtracks from a given overpass on a figure. Each 100m footprint is coloured by its ground photon return rate unless otherwise specified.

    atl03paths - Array of paths/to/atl03/file/
    atl08paths - Array of paths/to/atl08/file/
    ax - axis to plot figure on.
    c - value by which the tracks are coloured, either 'Eg' (default) or 'Ev'
    gtx - array of strings to indicate which specific groundtracks you want to see
    """

    res_field = cfg['parallel_blocks']['res_field']
    rebinned = cfg['parallel_blocks']['rebinned']
    sat_flag = cfg['parallel_blocks']['sat_flag']
    DW = cfg['parallel_blocks']['DW']
    landcover = cfg['parallel_blocks']['landcover']
    trim_atmospheric = cfg['parallel_blocks']['trim_atmospheric']

    big_df = pd.DataFrame(columns=['latitude', 'longitude', 'Eg', 'Ev', 'beam'])

    foldername = dirpath.split('/')[-2]

    for atl03path, atl08path in zip(atl03paths, atl08paths):
        A = h5py.File(atl03path, 'r')
        if list(A['orbit_info']['sc_orient'])[0] == 1:
            strong = ['gt1r', 'gt2r', 'gt3r']
            weak = ['gt1l', 'gt2l', 'gt3l']
        elif list(A['orbit_info']['sc_orient'])[0] == 0:
            strong = ['gt3l', 'gt2l', 'gt1l']
            weak = ['gt3r', 'gt2r', 'gt1r']
        else:
            print('Satellite in transition orientation.')
            A.close()
            return big_df
        tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
        A.close()

        if gtx is not None:
            tracks = [tracks[i - 1] for i in gtx]

        for i, gt in enumerate(tracks):
            try:
                atl03 = get_atl03_struct(atl03path, gt, atl08path)
                atl08 = get_atl08_struct(atl08path, gt, atl03)
            except (KeyError, ValueError, OSError):
                continue

            if rebinned != 0:
                if atl08.df.shape[0] == 0:
                    continue
                try:
                    atl08.df = rebin_atl08(atl03, atl08, gt, rebinned, res_field)
                except (KeyError, ValueError, OSError):
                    continue

            atl08.df = atl08.df[
                (atl08.df['photon_rate_can_nr'] < 16) &
                (atl08.df['photon_rate_te'] < 16)
            ]

            if DW:
                filepath = find_dynamicworld_file(foldername)
                da = rioxarray.open_rasterio(filepath, masked=True).rio.reproject("EPSG:4326")

                if atl08.df.shape[0] == 0:
                    atl08.df['DW'] = np.array([], dtype='float32')
                else:
                    atl08.df['DW'] = da.sel(band=1).interp(
                        y=("points", atl08.df.latitude.values),
                        x=("points", atl08.df.longitude.values),
                        method="nearest"
                    ).values

            if landcover == 'forest':
                if DW:
                    atl08.df = atl08.df[atl08.df['DW'] == 1]
                else:
                    atl08.df = atl08.df[atl08.df['segment_landcover'].isin(
                        [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126]
                    )]
            elif landcover == 'all':
                if DW:
                    atl08.df = atl08.df[~atl08.df['DW'].isin([0])]
                else:
                    atl08.df = atl08.df[~atl08.df['segment_landcover'].isin(
                        [60, 40, 100, 50, 70, 80, 200, 0]
                    )]

            if trim_atmospheric:
                atl08.df = atl08.df[
                    (atl08.df['layer_flag'] == 0) | (atl08.df['msw_flag'] == 0)
                ]

            if sat_flag:
                atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

            atl08.df = atl08.df.rename(columns={
                'photon_rate_te': 'Eg',
                'photon_rate_can_nr': 'Ev'
            })

            if i + 1 == 3:
                atl08.df['Eg'] /= 0.85
                atl08.df['Ev'] /= 0.85

            df = atl08.df.loc[:, ['latitude', 'longitude', 'Eg', 'Ev']].copy()
            df['beam'] = i

            if big_df.shape[0] == 0:
                big_df = df
            else:
                big_df = pd.concat([big_df, df], ignore_index=True)

    return big_df
    
def map_setup(map_path, extent = None):
    """
    Sets up the plot to show the tracks on. Requires a geotiff file as basemap.

    map_path - path/to/map/
    extent - controls the map extent if want to focus on specific part of the map.d
    """
    
    # Create plot with relevant projection, and set extent if given as parameter
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (10,7))
    if extent != None:
        ax.set_extent(extent)

    
    # Open image and show on the plot
    tif = rasterio.open(map_path)
    show(tif, ax=ax, transform=ccrs.PlateCarree())
    
    # Add labels, title, and legend
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Map of Tracks')
    # ax.legend()
    
    # Add latitude and longitude gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=1, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False  # Updated lines
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    return fig, ax7