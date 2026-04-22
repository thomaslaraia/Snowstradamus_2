from scripts.imports import *
from scripts.track_pairs import *
from scripts.DW import *
import geopandas as gpd
from shapely.geometry import Point, box as shapely_box
from scipy.optimize import least_squares, minimize
import scipy.sparse.linalg
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import zscore, norm
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import random

import sys
import gc

sys.path.insert(1,'/../../src/PhoREAL')
# sys.path.insert(1,'C:/Users/s1803229/Documents/PhoREAL')

from phoreal.reader import get_atl03_struct, get_atl08_struct
from phoreal.binner import rebin_atl08

import warnings
warnings.filterwarnings("ignore", module="sklearn.neighbors._lof")
warnings.filterwarnings("ignore", module="scipy.interpolate._interpolate")

def non_negative_subset(asr_list):
    cleaned_data = []
    
    for item in asr_list:
        # Check if the item is a pandas Series (from your dataframe)
        if isinstance(item, pd.Series):
            # Append the non-negative values from the pandas Series
            cleaned_data.extend(item.values)
        # Check if it's a list with a single value [-1]
        elif isinstance(item, list) and item == [-1]:
            continue  # Skip the [-1] list, as it represents missing data
        # If it's a regular list, append non-negative values
        elif isinstance(item, list) and ('strong' in item or 'weak' in item):
            cleaned_data.extend([x for x in item])
        else:
            cleaned_data.extend([x for x in item])
        # elif isinstance(item, list):
        #     cleaned_data.extend([x for x in item])
    
    return np.array(cleaned_data)  # Return as a numpy array

def divide_arrays(X, Y, n=2):
    '''
    This function takes arrays X and Y and splits it into partitions of equal size in increasing magnitude in x
    It returns the first and last partitions, used for estimating an initial guess for slope.
    '''
    combined = list(zip(X, Y))
    combined.sort(key=lambda tup: tup[0])
    midpoint = len(combined) // n
    
    lower_half = combined[:midpoint]
    upper_half = combined[midpoint*(n-1):]

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

# def non_negative_subset(asr_list):
#     '''
#     Function designed to remove [-1] from data, corresponding to empty beams
#     '''
#     cleaned_data = []
    
#     for item in asr_list:
#         if isinstance(item, pd.Series):
#             cleaned_data.extend(item.values)
#         elif isinstance(item, list) and item == [-1]:
#             continue
#         else:
#             cleaned_data.extend([x for x in item])
    
#     return np.array(cleaned_data)

# def safe_mean(arr):
#     if arr.size == 0:  # Check if the array is empty
#         return np.nan
#     else:
#         return np.mean(arr)

# def flatten_structure(structure):
#     flat_list = []
#     if isinstance(structure, (list, tuple, np.ndarray)):
#         for item in structure:
#             flat_list.extend(flatten_structure(item))
#     else:
#         flat_list.append(structure)
#     return flat_list

#This is a fairly standard linear model
def model(params, x):
    return params[0]*x + params[1]

# This defines the residuals orthogonal to the regression line
def residuals(params, x, y):
    return (y - model(params, x))/np.sqrt(1 + params[0]**2)

# This performs ODR regression for Y against X with initial guess init using least_squares()
def odr(X,Y, init, res = residuals):
    loss = cfg['parallel_blocks']['loss']
    bounds = ([cfg['parallel_blocks']['slope_lb'],0], [cfg['parallel_blocks']['slope_ub'],16])
    f_scale = cfg['parallel_blocks']['f_scale']
    
    result = least_squares(res, init, loss = loss, f_scale=f_scale, args=(X,Y), bounds = bounds)
    
    # a is the slope of the line, b is the intercept
    a, b = result.x
    return a, b

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

def parallel_odr(dataset, intercepts, maxes, cfg, model = parallel_model, res = parallel_residuals):
    """
    Performs the parallel orthogonal distance regression on the given dataset.
    
    dataset - Pandas Dataframe with columns Eg, Ev, and Beam _ for each beam with data.
    intercepts - Array holding the initial y_intercept guess for each beam. If e.g. only Beams 5 and 6 made it, then there are only two values in this array.
    maxes - Array that holds the maximum y-intercept allowed when fitting for each beam.
    cfg - config yaml holding parameters
    model - Model to estimate Ev and Eg.
    res - Residuals to put into least_squares function
    """
    init = cfg['parallel_blocks']['slope_init']
    lb = cfg['parallel_blocks']['slope_lb']
    ub = cfg['parallel_blocks']['slope_ub']
    loss=cfg['parallel_blocks']['loss']
    f_scale=cfg['parallel_blocks']['f_scale']
    outlier_removal = cfg['parallel_blocks']['outlier_removal']
    w=cfg['parallel_blocks']['w']
    
    cats = dataset.shape[1] - 5

    a = [lb] + [0]*cats
    b = [ub] + maxes
    bounds = (a,b)
    initial_params = [init] + intercepts

    beam_columns = [col for col in dataset.columns if 'Beam' in col]
    filtered_data = []
    full_data = []
    data_quantity = 0

    for beam in beam_columns:
        beam_data = dataset[dataset[beam] == True][['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'] + beam_columns].copy()

        if outlier_removal == None:
            beam_data['Outlier'] = 1
            full_data.append(beam_data[['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm', 'Outlier'] + beam_columns])
            continue

        elif len(beam_data) >= 2:
            if outlier_removal < 1:
                envelope = EllipticEnvelope(contamination=outlier_removal, random_state=42)
                envelope.fit(beam_data[['Eg', 'Ev']])
                beam_data['Outlier'] = envelope.predict(beam_data[['Eg', 'Ev']])
                beam_filtered = beam_data[beam_data['Outlier'] == 1]
            elif outlier_removal >= 2:
                outlier_flags = np.zeros(len(beam_data), dtype=bool)
                n = int(max(1,min(outlier_removal,len(beam_data)-3))) #config the 3
                lof = LocalOutlierFactor(n_neighbors=n, contamination='auto')
                preds = lof.fit_predict(beam_data[['Eg', 'Ev']])
                outlier_flags = (preds == -1) # this used to be |=, this fix could create bug?
                beam_data['Outlier'] = np.where(outlier_flags, -1, 1)
                beam_filtered = beam_data[beam_data['Outlier'] == 1]
            else:
                beam_filtered = beam_data
        else:
            beam_filtered = beam_data

        filtered_data.append(beam_filtered[['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'] + beam_columns])
        full_data.append(beam_data[['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm', 'Outlier'] + beam_columns])
        data_quantity = max(data_quantity, len(beam_data))

    full_dataset = pd.concat(full_data).reset_index(drop=True)
    if outlier_removal != None:
        filtered_dataset = pd.concat(filtered_data).reset_index(drop=True)
        dataset = filtered_dataset.copy()

    X = dataset.drop(columns=['Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'])
    Y = dataset[['Ev']]

    if loss == 'linear':
        params = least_squares(parallel_residuals, x0=initial_params, args=(X, Y, cfg, model), loss = loss, bounds = bounds)
    else:
        params = least_squares(parallel_residuals, x0=initial_params, args=(X, Y, cfg, model), loss = loss, bounds = bounds,
                               f_scale=f_scale, ftol=1e-15, xtol=1e-15, gtol=1e-15)

    return params.x, dataset, full_dataset

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

def pvpg_parallel(dirpath, atl03path, atl08path, cfg, coords = None, file_index = None,
                  model = parallel_model, res = parallel_residuals, odr = parallel_odr,
                  beam_focus = None, y_init = np.max, graph_detail = 0):
    """
    Parallel regression of all tracks on a given overpass.

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    cfg - config yaml holding parameters
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    model - model function to be used in least squares. Default is the parallel model function
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    odr - function that performs the orthogonal regression. Replace with great care if you do.
    beam - Default is None. Put in input in the form of an array of integers. For example, if you only want to display pv/pg on the plot for Beams 3 and 4, the input is [3,4]
    y_init - This is the function used to initialize the guess for the y intercept. Default is simply the maximum value, as this is expected to correspond with the data point closest to the y-intercept.
    graph_detail - Default is 0. If set to 1, will show a single pv/pg plot for all chosen, available beams. If set to 2, will also show each available groundtrack.
    """
    width = cfg['parallel_blocks']['roi_half_width']
    height = cfg['parallel_blocks']['roi_half_height']
    small_box = cfg['parallel_blocks']['small_box_half_side']
    rebinned = cfg['parallel_blocks']['rebinned']
    res_field = cfg['parallel_blocks']['res_field']

    altitude = cfg['parallel_blocks']['altitude']
    alt_thresh = cfg['parallel_blocks']['alt_thresh']

    threshold = cfg['parallel_blocks']['insufficient_data_threshold']
    trim_atmospheric = cfg['parallel_blocks']['trim_atmospheric']
    sat_flag = cfg['parallel_blocks']['sat_flag']
    outlier_removal = cfg['parallel_blocks']['outlier_removal']
    landcover = cfg['parallel_blocks']['landcover']
    DW = cfg['parallel_blocks']['DW']

    canopy_frac = cfg['parallel_blocks']['canopy_frac']
    terrain_frac = cfg['parallel_blocks']['terrain_frac']

    if (width is None) != (height is None):
        raise ValueError("roi_half_width and roi_half_height must either both be numbers or both be None.")
    if (coords is None) and (width is not None or height is not None):
        raise ValueError("Cannot set roi_half_width and roi_half_height as numbers while coords = None")
    if small_box is not None and 2 < small_box < 4:
        raise ValueError("small_box_half_side values between 2 and 4 are not acceptable.")

    has_roi = (coords is not None and width is not None and height is not None)
    has_small_boxes = (small_box is not None)
    shared_all_beams = (small_box is None) or (small_box >= 4 if small_box else False)

    foldername = dirpath.split('/')[-2]

    mid_date = parse_filename_datetime(atl03path)
    title_date = datetime_to_title(mid_date)
    table_date = datetime_to_date(mid_date)

    variable_names = [
        'msw_flag', 'night_flag', 'asr', 'canopy_openness',
        'snr', 'segment_cover', 'segment_landcover',
        'h_te_best_fit', 'h_te_std', 'terrain_slope', 'longitude', 'latitude',
        'cloud_flag_atm', 'layer_flag'
    ]
    if DW:
        variable_names.append('DW')

    # satellite orientation
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
        return 0
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    A.close()

    beam_names = [f"Beam {i}" for i in range(1, 7)]
    beam_infos = {}

    # ---------------------------
    # Load and optionally prefilter each beam
    # ---------------------------
    for i, gt in enumerate(tracks):
        try:
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError, IndexError):
            print(f"Failed to open ATL03 file for {foldername} file {file_index}'s beam {i+1}.")
            continue

        try:
            atl08 = get_atl08_struct(atl08path, gt, atl03)
        except (KeyError, ValueError, OSError):
            print(f"Failed to open ATL08 file for {foldername} file {file_index}'s beam {i+1}.")
            continue

        initial_center = coords if coords else _midpoint_from_frames(atl08.df, atl03.df)
        roi_bounds = None

        # Case: width/height are numbers -> initial ROI filter
        if has_roi:
            if initial_center is None:
                print(f"Beam {i + 1} in {foldername} file {file_index} has no data to define ROI center.")
                continue
            roi_bounds = make_box(initial_center, width, height).total_bounds
            atl03.df = _filter_to_bounds(atl03.df, 'lon_ph', 'lat_ph', roi_bounds)
            atl08.df = _filter_to_bounds(atl08.df, 'longitude', 'latitude', roi_bounds)

        print()
        print(f"Beam {i + 1}, file {file_index}")
        print(f"msw flag: {atl08.df.msw_flag.mean()}")
        print(f"layer flag: {atl08.df.layer_flag.mean()}")



        if atl08.df.msw_flag.mean() > cfg['parallel_blocks']['msw_flag_threshold'] or \
            atl08.df.layer_flag.mean() > cfg['parallel_blocks']['layer_flag_threshold']:
            print(f"Beam {i + 1} in {foldername} file {file_index} has significant atmospheric scattering.")
            continue

        if rebinned != 0:
            if atl08.df.shape[0] == 0:
                print(f"Nothing to rebin for {foldername} file {file_index}'s beam {i+1}.")
                continue
            atl08.df = rebin_atl08(atl03, atl08, gt, rebinned, res_field)

        atl08.df = atl08.df[
            (atl08.df.photon_rate_can_nr < 16) &
            (atl08.df.photon_rate_te < 16)
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

        if altitude:
            atl08.df = atl08.df[abs(atl08.df['h_te_best_fit'] - altitude) <= alt_thresh]

        if trim_atmospheric:
            atl08.df = atl08.df[(atl08.df['layer_flag'] == 0) | (atl08.df['msw_flag'] == 0)]

        if sat_flag:
            atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

        if atl08.df.shape[0] == 0:
            print(f"Beam {i + 1} in {foldername} file {file_index} has no data after filtering.")

        beam_infos[i] = {
            'gt': gt,
            'beam_name': beam_names[i],
            'atl03': atl03,
            'atl08': atl08,
            'roi_bounds': roi_bounds,
            'center': initial_center if initial_center else _midpoint_from_frames(atl08.df, atl03.df),
        }

    available_beams = [i for i in range(6) if i in beam_infos and beam_infos[i]['atl08'].df.shape[0] > 0]

    columns_list = ['camera', 'date', 'lon', 'lat', 'pvpg', 'pv', 'pg', 'Eg', 'Ev',
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