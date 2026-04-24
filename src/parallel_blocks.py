from src.imports import *
from src.track_pairs import *
from src.DW import *
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

sys.path.insert(1,'/home/s1803229/src/PhoREAL')
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

        if outlier_removal == False:
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
    if outlier_removal != False:
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
    if file_index is not None:
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
            elif canopy_frac is not None:
                axes[c].set_title(f"{beam_names[c]} - CF = {round(canopy_frac[c], 2)}")
            elif terrain_frac is not None:
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
    width=cfg['parallel_blocks']['roi_half_width']
    height=cfg['parallel_blocks']['roi_half_height']
    small_box = cfg['parallel_blocks']['small_box_half_side']
    rebinned = cfg['parallel_blocks']['rebinned']
    res_field=cfg['parallel_blocks']['res_field']

    altitude = cfg['parallel_blocks']['altitude']
    alt_thresh=cfg['parallel_blocks']['alt_thresh']
    
    threshold = cfg['parallel_blocks']['insufficient_data_threshold']
    trim_atmospheric=cfg['parallel_blocks']['trim_atmospheric']
    sat_flag = cfg['parallel_blocks']['sat_flag']
    outlier_removal=cfg['parallel_blocks']['outlier_removal']
    landcover = cfg['parallel_blocks']['landcover']
    DW=cfg['parallel_blocks']['DW']
    
    f_scale = cfg['parallel_blocks']['f_scale']
    loss = cfg['parallel_blocks']['loss']
    init = cfg['parallel_blocks']['slope_init']
    lb = cfg['parallel_blocks']['slope_lb']
    ub = cfg['parallel_blocks']['slope_ub']
    w = cfg['parallel_blocks']['w']

    canopy_frac = cfg['parallel_blocks']['canopy_frac']
    terrain_frac = cfg['parallel_blocks']['terrain_frac']
    
    foldername = dirpath.split('/')[-2]
    
    mid_date = parse_filename_datetime(atl03path)
    title_date = datetime_to_title(mid_date)
    table_date = datetime_to_date(mid_date)

    polygon = make_box(coords, width, height)
    min_lon, min_lat, max_lon, max_lat = polygon.total_bounds

    # Convert small_box from kilometers to degrees
    km_per_degree_lat = 111  # Kilometers per degree of latitude
    km_per_degree_lon = 111 * np.cos(np.radians(coords[1]))  # Kilometers per degree of longitude at the given latitude    

    # Calculate the increment in degrees for the small box size
    small_box_lat = small_box / km_per_degree_lat
    small_box_lon = small_box / km_per_degree_lon

    # Generate the latitude and longitude ranges using the converted small box sizes
    lats = np.arange(min_lat + small_box_lat,
                     max_lat + small_box_lat,
                     small_box_lat*2)
    lons = np.arange(min_lon + small_box_lon,
                     max_lon + small_box_lon,
                     small_box_lon*2)

    # This will hold all of the data in one place:
    # [[Eg, Ev, Beam 1],...[Eg,Ev,Beam 1],[Eg,Ev,Beam 2],...,[Eg,Ev,Beam6],[Eg,Ev,Beam 6]]
    # This will be made into a dataframe later.
    Eg = [[] for _ in range(3*len(lats)*len(lons))]
    Ev = [[] for _ in range(3*len(lats)*len(lons))]
    trad_cc = [[] for _ in range(3*len(lats)*len(lons))]
    beam_str = [[] for _ in range(3*len(lats)*len(lons))]
    beam = [[] for _ in range(3*len(lats)*len(lons))]
    data_quantity = [[] for _ in range(3*len(lats)*len(lons))]

    # Define base variable names
    variable_names = [
        'msw_flag', 'night_flag', 'asr', 'canopy_openness', 
        'snr', 'segment_cover', 'segment_landcover', 
        'h_te_best_fit', 'h_te_std', 'terrain_slope', 'longitude', 'latitude',
        'cloud_flag_atm', 'layer_flag'
    ]
    if DW != False:
        variable_names.append('DW') # config this
    var_dict = {}
    for var in variable_names:
        var_dict[var] = [[] for _ in range(3*len(lats)*len(lons))]

    dataset = [[] for _ in range(3*len(lats)*len(lons))]

    # for plotting
    plotX = [[] for _ in range(3*len(lats)*len(lons))]
    plotY = [[] for _ in range(3*len(lats)*len(lons))]
    atl03s = [[] for _ in range(3*len(lats)*len(lons))]
    colors = [[] for _ in range(3*len(lats)*len(lons))]

    # for starting inits
    slope_init = [[] for _ in range(3*len(lats)*len(lons))]
    slope_weight = [[] for _ in range(3*len(lats)*len(lons))]
    intercepts = [[] for _ in range(3*len(lats)*len(lons))]
    maxes = [[] for _ in range(3*len(lats)*len(lons))]

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

    # The only purpose of this is to keep the data organised later.
    beam_names = [f"Beam {i}" for i in range(1,7)]

    # through the loops this will track the lat/lon pairings for each strong beam
    LATS = []
    LONS = []

    # this holds the groups of lat/lon pairs of each strong beam
    ALL_LATS = []
    ALL_LONS = []

    skip_weak = 0
    box_index_base = weak_box_index_base = 0
    for i, gt in enumerate(tracks):
        
        if i % 2 == 1:
            if len(LONS) == 0:
                continue
        
        if skip_weak == 1:
            skip_weak = 0
            continue
        try:
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError, IndexError) as e:
            print(f"Failed to open ATL03 file for {foldername} file {file_index}'s beam {i+1}.")
            if i % 2 == 1:
                weak_box_index_base = box_index_base
            continue
        try:
            atl08 = get_atl08_struct(atl08path, gt, atl03)
        except (KeyError, ValueError, OSError) as e:
            print(f"Failed to open ATL08 file for {foldername} file {file_index}'s beam {i+1}.")
            if i % 2 == 1:
                weak_box_index_base = box_index_base
            continue

        atl03.df = atl03.df[(atl03.df['lon_ph'] >= min_lon) & (atl03.df['lon_ph'] <= max_lon) &\
                                (atl03.df['lat_ph'] >= min_lat) & (atl03.df['lat_ph'] <= max_lat)]
        atl08.df = atl08.df[(atl08.df['longitude'] >= min_lon) & (atl08.df['longitude'] <= max_lon) &\
                                (atl08.df['latitude'] >= min_lat) & (atl08.df['latitude'] <= max_lat)]

        if rebinned != 0:
            if atl08.df.shape[0] == 0:
                print(f"Nothing to rebin for {foldername} file {file_index}'s beam {i+1}.")
                if i % 2 == 1:
                    weak_box_index_base = box_index_base
                continue
            atl08.df = rebin_atl08(atl03, atl08, gt, rebinned, res_field)

        atl08.df = atl08.df[(atl08.df.photon_rate_can_nr < 16) & (atl08.df.photon_rate_te < 16)]# & (atl08.df.h_canopy < 100)]

        if DW != False:
            filepath = find_dynamicworld_file(foldername)
            da = rioxarray.open_rasterio(filepath, masked=True).rio.reproject("EPSG:4326")

            if atl08.df.shape[0] == 0:
                # Ensure the DW column exists even if there are no rows,
                # and skip the interpolation that would fail on empty coords.
                atl08.df['DW'] = np.array([], dtype='float32')
            else:
                atl08.df['DW'] = da.sel(band=1).interp(
                    y=("points", atl08.df.latitude.values),
                    x=("points", atl08.df.longitude.values),
                    method="nearest"
                ).values

        if landcover == 'forest':
            if DW != False:
                # DynamicWorld: 1 = trees
                atl08.df = atl08.df[atl08.df['DW'] == 1]
            else:
                # Original Corine-based forest mask
                atl08.df = atl08.df[atl08.df['segment_landcover'].isin(
                    [111,112,113,114,115,116,121,122,123,124,125,126]
                )]
        elif landcover == 'all':
            if DW != False:
                # Keep everything except obvious non-land / no-data (here: DW == 0)
                atl08.df = atl08.df[~atl08.df['DW'].isin([0])]
            else:
                atl08.df = atl08.df[~atl08.df['segment_landcover'].isin(
                    [60,40,100,50,70,80,200,0]
                )]

        if altitude != None:
            atl08.df = atl08.df[abs(atl08.df['h_te_best_fit'] - altitude) <= alt_thresh]

        if trim_atmospheric != False:
            atl08.df = atl08.df[(atl08.df['layer_flag'] == 0)|(atl08.df['msw_flag'] == 0)]

        if sat_flag != False:
            atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

        if i % 2 == 0:
            K = box_index_base
        else:
            K = weak_box_index_base

        box_index = K
        if i % 2 == 0:
            LATS = []
            LONS = []
            lats = np.arange(min_lat + small_box_lat,
                 max_lat + small_box_lat,
                 small_box_lat*2)
            if len(lats) <= 1:
                lats = [(min_lat + max_lat)/2]
        if i % 2 == 1:
            lats, lons = LATS, LONS

        for n, lat in enumerate(lats):
            if i % 2 == 0:
                polygon = make_box((coords[1],lat), width, small_box)
                sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds
                
                atl03_temp = atl03.df[(atl03.df['lat_ph'] >= sub_min_lat) & (atl03.df['lat_ph'] <= sub_max_lat)].copy()
                atl08_temp = atl08.df[(atl08.df['latitude'] >= sub_min_lat) & (atl08.df['latitude'] <= sub_max_lat)].copy()
    
                if len(atl08_temp) != 0:
                    lon = atl08_temp.longitude.mean()
                else:
                    print(f'Beam {i + 1}, box {n} in {foldername} file {file_index} has no data.')
                    continue

            if i % 2 == 1:
                lon = lons[n]

            polygon = make_box((lon,lat), small_box,small_box)
            sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds
            atl03_temp = atl03.df[(atl03.df['lon_ph'] >= sub_min_lon) & (atl03.df['lon_ph'] <= sub_max_lon) &\
                                    (atl03.df['lat_ph'] >= sub_min_lat) & (atl03.df['lat_ph'] <= sub_max_lat)].copy()
            atl08_temp = atl08.df[(atl08.df['longitude'] >= sub_min_lon) & (atl08.df['longitude'] <= sub_max_lon) &\
                                    (atl08.df['latitude'] >= sub_min_lat) & (atl08.df['latitude'] <= sub_max_lat)].copy()
            if atl08_temp.shape[0] < threshold:
                print(f'Beam {i + 1}, box {n} in {foldername} file {file_index} has insufficient data.')
                if i % 2 == 1:
                    box_index += 1
                continue

            X = atl08_temp.photon_rate_te
            Y = atl08_temp.photon_rate_can_nr
            if i + 1 == 3:
                X /= 0.85
                Y /= 0.85
            layer_flag = atl08_temp.layer_flag
            msw_flag = atl08_temp.msw_flag
            cloud_flag_atm = atl08_temp.cloud_flag_atm
            plotX[box_index].append(X)
            plotY[box_index].append(Y)
            if i % 2 == 0:
                LATS.append(lat)
                LONS.append(lon)
            atl03s[box_index].append(atl03_temp)
            colors[box_index].append(i)
            Eg[box_index].append(X)
            Ev[box_index].append(Y)
            data_quantity[box_index].append([len(X) for x in range(len(X))])
            for var in variable_names:
                var_dict[var][box_index].append(atl08_temp[var])
            if i % 2 == 0:
                beam_str[box_index].append(['strong' for _ in range(len(atl08_temp['n_ca_photons']))])
            else:
                beam_str[box_index].append(['weak' for _ in range(len(atl08_temp['n_ca_photons']))])
            beam[box_index].append([i+1 for _ in range(len(atl08_temp['n_ca_photons']))])

            for x, y, lf, mf, cfa in zip(X,Y, layer_flag, msw_flag, cloud_flag_atm):
                dataset[box_index].append([x, y, beam_names[i], lf, mf, cfa])
            intercept, slope = starting_intercept(X,Y,cfg['parallel_blocks']['divide_array'])
            slope_init[box_index].append(min(max(slope, -100 + 1e-3), -1/100 - 1e-3)) #config
            slope_weight[box_index].append(len(Y))
            intercepts[box_index].append(min(intercept,16))
            maxes[box_index].append(16)

            box_index += 1

        if i % 2 == 0:
            ALL_LATS.extend(LATS)
            ALL_LONS.extend(LONS)
            box_index_base = box_index

        if i % 2 == 1:
            LATS = []
            LONS = []
            weak_box_index_base = box_index_base

    rows = []

    box_index = 0
    for lat, lon in zip(ALL_LATS, ALL_LONS):
        if len(dataset[box_index]) == 0:
            box_index += 1
            continue

        slope_weight[box_index] /= np.sum([slope_weight[box_index]])
        slope_init[box_index] = np.dot(slope_init[box_index], slope_weight[box_index])

        df = pd.DataFrame(dataset[box_index], columns=['Eg', 'Ev', 'gt', 'layer_flag', 'msw_flag', 'cloud_flag_atm'])
        df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')
        coefs, xy, full_xy = odr(df_encoded, intercepts = intercepts[box_index], maxes = maxes[box_index], cfg=cfg, model = model, res = res)

        # Create the array of empty lists
        xx = [[] for _ in range(6)]
        yy = [[] for _ in range(6)]

        beams_in_play = []
        # Iterate over each beam column and append the Eg values belonging to that beam
        for i in range(1, 7):  # Beam 1 to Beam 6
            if f'Beam {i}' in xy.columns:
                xx[i-1] = xy[xy[f'Beam {i}'] == True]['Eg']
                yy[i-1] = xy[xy[f'Beam {i}'] == True]['Ev']
                beams_in_play.append(i)

        if len(colors) == 0:
            graph_detail = 0

        plot_parallel(coefs = coefs,
                      colors = colors[box_index],
                      title_date = title_date,
                      X = plotX[box_index],
                      Y = plotY[box_index],
                      xx = xx,
                      yy = yy,
                      beam = beam_focus,
                      file_index = file_index,
                      graph_detail = graph_detail,
                      atl03s = atl03s[box_index],
                      canopy_frac = None,
                      terrain_frac = None,
                      coords = (lat,lon))

        indices_to_insert = [i for i in range(1,7) if i not in beams_in_play]
        for index in indices_to_insert:
            coefs = np.insert(coefs, index, None)
        
        if np.all(np.isnan([coefs[1],coefs[3],coefs[5]])):
            y_strong = np.nan
        else:
            y_strong = np.nanmean([coefs[1],coefs[3],coefs[5]])
            y_strong_max = np.nanmax([coefs[1],coefs[3],coefs[5]])
            
        if np.all(np.isnan([coefs[2],coefs[4],coefs[6]])):
            y_weak = np.nan
        else:
            y_weak = np.nanmean([coefs[2],coefs[4],coefs[6]])
            y_weak_max = np.nanmax([coefs[2],coefs[4],coefs[6]])
            
        if np.any(np.isnan([y_strong, y_weak])):
            pv_ratio_mean = np.nan
            pv_ratio_max = np.nan
        else:
            pv_ratio_mean = y_strong/y_weak
            pv_ratio_max = y_strong_max/y_weak_max
        
        y_intercept_dict = {1: coefs[1], 2: coefs[2], 3: coefs[3], 4: coefs[4], 5: coefs[5], 6: coefs[6]}
        x_intercept_dict = {1: -coefs[1]/coefs[0], 2: -coefs[2]/coefs[0], 3: -coefs[3]/coefs[0], 4: -coefs[4]/coefs[0],
                           5: -coefs[5]/coefs[0], 6: -coefs[6]/coefs[0]}

        for j in range(len(Eg[box_index])):
            row_data = [foldername, table_date, lon, lat, -coefs[0],
                        [y_intercept_dict[x] for x in beam[box_index][j]], [x_intercept_dict[x] for x in beam[box_index][j]],
                        list(Eg[box_index][j]), list(Ev[box_index][j]),
                        data_quantity[box_index][j], altitude, pv_ratio_mean, pv_ratio_max,
                        beam[box_index][j], beam_str[box_index][j]]
            row_data.append(full_xy['Outlier'].iloc[j])

            # Add the rest of the strong-weak pairs dynamically
            for var in variable_names:  # Start from msw, as meanEg and meanEv are already included
                # row_data.append(non_negative_subset(var_dict[var][box_index])[j])
                row_data.append(list(var_dict[var][box_index][j]))
            rows.append(row_data)
        box_index+=1

    columns_list = ['camera', 'date', 'lon', 'lat', 'pvpg', 'pv', 'pg', 'Eg', 'Ev',
                    'data_quantity', 'altitude', 'pv_ratio_mean', 'pv_ratio_max','beam', 'beam_str',
                    'outlier']
    for var in variable_names:  # Start from msw, as meanEg and meanEv are already included
        columns_list.append(var)
    BIG_DF = pd.DataFrame(rows,columns=[columns_list])
    BIG_DF.columns = BIG_DF.columns.get_level_values(0)
    return BIG_DF.explode([c for c in BIG_DF.columns if isinstance(BIG_DF[c].iloc[0], list)], ignore_index=True)