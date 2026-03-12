import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
import pandas as pd
import contextily as ctx
import math
import cartopy 
from shapely.geometry import shape, Polygon, mapping, MultiPolygon, GeometryCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
import numpy as np
from shapely import contains_xy
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from typing import Union, Literal
from .utils import Utils

from .analogues import Analogues

# Plot
#===============================================================================================================================================================

# add standard deviation
def deprecated_n_day_accumulations_gdf(data, value_col, parameter, event_date, labelticks, labels, centering=False, datetime_col="valid_time", days=None, ylimit=None):

    fig, axs = plt.subplots(ncols=4, figsize=(20, 3), dpi=100, sharey=True)

    # Ensure datetime and sorted
    data = data.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values(datetime_col)

    for i in range(4):
        ax = axs[i]

        # Determine n-day window
        if days is not None:
            ndays = days[i]
        elif value_col == 't2m':
            ndays = [1, 3, 7, 14][i]
        elif value_col == 'tp':
            ndays = [1, 3, 5, 10][i]
        else:
            ndays = [1, 3, 5, 11][i]

        if value_col == "tp":
            data_nday = (
                data.set_index(datetime_col)
                    [value_col]
                    .rolling(ndays, min_periods=1, center=centering)
                    .sum()
                    .reset_index()
            )
        else:
            data_nday = (
                data.set_index(datetime_col)
                    [value_col]
                    .rolling(ndays, min_periods=1, center=centering)
                    .mean()
                    .reset_index()
            )


        # Plot each year in blue
        for y in data_nday[datetime_col].dt.year.unique():
            data_y = data_nday[data_nday[datetime_col].dt.year == y]
            ax.plot(
                data_y[datetime_col].dt.dayofyear,
                data_y[value_col],
                color="tab:blue",
                alpha=0.3
            )

        # Style the plot
        ax.set_xticks(labelticks)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", color="k", alpha=0.2)
        ax.set_title(f"{ndays}-day accumulated {parameter}")

        # Highlight date window
        ylim = ax.get_ylim()
        Utils.print(ylim)

        dayofyear = pd.to_datetime(event_date).dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000,
                               color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        # Highlight selected year
        year2 = pd.to_datetime(event_date)
        data_y = data_nday[data_nday[datetime_col] <= year2]

        if ylimit is not None:
            ax.set_ylim(0, ylimit)
            
        ax.plot(data_y[datetime_col].dt.dayofyear, data_y[value_col], color="k")
            

    return fig, axs


def deprecated_plot_n_day_accumulations(rolled_data_list, value_col, parameter, event_date, labelticks, labels, days, ylimit=None, datetime_col="valid_time"):
    """
    Plot n-day rolling accumulations for different windows.
    
    Parameters
    ----------
    rolled_data_list : list of DataFrames
        List of results from n_day_accumulations_gdf(), one per window.
    value_col : str
        Column that was rolled.
    parameter : str
        Parameter name for titles.
    event_date : str or datetime
        Highlight date.
    labelticks : list
        X-axis tick positions.
    labels : list
        X-axis tick labels.
    days : list
        List of window sizes.
    ylimit : int or None
        Upper limit for y-axis.
    datetime_col : str
        Column with datetimes.
    """
    fig, axs = plt.subplots(ncols=len(rolled_data_list), figsize=(5 * len(rolled_data_list), 3), dpi=100, sharey=True)

    if len(rolled_data_list) == 1:
        axs = [axs]  # make iterable if only one axis

    for ax, data_nday, ndays in zip(axs, rolled_data_list, days):
        # Plot each year in blue
        for y in data_nday[datetime_col].dt.year.unique():
            data_y = data_nday[data_nday[datetime_col].dt.year == y]
            ax.plot(
                data_y[datetime_col].dt.dayofyear,
                data_y[value_col],
                color="tab:blue",
                alpha=0.3
            )

        # Style the plot
        ax.set_xticks(labelticks)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", color="k", alpha=0.2)
        ax.set_title(f"{ndays}-day accumulated {parameter}")

        # Highlight date window
        ylim = ax.get_ylim()
        dayofyear = pd.to_datetime(event_date).dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000, color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        # Highlight selected year (all up to event_date)
        year2 = pd.to_datetime(event_date)
        data_y = data_nday[data_nday[datetime_col] <= year2]
        ax.plot(data_y[datetime_col].dt.dayofyear, data_y[value_col], color="k")

        if ylimit is not None:
            ax.set_ylim(0, ylimit)

    return fig, axs



def deprecated_subplot_gdf(gdfs:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time',
                polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
                cmap:str='coolwarm', legend_title:str='Temperature (°C)', borders:bool=True,
                coastlines:bool=True, gridlines:bool=True, subtitle:str=None,
                projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None,
                dpi:int=100, flatten_empty_plots:bool=True
                ):
    
    # Ensure datetime column is datetime type
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])

    # Unique days sorted
    unique_days = sorted(gdfs[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with Cartopy projection
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi,
        subplot_kw={'projection': projection}
    )
    axes = axes.flatten()

    # Normalize color scale across all data
    vmin = math.floor(gdfs[value_col].min())
    vmax = math.ceil(gdfs[value_col].max())

    for i, day in enumerate(unique_days):
        ax = axes[i]

        # Filter GeoDataFrame for this day
        day_gdf = gdfs[gdfs[datetime_col].dt.date == day]

        # Plot data on this subplot
        day_gdf.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            legend=False,  # legend handled once globally
            vmin=vmin,
            vmax=vmax,
            marker='s'
        )

        if gridlines:
            ax.gridlines(
                crs=projection,
                linewidth=0.5,
                color='black',
                draw_labels=["bottom", "left"],
                alpha=0.2
            )

        if coastlines:
            ax.coastlines()

        if borders:
            ax.add_feature(cartopy.feature.BORDERS, lw=1, alpha=0.7, ls="--")

        # Draw polygons if provided
        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='red', linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Add shared colorbar to the top
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top", fraction=0.01, pad=.07, aspect=40)
    cbar.set_label(legend_title, labelpad=10, fontsize=12)

    if subtitle:
        fig.suptitle(subtitle, fontsize=16)
    
    # Set extent if provided
    if extends is not None:
      ax.set_extent(extends, crs=projection)

    #plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave room for suptitle and colorbar
    return fig, axes


# plots multiple subplots of a GeoDataFrame in a single figure
def deprecated_subplot_gdf2(gdfs:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time',
                polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
                cmap:str='coolwarm', legend_title:str='Temperature (°C)', borders:bool=True,
                coastlines:bool=True, gridlines:bool=True, subtitle:str=None,
                projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None,
                dpi:int=100, flatten_empty_plots:bool=True, marker:str='o'
                ):

    # Ensure datetime column is datetime type
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])

    # Unique days sorted
    unique_days = sorted(gdfs[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with Cartopy projection
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi,
        subplot_kw={'projection': projection},
    )
    axes = axes.flatten()

    # Normalize color scale across all data
    vmin = math.floor(gdfs[value_col].min())
    vmax = math.ceil(gdfs[value_col].max())

    for i, day in enumerate(unique_days):
        ax = axes[i]

        # Filter GeoDataFrame for this day
        day_gdf = gdfs[gdfs[datetime_col].dt.date == day]

        # Plot data on this subplot
        day_gdf.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            legend=False,  # legend handled once globally
            vmin=vmin,
            vmax=vmax,
            marker=marker
        )

        if gridlines:
            ax.gridlines(
                crs=projection,
                linewidth=0.5,
                color='black',
                draw_labels=["bottom", "left"],
                alpha=0.2
            )

        if coastlines:
            ax.coastlines()

        if borders:
            ax.add_feature(cartopy.feature.BORDERS, lw=1, alpha=0.7, ls="--")

        # Draw polygons if provided
        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='red', linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=18, color='darkblue', weight='medium')

            # Set extent if provided
        if extends is not None:
            ax.set_extent(extends, crs=projection)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Add shared colorbar to the top
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top", fraction=0.01, pad=.07, aspect=40)
    cbar.set_label(legend_title, labelpad=10, fontsize=27, weight='bold', color='darkblue')
    # set colorbar ticklabels
    cbar.ax.xaxis.set_tick_params(color='darkgrey') # dont work
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='darkgrey') # dont work

    if subtitle:
        fig.suptitle(subtitle)

    #plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave room for suptitle and colorbar
    return fig, axes




# Process
#===============================================================================================================================================================

def deprecated_calculate_mean_gdf(
    gdf:gpd.GeoDataFrame, 
    date_range: pd.core.arrays.datetimes.DatetimeArray,  # pyright: ignore[reportAttributeAccessIssue]
    value_col: str, 
    datetime_col: str="valid_time", 
    padding: int=15,
    year_range: None|tuple[int, int]=None, 
    group_by: list[str]=["longitude", "latitude", "geometry"]
):
    '''
    Calculate mean climatology values around each date in date_range across years,
    returning a GeoDataFrame with the same structure as the input.

    Parameters:
        gdf (GeoDataFrame, required):
            Historical data with datetime and geometry.
        date_range (pd.DatetimeIndex, required):
            Dates for which to calculate climatology means.
        value_col (str, required):
            Column with numeric values.
        datetime_col (str, optional):
            Column with datetimes.
        padding (int, optional):
            +/- days for the averaging window.
        year_range (tuple[int, int], optional):
            Year range to consider (start_year, end_year).
        group_by (list[str], optional):
            Columns to group by when averaging. If None, averages globally.
            Default groups by: longitude, latitude, geometry.


    Returns:
        data (GeoDataFrame):
            Same structure as input: longitude, latitude, valid_time, value_col, geometry
    '''
    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

    # Filter by year range if specified
    if year_range is not None:
        start_year, end_year = year_range
        gdf = gdf[(gdf[datetime_col].dt.year >= start_year) & (gdf[datetime_col].dt.year <= end_year)]

    climatology = []

    date_min = gdf[datetime_col].min()
    date_max = gdf[datetime_col].max()

    for target_date in date_range:
        results = []
        for year in gdf[datetime_col].dt.year.unique():
            try:
                center_date = datetime(year, target_date.month, target_date.day)
            except ValueError:
                # should we skip feb 29? use 28 instead? or use next day?
                continue

            # skip if center_date not in gdf
            if pd.Timestamp(center_date) not in gdf[datetime_col].values:
                Utils.print(f"Skipping {center_date} as it is not in the data date range {date_min} to {date_max}")
                continue

            # range should never exceed data range
            start = max(center_date - timedelta(days=padding), date_min)
            end = min(center_date + timedelta(days=padding), date_max)

            subset = gdf[(gdf[datetime_col] >= start) & (gdf[datetime_col] <= end)]
            results.append(subset)

        combined = pd.concat(results, ignore_index=True)

        # Average per geometry
        if group_by:
            mean_gdf = combined.groupby(group_by)[value_col].mean().reset_index()
        else:
            mean_val = combined[value_col].mean()
            mean_gdf = pd.DataFrame({
                value_col: [mean_val],
                datetime_col: [target_date]
            })

        # Assign the target date as valid_time
        mean_gdf[datetime_col] = target_date

        climatology.append(mean_gdf)

    climatology_gdf = gpd.GeoDataFrame(
        pd.concat(climatology, ignore_index=True),
        geometry="geometry",
        crs=gdf.crs
    )
    return climatology_gdf

def deprecated_calculate_anomaly(final_gdf:gpd.GeoDataFrame, event_gdf:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time'):
    # Ensure datetime columns
    final_gdf = final_gdf.copy()
    event_gdf = event_gdf.copy()
    final_gdf[datetime_col] = pd.to_datetime(final_gdf[datetime_col])
    event_gdf[datetime_col] = pd.to_datetime(event_gdf[datetime_col])
    
    # Prepare list for adjusted results
    adjusted_list = []
    
    for target_date in event_gdf[datetime_col].unique():
        # --- Step 1: get mean around date for all years ---
        results = []
        for year in final_gdf[datetime_col].dt.year.unique():
            center_date = datetime(year, target_date.month, target_date.day)
            start_window = center_date - timedelta(days=15)
            end_window = center_date + timedelta(days=15)
            
            subset = final_gdf[
                (final_gdf[datetime_col] >= start_window) &
                (final_gdf[datetime_col] <= end_window)
            ]
            results.append(subset)
        
        combined = pd.concat(results, ignore_index=True)
        mean_gdf = combined.groupby("geometry")[value_col].mean().reset_index()
        
        # --- Step 2: subtract from gdfs for this target date ---
        current_day_gdf = event_gdf[event_gdf[datetime_col] == target_date]
        
        # Merge on geometry so subtraction matches locations
        merged = current_day_gdf.merge(mean_gdf, on="geometry", suffixes=("", "_mean"))

        if value_col == "t2m":
            merged[value_col] = merged[value_col] - merged[f"{value_col}_mean"]
        elif value_col == "tp":
            merged[value_col] = merged[value_col] / merged[f"{value_col}_mean"]
        else:
            raise ValueError("value_col must be 't2m' or 'tp'")
    
        merged.drop(columns=[f"{value_col}_mean"], inplace=True)
        
        adjusted_list.append(merged)
    
    # Combine adjusted daily frames back together
    final_adjusted_gdf = gpd.GeoDataFrame(
        pd.concat(adjusted_list, ignore_index=True),
        crs=event_gdf.crs
    )
    
    return final_adjusted_gdf



def deprecated_n_day_accumulations_gdf1(gdf, value_col:str, padding:int, centering:bool=False, datetime_col:str="valid_time"):
    """
    Compute rolling n-day accumulation (sum for 'tp', mean otherwise).
    
    Parameters
    ----------
    data : GeoDataFrame or DataFrame
        Input data.
    value_col : str
        Column to roll (e.g. 't2m', 'tp').
    days : int
        Window size in days.
    centering : bool
        Center the window.
    datetime_col : str
        Column with datetimes.

    Returns
    -------
    DataFrame
        Rolled values with same columns.
    """
    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    #data = data.sort_values(datetime_col)

    if value_col == "tp":
        data_nday = (
            gdf.set_index(datetime_col)[value_col]
            .rolling(padding, min_periods=1, center=centering)
            .sum()
            .reset_index()
        )
    else:
        data_nday = (
            gdf.set_index(datetime_col)[value_col]
            .rolling(padding, min_periods=1, center=centering)
            .mean()
            .reset_index()
        )

    return data_nday

def deprecated_n_day_accumulations_gdf2(
    gdf:Union[pd.DataFrame, gpd.GeoDataFrame], 
    value_col:str, 
    padding:int,
    centering:bool=False, 
    datetime_col:str="valid_time",
    method:Literal["sum", "mean", "std", "quantile"]="mean", 
    quantile:float=0.9
):
    """
    Compute rolling n-day statistics (sum, mean, std, quantile).
    
    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Input data.
    value_col : str
        Column to roll (e.g. 't2m', 'tp').
    padding : int
        Window size in days.
    centering : bool
        Center the window.
    datetime_col : str
        Column with datetimes.
    method : {"sum", "mean", "std", "quantile"}
        Rolling aggregation method.
    q : float, optional
        Quantile to compute if method="quantile". Default is 0.5 (median).

    Returns
    -------
    DataFrame
        Rolled values with same columns.
    """

    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    
    roller = gdf.set_index(datetime_col)[value_col].rolling(
        padding, min_periods=1, center=centering
    )

    if method == "sum":
        result = roller.sum()
    elif method == "mean":
        result = roller.mean()
    elif method == "std":
        result = roller.std()
    elif method == "quantile":
        result = roller.quantile(quantile)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return result.reset_index()

# Util
#===============================================================================================================================================================

def deprecated_get_save_directory(dir: str="data", relative: bool=True, makedir: bool=True) -> str :
        '''
        Get (and) create a directory path for saving files.

        Parameters:
            dir (str):
                directory name or path ('data' by default relative to the current working directory)
            relative (bool):
                whether the directory is relative to the current working directory (True by default)
            makedir (bool):
                whether to create the directory if it does not exist (True by default)

        Returns:
            str: The absolute path to the save directory.
        '''

        CURRENT_DIRECTORY = os.getcwd()
        your_save_directory = os.path.abspath(os.path.join(CURRENT_DIRECTORY, dir)) if relative else dir

        if makedir:
            os.makedirs(your_save_directory, exist_ok=True)

        return your_save_directory

def deprecated_split_time_range_by_year_and_months(
        start: datetime,
        end: datetime,
        months: list[str]|list[int]
    ) -> list[tuple[datetime, datetime]]:
        '''
        Split a time range into sub-ranges filtered by specific months.

        This helper method iterates through the time period between start and end,
        extracting intervals that fall within the requested months. Each resulting
        tuple represents a continuous range within a single calendar month.

        Parameters:
            start (datetime):
                The beginning of the overall time range.
            end (datetime):
                The end of the overall time range.
            months (list[str] | list[int]):
                A list of months to include, provided as integers (1-12) or 
                strings.
        
        Returns:
            list[tuple[datetime, datetime]]: A list of time ranges as tuples of 
            (start_date, end_date) defining the periods within the specified months.
        '''
        result = []

        def last_day_of_month(dt: datetime) -> datetime:
            next_month = dt.replace(day=28) + timedelta(days=4)  # always moves to the next month
            return next_month.replace(day=1) - timedelta(days=1)    # always returns back to the current month
        
        current = datetime(start.year, start.month, start.day)

        while current <= end:
            if current.month in months:
                month_start = current
                month_end = last_day_of_month(current)

                actual_start = max(month_start, start)
                actual_end = min(month_end, end)

                if actual_start <= actual_end:
                    result.append((actual_start, actual_end))
                
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return result

def deprecated_shift_datetime_by_months(gdf:gpd.GeoDataFrame, shift_by:int, datetime_col:str='valid_time', direction:str='forward') -> gpd.GeoDataFrame:
        '''
        Shifts the datetime values in a specified column forward or backward by a given number of months.

        Parameters:
            gdf (gpd.GeoDataFrame, required):
                The input GeoDataFrame.
            shift_by (int, required):
                The number of months by which to shift the dates.
            datetime_col (str, optional):
                The column name containing datetime objects to be shifted.
            direction (str, optional):
                The direction of the shift. Must be 'forward' (increase date) or 'backward' (decrease date). Defaults to 'forward'.

        Returns:
            gpd.GeoDataFrame: A copy of the input GeoDataFrame with the datetime column shifted.
        '''
        n_direction = 1 if direction == 'forward' else -1 if direction == 'backward' else 0

        gdf = gdf.copy()

        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        gdf[datetime_col] = gdf[datetime_col] + pd.DateOffset(months=shift_by) * n_direction

        return gdf

# Data
#===============================================================================================================================================================





# Analogues

@staticmethod
def ED_similarity(event: iris.cube.Cube, p_cube: iris.cube.Cube, region: list[float], method: str) -> list:
    '''
    .. Deprecated:: 0.1.0
    Use ed_similarity instead. 
    '''
    return Analogues.ed_similarity(event, p_cube, region, method)

@staticmethod
def ed_similarity(event: iris.cube.Cube, p_cube: iris.cube.Cube, region: list[float], method: str) -> list:
    
    '''
    Returns similarity values based on euclidean distance

    Parameters:
        event (iris.cube.Cube):
            Cube containing the event field to be compared.
        p_cube (iris.cube.Cube):
            Cube containing candidate fields to compare against the event.
        region (list[float]):
            Region for data selecion
        method (str):
            Method chosen, either 'ED' or 'CC'

    Returns:
        list:
            List of similarity values for each spatial slice in P_cube, normalised to the range [0, 1].
    '''
    
    if method not in ['ED', 'CC']:
        raise ValueError("Invalid method. Choose 'ED' for Euclidean Distance or 'CC' for Correlation Coefficient.")

    var_e = Analogues.extract_region(event, region)
    var_p = Analogues.extract_region(p_cube, region)
    var_d = []
    
    for yx_slice in var_p.slices(['grid_latitude', 'grid_longitude']):
        
        if method == 'ED':
            var_d.append(Analogues.euclidean_distance(yx_slice, var_e))
        
        elif method == 'CC':
            var_d.append(Analogues.correlation_coeffs(yx_slice, var_e))
    
    ED_max = np.max(np.max(var_d))
    
    return [(1-x / ED_max) for x in var_d]