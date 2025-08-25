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



# Plot
#===============================================================================================================================================================

# add standard deviation
def depreciated_n_day_accumulations_gdf(data, value_col, parameter, event_date, labelticks, labels, centering=False, datetime_col="valid_time", days=None, ylimit=None):

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
        print(ylim)

        dayofyear = pd.to_datetime(event_date).dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000,
                               color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        # Highlight selected year
        year2 = pd.to_datetime(event_date)
        data_y = data_nday[data_nday[datetime_col] <= year2]
        ax.plot(data_y[datetime_col].dt.dayofyear, data_y[value_col], color="k")

    if ylimit is not None:
        ax.set_ylim(0, ylimit)

    return fig, axs


def depreciated_plot_n_day_accumulations(rolled_data_list, value_col, parameter, event_date, labelticks, labels, days, ylimit=None, datetime_col="valid_time"):
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


# Process
#===============================================================================================================================================================

def depreciated_calculate_anomaly(final_gdf:gpd.GeoDataFrame, event_gdf:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time'):
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





# Util
#===============================================================================================================================================================






# Data
#===============================================================================================================================================================





