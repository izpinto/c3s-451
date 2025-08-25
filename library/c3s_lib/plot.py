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
from datetime import datetime
import xarray as xr



def visualize_geo(
    df,
    value_col: str,
    lon_col: str,
    lat_col: str,
    backend: str = "plotly",
    *,
    # matplotlib kwargs
    figsize=(10, 10),
    cmap="viridis",
    edgecolor="black",
    alpha=0.7,
    # plotly kwargs
    mapbox_style="carto-positron",
    zoom=3,
    width=800,
    height=600,
    size=None,
    hover_name=None,
    **unused
):
    """
    Visualize a tabular or GeoDataFrame spatially.
    
    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
      Your table containing lon/lat or (already) a geometry column.
    value_col : str
      Column name to drive color (or size) of points/polys.
    lon_col, lat_col : str
      Column names for longitude and latitude (if df is not yet GeoDataFrame).
    backend : {'matplotlib','plotly'}
      Which renderer to use.
    **kwargs : 
      Backend‐specific styling options (see signature).
      
    Returns
    -------
    fig : matplotlib Figure or plotly.graph_objs._figure.Figure
    """
    # 1) ensure GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
    else:
        # if it's already a GeoDataFrame but has no geometry, force from lon/lat
        if df.geometry.isna().all():
            df = df.set_geometry(
                gpd.points_from_xy(df[lon_col], df[lat_col])
            )
    
    if backend.lower() == "matplotlib":
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(
            column=value_col,
            cmap=cmap,
            edgecolor=edgecolor,
            alpha=alpha,
            legend=True,
            ax=ax
        )
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    elif backend.lower() == "plotly":
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=value_col,
            size=size,
            hover_name=hover_name,
            zoom=zoom,
            width=width,
            height=height,
            mapbox_style=mapbox_style
        )
        return fig

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'matplotlib' or 'plotly'.")




# plots a single plot of a GeoDataFrame
def plot_gdf(gdf:gpd.GeoDataFrame, value_col:str, borders:bool=True, coastlines:bool=True,
             gridlines:bool=True, title:str=None, legend:bool=True, legend_title:str=None,
             cmap:str='coolwarm', fig_size:tuple[int, int]=(7,5), polygons:list[Polygon]=None,
             projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None
             ):
    
    fig, ax = plt.subplots(
        ncols = 1, nrows = 1, figsize = fig_size, dpi = 100, 
        subplot_kw = {"projection" : projection}
        )   

    # set color map   
    # vmin = -50
    # vmax = 50   # this should be variable based on temp or preticipation
    # temp_kwargs = {"cmap" : cmap, "vmin":vmin, "vmax":vmax}
    temp_kwargs = {"cmap" : cmap}

    # Set the colorbar properties
    legend_title = legend_title if legend_title else "legend"

    # Plot the GeoDataFrame
    gdf.plot(ax = ax, **temp_kwargs,
        legend=legend, legend_kwds={'label': legend_title},
        column = value_col,
        )

    # Add contextily basemap
    if gridlines:
        ax.gridlines(
            crs=projection, 
            linewidth=0.5, color='black', 
            draw_labels=["bottom", "left"], alpha=0.2
        )

    # Add coastlines
    if coastlines:
        ax.coastlines()

    # Add contextily basemap
    if borders:
        ax.add_feature(
            cartopy.feature.BORDERS, 
            lw = 1, alpha = 0.7, ls = "--", zorder = 99
        )

    # Draw polygons if provided
    if polygons is not None:
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, transform=projection)

    if title is not None:
        ax.set_title(title)

    # Set extent if provided
    if extends is not None:
      ax.set_extent(extends, crs=projection)

    return fig, ax




# plots multiple subplots of a GeoDataFrame in a single figure
def subplot_gdf(gdfs:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time',
                polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
                cmap:str='coolwarm', legend_title:str='Temperature (°C)', borders:bool=True,
                coastlines:bool=True, gridlines:bool=True, subtitle:str=None,
                projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None
                ):
    
    # Ensure datetime column is datetime type
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])

    # Unique days sorted
    unique_days = sorted(gdfs[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with Cartopy projection
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        subplot_kw={'projection': projection}
    )
    axes = axes.flatten()

    # Normalize color scale across all data
    vmin = gdfs[value_col].min()
    vmax = gdfs[value_col].max()

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
        axes[j].set_visible(False)

    # Add shared colorbar to the right
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    #cbar_ax = fig.add_axes([.2, .95, .6, .02])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top", fraction=0.01, pad=.07, aspect=40)
    cbar.set_label(legend_title, labelpad=10, fontsize=12)

    if subtitle:
        fig.suptitle(subtitle, fontsize=16)
    
    # Set extent if provided
    if extends is not None:
      ax.set_extent(extends, crs=projection)

    #plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave room for suptitle and colorbar
    return fig, axes





def plot_poly(polygons:list[Polygon], coords:list[list[float]], elevation:xr.DataArray=None, projection:cartopy.crs=ccrs.PlateCarree()):
    lons, lats = zip(*coords)
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    if elevation is not None:
        elevation_subset = elevation.sel(
            lon=slice(min_lon-3, max_lon+3),
            lat=slice(min_lat-3, max_lat+3)  
        ) 

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection}) 

    ax.set_extent([min_lon - 3, max_lon + 3, min_lat - 3, max_lat + 3], crs=projection)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(draw_labels=True)  

    # Set colorbar to 1:4 to keep water blue and land green
    cax = inset_axes(
        ax,
        width="3%", height="100%",
        loc='center left',
        bbox_to_anchor=(1.1, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )   
    if elevation is not None:
        elevation_plot = elevation_subset.plot(
            ax=ax,
            transform=projection,
            cmap="terrain",
            vmin=-250, vmax= 1000,
            cbar_ax=cax,
            cbar_kwargs={"label": "Elevation (m)"},
            add_colorbar=True,
            add_labels=False
        ) 

    ax.set_title("Selected regions") 

    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='red', linewidth=2, transform=projection)
        ax.fill(x, y, color='red', alpha=0.3, transform=projection)

    return fig, ax




def plot_geometry(geom, ax, color:str='green', alpha:float=0.3, projection:cartopy.crs=ccrs.PlateCarree()):
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=2, transform=projection)
        ax.fill(x, y, color=color, alpha=alpha, transform=projection)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            plot_geometry(poly, ax, color=color, alpha=alpha)
    elif isinstance(geom, GeometryCollection):
        for subgeom in geom.geoms:
            if isinstance(subgeom, (Polygon, MultiPolygon, GeometryCollection)):
                plot_geometry(subgeom, ax, color=color, alpha=alpha)
            else:
                # Optionally handle or ignore other geometry types
                pass





def elevation_region(data:dict, polygons:list[Polygon], elevation:xr.DataArray, threshold:int, projection:cartopy.crs=ccrs.PlateCarree()):
  
    all_coords = []
    adjusted_polygons = []

    for feature in data["features"]:

        coords = feature['geometry']['coordinates'][0]
        all_coords.extend(coords)
        poly = Polygon(coords)  
        minx, miny, maxx, maxy = poly.bounds
        elev_subset = elevation.sel(
            lon=slice(minx-0.5, maxx+0.5),
            lat=slice(miny-0.5, maxy+0.5)  
        )   
        elev_vals = elev_subset.squeeze().values

        if elev_subset.lat.values[0] < elev_subset.lat.values[-1]:
            elev_vals = elev_vals[::-1, :]  
            lat = elev_subset.lat.values[::-1]
        else:
            lat = elev_subset.lat.values

        lon = elev_subset.lon.values    
        lon2d, lat2d = np.meshgrid(lon, lat)    
        below_thresh = elev_vals <= threshold   
        inside_poly = contains_xy(poly, lon2d, lat2d)   
        final_mask = below_thresh & inside_poly

        transform = rasterio.transform.from_bounds(
            lon.min(), lat.min(),   
            lon.max(), lat.max(),   
            len(lon), len(lat)
        )   
        from shapely.geometry import shape

        for geom, val in rasterio.features.shapes(
                final_mask.astype(np.uint8),
                mask=final_mask,
                transform=transform):
            
            if val == 1:
                new_poly = shape(geom)
                clipped_poly = new_poly.intersection(poly)

                if not clipped_poly.is_empty:
                    adjusted_polygons.append(clipped_poly)

    lons, lats = zip(*all_coords)
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

    ax.set_title(f"Selected regions under {threshold} m elevation")
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.set_extent([min_lon - 3, max_lon + 3, min_lat - 3, max_lat + 3], crs=projection)
    ax.gridlines(draw_labels=True)

    cax = inset_axes(
        ax,
        width="3%", height="100%",
        loc='center left',
        bbox_to_anchor=(1.1, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    elev_plot = elevation.plot(
        ax=ax,
        transform=projection,
        cmap="terrain",
        cbar_ax=cax,
        cbar_kwargs={"label": "Elevation (m)"},
        add_colorbar=True,
        add_labels=False,
        vmin=0, 
        vmax=threshold  
    )

    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='red', linewidth=2, transform=projection)

    for geom in adjusted_polygons:
        plot_geometry(geom, ax)

    return fig, ax, adjusted_polygons




# plot a timeseries of a GeoDataFrame [date, value]
def plot_timeseries(data:gpd.GeoDataFrame, title:str, x_label:str, y_label:str, label_rotation:int=0, dateformat:str="%Y-%m-%d", x_ticks:mdates.DayLocator=mdates.DayLocator(), color:str='darkblue', linewidth:float=2.0, linestyle:str='-'):

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot directly onto the axes
    data.plot(ax=ax,
              color=color,     # line color
              linewidth=linewidth,        # line width
              linestyle=linestyle        # dashed line; use "-" for solid, ":" for dotted
              )

    

    # Set major ticks to the 1st of each month
    ax.xaxis.set_major_locator(x_ticks)

    # Format the ticks as full dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(dateformat))

    # Rotate tick labels
    for label in ax.get_xticklabels():
        label.set_rotation(label_rotation)
        label.set_horizontalalignment("right")

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add grid
    ax.grid(True)

    # Make sure everything fits
    fig.tight_layout()

    return fig, ax




def plot_n_day_accumulations(
    rolled_data_list:list[gpd.GeoDataFrame], value_col:str, parameter:str, event_date:datetime,
    labelticks:list[int], labels:list[any], days:list[int], ylimit:int=None, datetime_col:str="valid_time"
):
    """
    Plot n-day rolling accumulations for different windows.
    """
    fig, axs = plt.subplots(
        ncols=len(rolled_data_list),
        figsize=(5 * len(rolled_data_list), 3),
        dpi=100,
        sharey=True
    )

    if len(rolled_data_list) == 1:
        axs = [axs]  # make iterable if only one axis

    event_date = pd.to_datetime(event_date)
    event_year = event_date.year

    for ax, data_nday, ndays in zip(axs, rolled_data_list, days):
        # Plot all years EXCEPT event year in blue
        for y in data_nday[datetime_col].dt.year.unique():
            if y == event_year:
                continue
            data_y = data_nday[data_nday[datetime_col].dt.year == y]
            ax.plot(
                data_y[datetime_col].dt.dayofyear,
                data_y[value_col],
                color="tab:blue",
                alpha=0.3
            )

        # Plot event year only up to event_date in black
        data_event = data_nday[
            (data_nday[datetime_col].dt.year == event_year) &
            (data_nday[datetime_col] <= event_date)
        ]
        ax.plot(
            data_event[datetime_col].dt.dayofyear,
            data_event[value_col],
            color="k"
        )

        # Style
        ax.set_xticks(labelticks)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", color="k", alpha=0.2)
        ax.set_title(f"{ndays}-day accumulated {parameter}")

        # Highlight date window
        ylim = ax.get_ylim()
        dayofyear = event_date.dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000, color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        if ylimit is not None:
            ax.set_ylim(0, ylimit)

    return fig, axs
