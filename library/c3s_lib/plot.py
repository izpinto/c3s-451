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
import matplotlib.font_manager as fm
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import cmocean
import matplotlib.image as mpimg
from IPython.display import display, clear_output

# set font directory
BASE_DIR = os.path.dirname(__file__)
font_path = os.path.join(BASE_DIR, "RobotoCondensed-Regular.ttf")
roboto_condensed_regular = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

logo_horizon_path = os.path.join(BASE_DIR, "LogoLine_horizon_C3S.png")

# set font family and settings globally
plt.rcParams["font.family"] = roboto_condensed_regular.get_name()
plt.rcParams["axes.titlecolor"] = "#364563"
plt.rcParams["figure.titlesize"] = 27
plt.rcParams["axes.titlesize"] = 27
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelcolor"] = "#6a6a6b"     # 🔹 x/y axis labels
plt.rcParams["xtick.color"] = "#6a6a6b"         # 🔹 x-axis tick labels
plt.rcParams["xtick.labelcolor"] = "#6a6a6b"
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.color"] = "#6a6a6b"         # 🔹 y-axis tick labels
plt.rcParams["ytick.labelcolor"] = "#6a6a6b"
plt.rcParams["ytick.labelsize"] = 13

# set colormap color values
temperature_colors = [
    "#204182", "#24569c", "#559bd4", "#95d0f0", "#cee9f5", "#f6fcfe",
    "#fff1ba", "#ffc656", "#f6862f", "#e8432a", "#b92027"
]
precipitation_colors = ["#693f18", "#ffffff", "#204182"]
anomaly_colors = ["#af1f29", "#ffffff", "#204282"]

# create colormap from colors
temperature_cmap = ListedColormap(temperature_colors, name="temperature_cmap")
precipitation_cmap = LinearSegmentedColormap.from_list("precipitation_cmap", precipitation_colors, N=11)
anomaly_cmap = LinearSegmentedColormap.from_list("precipitation_cmap", anomaly_colors, N=11)

# get colormap normalization
def cmap_norm(vmin:int|float, vmax:int|float, steps:int):
    boundaries = np.linspace(vmin, vmax, steps + 1)
    return BoundaryNorm(boundaries, steps)

# set global style paramaters
def set_style(param:str, value:str|int|float):
    plt.rcParams[param] = value

# get colormap
def get_colormap(map:str): 

    original_map = map
    map = map.lower()

    match map:
        case 't2m':
            return temperature_cmap
        case 'tp':
            return precipitation_cmap
        case 'anomaly' | 'sst':
            return anomaly_cmap
        case _:
            return original_map

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

def add_image_below(fig, image_path,
                    min_height_frac=0.06,
                    max_height_frac=0.40,
                    pad_frac=0.02,
                    save_path="output_with_logo.png"
):
    """
    Add an image below a figure, spanning full width and preserving aspect ratio.
    The figure is updated in-place, shown in the notebook, and also saved to a PNG.
    """
    img = mpimg.imread(image_path)
    img_h, img_w = img.shape[0:2]

    fig_w, fig_h = fig.get_size_inches()
    img_aspect = img_h / img_w
    fig_aspect = fig_w / fig_h
    true_height_frac = img_aspect * fig_aspect

    # Clamp height
    height = max(min_height_frac, min(max_height_frac, true_height_frac))

    # Reposition existing axes upward
    available_space = 1.0 - height - pad_frac
    for ax in fig.get_axes():
        left, bottom, width, h = ax.get_position().bounds
        new_bottom = height + pad_frac + bottom * available_space
        new_height = h * available_space
        ax.set_position([left, new_bottom, width, new_height])

    # Add logo axes
    ax_img = fig.add_axes([0.1, 0.0, .8, height*.8])
    ax_img.imshow(img, aspect="auto")
    ax_img.axis("off")

    # Save to file
    #fig.savefig(save_path, bbox_inches="tight", dpi=fig.dpi)

    clear_output(wait=True)
    display(fig)

    return fig, ax_img




# plots a single plot of a GeoDataFrame
def plot_gdf(gdf:gpd.GeoDataFrame, value_col:str, borders:bool=True, coastlines:bool=True,
             gridlines:bool=True, title:str=None, legend:bool=True, legend_title:str=None,
             cmap:str=None, fig_size:tuple[int, int]=(7,5), polygons:list[Polygon]=None,
             projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None,
             dpi:int=100, marker:str='s', add_logos:bool=True
):
    
    # get colormap
    cmap = get_colormap(cmap if cmap else value_col)

    fig, ax = plt.subplots(
        ncols = 1, nrows = 1, figsize = fig_size, dpi = dpi, 
        subplot_kw = {"projection" : projection}
        )   

    # set color map   
    vmin = gdf[value_col].min()
    vmax = gdf[value_col].max()
    norm = cmap_norm(vmin, vmax, 11)
    colorbar_kwargs = {"cmap" : cmap, "norm": norm}

    # Set the colorbar properties
    legend_title = legend_title if legend_title else "legend"

    # Plot the GeoDataFrame
    gdf.plot(ax = ax, **colorbar_kwargs,
        legend=legend, legend_kwds={'label': legend_title, "ticks": norm.boundaries},
        column = value_col, marker=marker
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
        ax.set_title(title,
                     fontdict={
                        'fontsize': 27,
                        'fontweight': 'bold',
                        # 'color': '#364563'
                     })

    # Set extent if provided
    if extends is not None:
      ax.set_extent(extends, crs=projection)

    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=0)
        return fig, ax, img_ax
    else:
        return fig, ax



def subplot_gdf(
    gdfs:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time',
    polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
    cmap:str=None, legend_title:str='Temperature (°C)', borders:bool=True,
    coastlines:bool=True, gridlines:bool=True, subtitle:str=None,
    projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None,
    dpi:int=100, flatten_empty_plots:bool=True, marker:str='s',
    shared_colorbar:bool=True, add_logos:bool=True
):
    
    # get colormap
    cmap = get_colormap(cmap if cmap else value_col)

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

    # Shared color scale (only if shared_colorbar=True)
    if shared_colorbar:
        vmin = gdfs[value_col].min()
        vmax = gdfs[value_col].max()

    for i, day in enumerate(unique_days):
        ax = axes[i]
        day_gdf = gdfs[gdfs[datetime_col].dt.date == day]

        # Individual scale if not shared
        if not shared_colorbar:
            vmin = day_gdf[value_col].min()
            vmax = day_gdf[value_col].max()

        # Plot
        day_gdf.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            legend=False,  # individual legends if no shared colorbar
            vmin=vmin,
            vmax=vmax,
            marker=marker,
        )

        if not shared_colorbar:
            # Add colorbar per axis
            norm = cmap_norm(vmin, vmax, 11)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04, ticks=norm.boundaries)
            cbar.set_label(legend_title)

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

        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='red', linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=18, weight='medium')

        if extends is not None:
            ax.set_extent(extends, crs=projection)

    fig.subplots_adjust(wspace=0.4)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Shared colorbar if requested
    if shared_colorbar:
        norm = cmap_norm(vmin, vmax, 11)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top", fraction=0.01, pad=.07, aspect=60, ticks=norm.boundaries)
        cbar.set_label(legend_title, labelpad=10, fontsize=27, weight='bold', color='#364563')
        # cbar.set_ticks(np.round(norm.boundaries).astype(int))
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), family=roboto_condensed_regular.get_name(), fontsize=13)
        plt.show()
    
    if subtitle:
        fig.suptitle(subtitle)

    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=-0.1)
        return fig, axes, img_ax
    else:
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
                    if clipped_poly.geom_type == "Polygon":
                        adjusted_polygons.append(clipped_poly)
                    elif clipped_poly.geom_type == "MultiPolygon":
                        adjusted_polygons.extend(list(clipped_poly.geoms))

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




def plot_timeserie(data, value_col:str, title:str, x_label:str, y_label:str, datetime_col:str='valid_time', 
                   fig_size:tuple=(12,6), dpi:int=100, show_grid:bool=True, line_style:str=':', marker_style:str=None, 
                   draw_style:str='default', label_rotation:int=0, line_width:float=1.5, labelticks:list[str]=None, labels:list[str]=None,
                   add_logos:bool=True
):
    
    #set font family globally
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    ax.plot(data[datetime_col], data[value_col], 
            color='darkblue', 
            linewidth=line_width, 
            linestyle=line_style, 
            drawstyle=draw_style,
            **(marker_style if marker_style is not None else {})
            )

    ax.set_title(label=title)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)

    if labelticks is not None:
        ax.set_xticks(labelticks)
    if labels is not None:
        ax.set_xticklabels(labels)

    if show_grid:
        ax.grid(True)

    # Format x-axis with date labels
    #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=label_rotation)

    # plt.tight_layout()
    # plt.show()
    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=-.1)
        return fig, ax, img_ax
    else:
        return fig, ax





def plot_n_day_accumulations(
    rolled_data_list:list[gpd.GeoDataFrame], value_col:str, parameter:str, event_date:datetime,
    labelticks:list[int], labels:list[any], days:list[int], ylimit:int=None, datetime_col:str="valid_time",
    add_logos:bool=True, fig_height: int = 3, xtick_rotation: int = 0
):
    """
    Plot n-day rolling accumulations for different windows.
    """

    fig, axs = plt.subplots(
        ncols=len(rolled_data_list),
        figsize=(5 * len(rolled_data_list), fig_height),
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
        ax.set_xticklabels(labels, rotation=xtick_rotation)
        ax.grid(axis="x", color="k", alpha=0.2) # set vertical grid lines
        ax.set_title(f"{ndays}-day accumulated {parameter}", fontsize=18)

        # Highlight date window
        ylim = ax.get_ylim()
        dayofyear = event_date.dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000, color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        if ylimit is not None:
            ax.set_ylim(0, ylimit)

    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=0.1)
        return fig, ax, img_ax
    else:
        return fig, ax
