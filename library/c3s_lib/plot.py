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
from rasterio import features
import numpy as np
from shapely import contains_xy
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import xarray as xr
import matplotlib.font_manager as fm
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm, TwoSlopeNorm
import matplotlib.colors as mcolors
import cmocean
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from IPython.display import display, clear_output
import re
from shapely.geometry import box

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
precipitation_colors = [
    "#f3f7fb",   "#deebf7",  "#c6dbef",   "#9ecae1",  "#6baed6",  "#4292c6",  
    "#1d609b",  "#08396b"
]
anomaly_colors = ["#693f18", "#B1967E", "#D2C7BE", "#ffffff", "#A6B4CB", "#7888A4" , "#204282"]


# create colormap from colors
temperature_cmap = ListedColormap(temperature_colors, name="temperature_cmap")
temperature_positive_cmap = ListedColormap(temperature_colors[5:], name="temperature_positive_cmap")
temperature_negative_cmap = ListedColormap(temperature_colors[:6], name="temperature_negative_cmap")
precipitation_cmap = LinearSegmentedColormap.from_list("precipitation_cmap", precipitation_colors, N=len(precipitation_colors))
anomaly_cmap = ListedColormap(anomaly_colors, name="anomaly_cmap")
anomaly_cmap = LinearSegmentedColormap.from_list(
    "anomaly_cmap",
    anomaly_colors,
    N=256
)
anomaly_positive_cmap = ListedColormap(anomaly_colors[1:], name="anomaly_positive_cmap")
anomaly_negative_cmap = ListedColormap(anomaly_colors[:3], name="anomaly_negative_cmap")

# get colormap normalization
def cmap_norm_boundary(vmin: int | float, vmax: int | float, steps: int):
    # Round limits to nearest integer to avoid decimals
    vmin_i, vmax_i = int(np.floor(vmin)), int(np.ceil(vmax))
    # Create integer step boundaries
    step_size = max(1, math.ceil((vmax_i - vmin_i) / steps))
    boundaries = np.arange(vmin_i, vmax_i + step_size, step_size)
    # Ensure we include the upper limit
    if boundaries[-1] < vmax_i:
        boundaries = np.append(boundaries, vmax_i)
    return BoundaryNorm(boundaries, len(boundaries) - 1)

def cmap_norm_twoslope(vmin, vmax, center):
    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

# set global style paramaters
def set_style(param:str, value:str|int|float):
    plt.rcParams[param] = value

def precip_bins(vmax: float):
    """
    Adaptiveprecipitation bins based on data range (mm/day or mm).
    """
    if vmax <= 15:
        return np.array([0, 1, 2, 3, 4, 6, 7, 8, 10])
    elif vmax <= 40:
        return np.array([0, 1, 2, 4, 6, 8, 10, 15, 20])
    elif vmax <= 75:
        return np.array([0, 5, 10, 15, 20, 25, 30, 40, 50])
    elif vmax <= 150:
        return np.array([0, 5, 10, 20, 30, 40, 60, 80, 100])
    elif vmax <= 300:
        return np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
    else:
        return np.array([0, 50, 100, 150, 200, 250, 300, 350, 400])


# get colormap
def get_colormap(map:str, vmin, vmax, value_col:str=None): 

    original_map = map

    match map:
        case 't2m':
            cmap = temperature_positive_cmap if vmin >= 0 else temperature_negative_cmap if vmax <= 0 else temperature_cmap
            norm = cmap_norm_boundary(vmin, vmax, 6) if vmin >= 0 else cmap_norm_boundary(vmin, vmax, 6) if vmax <= 0 else cmap_norm_boundary(vmin, vmax, 11)
            return cmap, norm
        case 'tp':
            boundaries = precip_bins(vmax)
            cmap = precipitation_cmap
            norm = BoundaryNorm(boundaries, len(boundaries) - 1)
            return cmap, norm
        case 'anomaly' | 'sst':
            if value_col in ["t2m", "sst"]:
                max_abs = max(abs(vmin), abs(vmax))
                cmap = temperature_cmap
                norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
                return cmap, norm
            else:
                cmap = anomaly_cmap
                vmax_adj = 0.7 * vmax
                if vmax_adj <= 0: # Prevent error when anomaly is all negative
                    vmax_adj = 1
                norm = cmap_norm_twoslope(vmin=-1, vmax=vmax_adj, center=0)
                return cmap, norm
        case _:
            return original_map, cmap_norm_boundary(vmin, vmax, 11)

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
             gridlines:bool=True, title:str|None=None, legend:bool=True, legend_title:str|None=None,
             cmap:str=None, fig_size:tuple[int, int]=(7,5), polygons:list[Polygon]|None=None,
             projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]|None=None,
             dpi:int=100, marker:str='s', add_logos:bool=True, polygon_color='cyan', ax=None):
    
    # get colormap
    gdfs_local = gdf.copy()

    if ax is None:
        fig, ax = plt.subplots(
            ncols = 1, nrows = 1, figsize = fig_size, dpi = dpi, 
            subplot_kw = {"projection" : projection}
            )
    else:
        fig = ax.figure

    if cmap == "anomaly" and value_col in ["tp"]:
        gdfs_local[value_col] = gdfs_local[value_col].clip(lower=-0.5, upper=None)

    # set color map   
    vmin = gdfs_local[value_col].min()
    vmax = gdfs_local[value_col].max()
    cmap, norm = get_colormap(cmap if cmap else value_col, vmin, vmax, value_col=value_col)

    # Plot the GeoDataFrame
    cell_size = 0.25  # degrees
    gdfs_local["geometry"] = gdfs_local.geometry.apply(
                lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                              p.x + cell_size/2, p.y + cell_size/2)
           )
    gdfs_local.plot(
        ax=ax, column=value_col, cmap=cmap, norm=norm,
        legend=False, marker=marker
    )

    # Add contextily basemap
    if gridlines:
        ax.gridlines(crs=projection, linewidth=0.5, color='black', draw_labels=["bottom", "left"], alpha=0.2)
    # Add coastlines
    if coastlines:
        ax.coastlines()
    # Add contextily basemap
    if borders:
        ax.add_feature(cartopy.feature.BORDERS, lw = 1, alpha = 0.7, ls = "--", zorder = 99)

    # Draw polygons if provided
    if polygons is not None:
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=polygon_color, linewidth=2, transform=projection)

    if title is not None:
        ax.set_title(title, fontdict={'fontsize': 27, 'fontweight': 'bold'})

    # Set extent if provided
    if extends is not None:
      ax.set_extent(extends, crs=projection)

    # Custom colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []

    if hasattr(norm, "boundaries") and norm.boundaries is not None:
        ticks = norm.boundaries
        tick_labels = [int(b) for b in ticks]
    else:
        if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
            ticks = np.linspace(norm.vmin, norm.vmax, len(anomaly_colors))
            for t in [0, -0.5]:
                if t not in ticks:
                    ticks = np.sort(np.append(ticks, t))  # ensure 0 is in there
            tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]
        else:
            ticks = np.linspace(norm.vmin, norm.vmax, 11)
            tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]

    cbar = fig.colorbar(
        sm, ax=ax, orientation='vertical', location="right",
        fraction=0.04, pad=.04, aspect=25, ticks=ticks
    )

    if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
        cbar.ax.set_ylim(-0.5, None)
            
    legend_title = legend_title if legend_title else value_col
    cbar.set_label(legend_title, labelpad=10, fontsize=20, weight='bold', color='#364563')
    cbar.set_ticklabels(tick_labels)
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
             family=roboto_condensed_regular.get_name(), fontsize=13)

    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=0)
        return fig, ax, img_ax
    else:
        return fig, ax


# adjust this so:
## shared colorbar is title + legend_title
## individual colorbars have legend_title and complete fig has title
def subplot_gdf(
    gdfs:gpd.GeoDataFrame, value_col:str, legend_title:str, datetime_col:str='valid_time',
    polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
    cmap:str=None, borders:bool=True, coastlines:bool=True, gridlines:bool=True,
    subtitle:str=None, projection:cartopy.crs=ccrs.PlateCarree(),
    extends:tuple[float, float, float, float]=None, dpi:int=100,
    flatten_empty_plots:bool=True, marker:str='s', shared_colorbar:bool=True,
    add_logos:bool=True, polygon_color='cyan'
):
    
    gdfs_local = gdfs.copy()
    # set cmap type
    cmap = cmap if cmap else value_col

    # Ensure datetime column is datetime type
    gdfs_local[datetime_col] = pd.to_datetime(gdfs_local[datetime_col])

    # Unique days sorted
    unique_days = sorted(gdfs_local[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with Cartopy projection
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi,
        subplot_kw={'projection': projection},
    )
    axes = axes.flatten()

    if cmap == "anomaly" and value_col in ["tp"]:
        gdfs_local[value_col] = gdfs_local[value_col].clip(lower=-0.5, upper=None)

    # Shared color scale (only if shared_colorbar=True)
    if shared_colorbar:
        vmin = gdfs_local[value_col].min()
        vmax = gdfs_local[value_col].max()
        cmap, norm = get_colormap(cmap, vmin, vmax, value_col=value_col)


    for i, day in enumerate(unique_days):
        ax = axes[i]
        day_gdf = gdfs_local[gdfs_local[datetime_col].dt.date == day]

        # Individual scale if not shared
        if not shared_colorbar:
            vmin = day_gdf[value_col].min()
            vmax = day_gdf[value_col].max()
            cmap, norm = get_colormap(cmap, vmin, vmax, value_col=value_col)

        # Plot
        cell_size = 0.25  # degrees
        day_gdf["geometry"] = day_gdf.geometry.apply(
                lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                              p.x + cell_size/2, p.y + cell_size/2)
            )
        
        day_gdf.plot(
            ax=ax, column=value_col, cmap=cmap,
            legend=False, vmin=vmin, vmax=vmax,
            marker=marker, norm=norm
        )

        if not shared_colorbar:
            # Add colorbar per axis
            #norm = cmap_norm_boundary(vmin, vmax, 11)

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
                ax.plot(x, y, color=polygon_color, linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=18, weight='medium')

        if extends is not None:
            ax.set_extent(extends, crs=projection)

    fig.subplots_adjust(wspace=0.4)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Shared colorbar if requested
    if shared_colorbar:
        #norm = cmap_norm_boundary(vmin, vmax, 11)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []

        # Handle any normalization type
        if hasattr(norm, "boundaries") and norm.boundaries is not None:
            # BoundaryNorm (discrete bins)
            ticks = norm.boundaries
            tick_labels = [int(b) for b in ticks]
        else:
            if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
                # Generate ticks that always include 0
                ticks = np.linspace(norm.vmin, norm.vmax, len(anomaly_colors))
                for t in [0, -0.5]:
                    if t not in ticks:
                        ticks = np.sort(np.append(ticks, t))  # ensure 0 is in there
                tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]
            else:
                # Normal case (temperature anomaly, etc.)
                if vmin >= 0:
                    n_ticks = temperature_positive_cmap.N + 1
                elif vmax <= 0:
                    n_ticks = temperature_negative_cmap.N + 1
                else:
                    n_ticks = temperature_cmap.N + 1

                ticks = np.linspace(norm.vmin, norm.vmax, n_ticks)
                tick_labels = [round(t, 1) for t in ticks]

        cbar = fig.colorbar(
            sm, ax=axes.tolist(), 
            orientation='horizontal', location="top",
            fraction=0.01, pad=.07, aspect=60, ticks=ticks)
        
        if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
            cbar.ax.set_xlim(-0.5, None)
        # Label for colorbar
        cbar.set_label(legend_title, labelpad=10, fontsize=27, weight='bold', color='#364563')
        # cbar.set_ticks(np.round(norm.boundaries).astype(int))
        # Make tick labels clean integers
    
        cbar.set_ticklabels(tick_labels)
        # Font settings
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




def plot_poly(polygons:list[Polygon], coords:list[list[float]], 
              layer:xr.DataArray=None, cmap=None, norm=None, layer_type:str="elevation",
              projection:cartopy.crs=ccrs.PlateCarree()):

    lons, lats = zip(*coords)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    if layer is not None:
        layer_subset = layer.sel(
            lon=slice(min_lon-3, max_lon+3),
            lat=slice(min_lat-3, max_lat+3)
        )

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.set_extent([min_lon - 3, max_lon + 3, min_lat - 3, max_lat + 3], crs=projection)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(draw_labels=True)

    cax = inset_axes(
        ax, width="3%", height="100%", loc='center left',
        bbox_to_anchor=(1.1, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0
    )

    if layer is not None:
        if layer_type == "elevation":
            layer_subset.plot(
                ax=ax, transform=projection, cmap="terrain",
                cbar_ax=cax, cbar_kwargs={"label": "Elevation (m)"},
                add_labels=False
            )
        elif layer_type == "koppen":
            layer_subset.plot(
                ax=ax, transform=projection, cmap=cmap, norm=norm,
                cbar_ax=cax, cbar_kwargs={"label": "Köppen-Geiger class"},
                add_labels=False
            )

    ax.set_title("Selected region")

    # Plot selected polygons
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





def elevation_region(data:dict, polygons:list[Polygon], elevation:xr.DataArray, threshold:int=None, projection:cartopy.crs=ccrs.PlateCarree()):

    threshold = threshold if threshold else 100000

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
                   add_logos:bool=True, center_month_labels:bool=False, full_month_names:bool=False, ax=None):
    

    # plot_df = data.copy()
    # plot_df[datetime_col] = pd.to_datetime(plot_df[datetime_col])
    # plot_df["plot_time"] = plot_df[datetime_col]

    # start_month, end_month = month_range

    # # Determine if the period crosses the year boundary
    # crosses_year = (end_month < start_month)

    # # Adjust months so they plot in correct chronological order
    # if crosses_year:
    #     # For ranges like (7,6) or (9,3): shift early months (those before start_month) forward by one year
    #     early_mask = plot_df["plot_time"].dt.month < start_month
    #     plot_df.loc[early_mask, "plot_time"] += pd.DateOffset(years=1)

    # # Sort chronologically after shifting
    # plot_df = plot_df.sort_values("plot_time").reset_index(drop=True)

    # # ----- Create label ticks -----
    # # Define the logical start month (the first month of the period)
    # if crosses_year:
    #     # e.g. (7,6) → start in July 2024 and wrap to June 2025
    #     label_start = pd.Timestamp(f"2024-{start_month:02d}-01")
    # else:
    #     # e.g. (1,6) or (3,9): simple one-year span
    #     label_start = pd.Timestamp(f"2024-{start_month:02d}-01")

    # # Always 12 months long
    # labelticks = pd.date_range(label_start, periods=12, freq="MS")
    # labels = labelticks.strftime("%b")


    #set font family globally
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    else:
        fig = ax.figure

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

    # Center month labels
    if center_month_labels:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
        fmt = "%B" if full_month_names else "%b"
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

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
        img_ax = None
        return fig, ax, img_ax





def plot_n_days(
    rolled_data_list:list[gpd.GeoDataFrame], value_col:str, parameter:str, event_date:datetime,
    labelticks:list[int], labels:list[any], days:list[int], title:str,
    datetime_col:str="valid_time", add_logos:bool=True, fig_height:int=3, xtick_rotation:int=0, ncols:int=0
):
    """
    Plot n-day rolling accumulations for different windows.
    """

    # fig size
    nplots = len(rolled_data_list)
    ncols = ncols if ncols else nplots
    nrows = math.ceil(nplots/ncols)

    if value_col == 'tp':
        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(5 * ncols, fig_height * nrows),
            dpi=100,
            sharey=False
        )
    else:
        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(5 * ncols, fig_height * nrows),
            dpi=100,
            sharey=True
        )

    axs = np.array(axs).reshape(-1)

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
        ax.set_title(f"{ndays}-day {title}", fontsize=18)

        # Highlight date window
        ylim = ax.get_ylim()
        dayofyear = event_date.dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000, color="gold", alpha=0.3))
        ax.set_ylim(ylim)


    fig.subplots_adjust(hspace=0.4)

    for ax in axs[nplots:]:
        fig.delaxes(ax)
    axs = axs[:nplots]

    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=0)
        return fig, ax, img_ax
    else:
        return fig, ax



def subplot_contours(
    contour_gdf:gpd.GeoDataFrame, gdf:gpd.GeoDataFrame, value_col:str, contour_col:str,
    legend_title:str=None, datetime_col:str="valid_time",
    polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int,int]=(13,10),
    cmap:str|None=None, borders:bool=True, coastlines:bool=True, gridlines:bool=True,
    subtitle:str=None, extends:tuple[float,float,float,float]=None, dpi:int=100,
    flatten_empty_plots:bool=True, marker:str='s', shared_colorbar:bool=True,
    add_logos:bool=False, polygon_color:str='cyan', contour_steps:int=200,
    projection:cartopy.crs=ccrs.PlateCarree(), grid_line_col:str='gray', grid_line_size:float=.4,
    grid_line_alpha:float=.5
):
    """
    Plot daily values from a GeoDataFrame with Z500 contours from another GeoDataFrame.
    """

    # set cmap type
    cmap = cmap if cmap else value_col

    # Ensure datetime
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    contour_gdf[datetime_col] = pd.to_datetime(contour_gdf[datetime_col])

    # Unique sorted days
    unique_days = sorted(contour_gdf[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Projection fixed to LambertConformal
    mid_lon = contour_gdf["longitude"].mean()
    mid_lat = contour_gdf["latitude"].mean()
    proj = ccrs.LambertConformal(central_longitude=mid_lon, central_latitude=mid_lat)

    # Figure
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi,
        subplot_kw={"projection": proj}
    )
    axes = axes.flatten()

    # Shared color scale
    if shared_colorbar:
        vmin = gdf[value_col].min()
        vmax = gdf[value_col].max()
        cmap, norm = get_colormap(cmap, vmin, vmax, value_col=value_col)
    else:
        norm = None

    # contour contour levels
    zmin, zmax = contour_gdf[contour_col].min(), contour_gdf[contour_col].max()
    z_lev = np.arange(round(zmin), round(zmax), contour_steps)

    for i, day in enumerate(unique_days):
        ax = axes[i]
        day_gdf = gdf[gdf[datetime_col].dt.date == day]

        if not shared_colorbar:
            vmin = day_gdf[value_col].min()
            vmax = day_gdf[value_col].max()
            cmap, norm = get_colormap(cmap, vmin, vmax, value_col=value_col)

        if not day_gdf.empty:
            cell_size = 0.25  # degrees
            day_gdf["geometry"] = day_gdf.geometry.apply(
                lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                              p.x + cell_size/2, p.y + cell_size/2)
            )
            day_gdf.plot(
                ax=ax, column=value_col, cmap=cmap,
                legend=False, vmin=vmin, vmax=vmax,
                norm=norm, marker=marker,
                transform=projection
            )

        # --- Z500 contours ---
        contour_day = contour_gdf[contour_gdf[datetime_col].dt.date == day]
        if not contour_day.empty:
            pivot = contour_day.pivot_table(index="latitude", columns="longitude", values=contour_col)
            lon, lat, Z = pivot.columns.values, pivot.index.values, pivot.values
            cn = ax.contour(lon, lat, Z, z_lev, colors="dimgray", linewidths=0.5, transform=projection)
            ax.clabel(cn, inline=1)

        if not shared_colorbar:
            # Add colorbar per axis
            #norm = cmap_norm_boundary(vmin, vmax, 11)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04, ticks=norm.boundaries)
            cbar.set_label(legend_title)

        # Map features

        if gridlines:
            gl = ax.gridlines(
                draw_labels=False, x_inline=False, y_inline=False,
                linewidth=grid_line_size, color=grid_line_col, alpha=grid_line_alpha
            )
            gl.right_labels = gl.top_labels = False
            # gl.xlabel_style = {"size": 8, "color": grid_line_col}
            # gl.ylabel_style = {"size": 8, "color": grid_line_col}

        if coastlines:
            ax.coastlines(resolution="50m", color="black", linewidth=0.5, alpha=0.7)
        if borders:
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor="black", linewidth=0.5)

        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=polygon_color, linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=18, weight='medium')

        if extends is not None:
            ax.set_extent(extends, crs=projection)


    fig.subplots_adjust(hspace=.25, wspace=.05)
    fig.tight_layout(pad=0.5)

    # Hide empty plots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Shared colorbar
    if shared_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []        
        cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top",
                            fraction=0.01, pad=.07, aspect=60, ticks=norm.boundaries)
        cbar.set_label(legend_title if legend_title else value_col,
                       labelpad=10, fontsize=27, weight='bold', color='#364563')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), family=roboto_condensed_regular.get_name(), fontsize=13)
        plt.show()

    if subtitle:
        fig.suptitle(subtitle, y=1.05)

    # fig.subplots_adjust(wspace=0.25, hspace=.45, top=.85)
    fig.tight_layout()

    if add_logos:
        plt.close(fig)
        fig, img_ax = add_image_below(fig=fig, image_path=logo_horizon_path, pad_frac=-0.1)
        return fig, axes, img_ax
    else:
        return fig, axes

def plot_koppen_geiger(
    kg_da: xr.DataArray,
    polygons: list[Polygon],
    coords: list[list[float]],
    legend_path: str,
    projection=ccrs.PlateCarree(),
    fontsize: int = 8,
    figsize=(10, 10),
    extra_polygons: list[Polygon] = None
):
    """
    Plot Köppen–Geiger classifications for a selected region and polygons
    with a grouped, 4-column legend at the bottom.
    """

    # =============================
    # 1. Load legend and colormap
    # =============================

    def load_kg_legend(path):
        rows = []
        pattern = re.compile(r"^\s*(\d+):\s+(\w+)\s+(.*?)\s+\[(.*?)\]")

        with open(path, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    code   = int(match.group(1))
                    klass  = match.group(2)
                    desc   = match.group(3).strip()
                    rgb    = tuple(int(v)/255 for v in match.group(4).split())
                    rows.append((code, klass, desc, rgb))

        return pd.DataFrame(rows, columns=["code", "class", "description", "rgb"])

    def draw_koppen_legend(fig, kg_legend, fontsize=8):
        """
        Draw a Köppen-Geiger legend at the bottom of the figure in 4 columns.
        Classes grouped by main climate categories.
        """
        
        KG_GROUPS = {
        "Tropical": [1,2,3],
        "Arid": [4,5,6,7],
        "Temperate": list(range(8,17)),
        "Cold": list(range(17,29)),
        "Polar": [29,30]
        }

        # Bottom inset axes for legend
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_axes([0.08, 0.04, 0.85, 0.3])
        ax.axis("off")
        ax.set_frame_on(False)

        grouped_rows = []
        for group, codes in KG_GROUPS.items():
            # Group header (rgb=None)
            grouped_rows.append((None, group))

            # Rows for each class in group
            subset = kg_legend[kg_legend["code"].isin(codes)]
            for _, r in subset.iterrows():
                text = f"{r['class']} — {r['description']}"
                grouped_rows.append((r["rgb"], text))
        
        # Split into 4 columns
        num_cols = 4
        rows_per_col = int(np.ceil(len(grouped_rows) / num_cols))

        for col in range(num_cols):
            col_data = grouped_rows[col * rows_per_col:(col + 1) * rows_per_col]
            if col == 1:
                x0 = 0.02 + col * 0.20
                y = 0.5
            else:
                x0 = 0.02 + col * 0.28
                y = 0.5

            for rgb, text in col_data:

                if rgb is None:  # Group header
                    ax.text(x0, y, text, fontweight="bold", fontsize=fontsize+5, va="top")
                    y -= 0.065
                    continue

                # Small color box
                ax.add_patch(mpatches.Rectangle(
                    (x0, y - 0.03), 0.02, 0.03,
                    color=rgb, ec="black", lw=0.3
                ))

                ax.text(x0 + 0.03, y, text, fontsize=fontsize, va="top")
                y -= 0.05

        return ax
    kg_legend = load_kg_legend("../data/legend.txt")
    kg_da_masked = kg_da.where(kg_da >= 1)


    # Build listed colormap & norm
    kg_colors = [tuple(rgb) for rgb in kg_legend["rgb"]]
    kg_cmap = mcolors.ListedColormap(kg_colors)
    kg_norm = mcolors.BoundaryNorm(list(kg_legend["code"]) + [31], kg_cmap.N)

    fig, ax = plot_poly(polygons, coords, layer=kg_da_masked, cmap=kg_cmap, norm=kg_norm, layer_type="koppen")
    if extra_polygons is not None and len(extra_polygons) > 0:
        for poly in extra_polygons:
            x, y = poly.exterior.xy
            ax.fill(
                x, y,
                color="green", alpha=0.1,
                transform=projection,
                zorder=1  # lower -> below main polygons
            )
            ax.plot(
                x, y,
                color="green", linewidth=2,
                transform=projection,
                label="Original region",
                zorder=1
            )
    handles = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor="white", linewidth=2, edgecolor="green", label="Original region"),
        mpatches.Rectangle((0, 0), 1, 1, facecolor="white", linewidth=2, edgecolor="red", label="Filtered region")
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fontsize=fontsize,
        title="Regions",
        title_fontsize=fontsize + 1
    )


    draw_koppen_legend(fig, kg_legend)

    return fig, ax