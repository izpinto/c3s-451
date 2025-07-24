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

# 
def plot_gdf(gdf, borders=True, coastlines=True, gridlines=True, title=None, legend=True, legend_title=None, cmap='coolwarm', fig_size = (7,5), polygons:Polygon = None, projection=ccrs.PlateCarree()):
    
    fig, ax = plt.subplots(
        ncols = 1, nrows = 1, figsize = fig_size, dpi = 100, 
        subplot_kw = {"projection" : projection}
        )   

    # set color map
    temp_kwargs = {"cmap" : cmap}

    # Set the colorbar properties
    legend_title = legend_title if legend_title else "legend"

    # Plot the GeoDataFrame
    gdf.plot(ax = ax, **temp_kwargs,
        legend=legend, legend_kwds={'label': legend_title},
        column = 't2m',
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
    if polygons != None:
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, transform=projection)

    # add box around area of interest
    if title != None:
        ax.set_title(title)

    return fig, ax


def subplot_gdf(gdfs, datetime_col='valid_time', column='t2m', polygons=None, ncols=5, figsize=(20, 12), cmap='coolwarm', legend_title='Temperature (°C)', borders=True, coastlines=True, gridlines=True, suptitle=None, projection=ccrs.PlateCarree()):
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
    vmin = gdfs[column].min()
    vmax = gdfs[column].max()

    for i, day in enumerate(unique_days):
        ax = axes[i]

        # Filter GeoDataFrame for this day
        day_gdf = gdfs[gdfs[datetime_col].dt.date == day]

        # Plot data on this subplot
        day_gdf.plot(
            ax=ax,
            column=column,
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

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    #plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave room for suptitle and colorbar
    return fig, axes

def plot_poly(polygons:Polygon, coords, elevation=None):
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

  fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

  ax.set_extent([min_lon - 3, max_lon + 3, min_lat - 3, max_lat + 3], crs=ccrs.PlateCarree())
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
        transform=ccrs.PlateCarree(),
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
      ax.plot(x, y, color='red', linewidth=2, transform=ccrs.PlateCarree())
      ax.fill(x, y, color='red', alpha=0.3, transform=ccrs.PlateCarree())
  
  return fig, ax
