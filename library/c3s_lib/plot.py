import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd


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
