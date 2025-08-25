from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd


# this does not account for using 1990-12-16 and 2021-1-15 as edge cases for when the event is located at the start or end of the year
def calculate_mean_gdf(gdf:gpd.GeoDataFrame, date_range:pd.core.arrays.datetimes.DatetimeArray,
                       value_col:str, padding:int=15, year_range:tuple[int, int]=None,
                       datetime_col:str="valid_time"
                       ):
    """
    Calculate mean climatology values around each date in date_range across years,
    returning a GeoDataFrame with the same structure as the input.

    Parameters
    ----------
    gdf : GeoDataFrame
        Historical data with datetime and geometry.
    date_range : pd.DatetimeIndex
        Dates for which to calculate climatology means.
    padding : int
        +/- days for the averaging window.
    value_col : str
        Column with numeric values.
    datetime_col : str
        Column with datetimes.

    Returns
    -------
    GeoDataFrame
        Same structure as input: longitude, latitude, valid_time, value_col, geometry
    """
    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

    # Filter by year range if specified
    if year_range is not None:
        start_year, end_year = year_range
        gdf = gdf[(gdf[datetime_col].dt.year >= start_year) & (gdf[datetime_col].dt.year <= end_year)]

    climatology = []

    for target_date in date_range:
        results = []
        for year in gdf[datetime_col].dt.year.unique():
            center_date = datetime(year, target_date.month, target_date.day)
            start = center_date - timedelta(days=padding)
            end = center_date + timedelta(days=padding)

            subset = gdf[(gdf[datetime_col] >= start) & (gdf[datetime_col] <= end)]
            results.append(subset)

        combined = pd.concat(results, ignore_index=True)

        # Average per geometry
        mean_gdf = combined.groupby(["longitude", "latitude", "geometry"])[value_col].mean().reset_index()

        # Assign the target date as valid_time
        mean_gdf[datetime_col] = target_date

        climatology.append(mean_gdf)

    climatology_gdf = gpd.GeoDataFrame(
        pd.concat(climatology, ignore_index=True),
        geometry="geometry",
        crs=gdf.crs
    )
    return climatology_gdf




def calculate_anomaly(event_gdf:gpd.GeoDataFrame, mean_climatology_gdf:gpd.GeoDataFrame,
                      value_col:str, calcation:str, datetime_col:str="valid_time"
                      ):
    """
    Calculate anomalies by subtracting/dividing event data from climatology means.

    Parameters
    ----------
    event_gdf : GeoDataFrame
        Event data with same structure as climatology.
    mean_climatology_gdf : GeoDataFrame
        Mean climatology from `calculate_mean_gdf`.
    value_col : str
        Column with numeric values to adjust.
    datetime_col : str
        Column with datetimes.
    calc : str
        Operation: "subtract" or "divide".

    Returns
    -------
    GeoDataFrame
        Same structure as input with anomaly values in value_col.
    """
    event_gdf = event_gdf.copy()
    mean_climatology_gdf = mean_climatology_gdf.copy()

    # Ensure datetime is aligned
    event_gdf[datetime_col] = pd.to_datetime(event_gdf[datetime_col])
    mean_climatology_gdf[datetime_col] = pd.to_datetime(mean_climatology_gdf[datetime_col])

    # Merge on time + location
    merged = event_gdf.merge(
        mean_climatology_gdf[[ "longitude", "latitude", "geometry", datetime_col, value_col ]],
        on=["longitude", "latitude", "geometry", datetime_col],
        suffixes=("", "_mean")
    )

    # Apply calculation
    if calcation == "subtract":
        merged[value_col] = merged[value_col] - merged[f"{value_col}_mean"]
    elif calcation == "divide":
        merged[value_col] = merged[value_col] / merged[f"{value_col}_mean"]
    else:
        raise ValueError("calcation must be 'subtract' or 'divide'")

    # Drop helper column
    merged.drop(columns=[f"{value_col}_mean"], inplace=True)

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=event_gdf.crs)




def n_day_accumulations_gdf(gdf:gpd.GeoDataFrame, value_col:str, padding:int, centering:bool=False, datetime_col:str="valid_time"):
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