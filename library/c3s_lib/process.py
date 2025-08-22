from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd

# Rewrite so only the mean is calculated and not yet the anomoly
def calculate_anomaly_depreciated(final_gdf, gdfs, value_col="t2m", datetime_col='valid_time'):
    # Ensure datetime columns
    final_gdf = final_gdf.copy()
    gdfs = gdfs.copy()
    final_gdf[datetime_col] = pd.to_datetime(final_gdf[datetime_col])
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])
    
    # Prepare list for adjusted results
    adjusted_list = []
    
    for target_date in gdfs[datetime_col].unique():
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
        current_day_gdf = gdfs[gdfs[datetime_col] == target_date]
        
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
        crs=gdfs.crs
    )
    
    return final_adjusted_gdf




def calculate_mean_gdf(gdf, date_range, year_range:tuple[int, int]=None, padding: int = 15, value_col: str = "t2m", datetime_col: str = "valid_time"):
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




def calculate_anomaly(event_gdf, mean_climatology_gdf, value_col: str = "t2m", 
                      datetime_col: str = "valid_time", calcation: str = "subtract"):
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




def n_day_accumulations_gdf(data, value_col, days, centering=False, datetime_col="valid_time"):
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
    data = data.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    #data = data.sort_values(datetime_col)

    if value_col == "tp":
        data_nday = (
            data.set_index(datetime_col)[value_col]
            .rolling(days, min_periods=1, center=centering)
            .sum()
            .reset_index()
        )
    else:
        data_nday = (
            data.set_index(datetime_col)[value_col]
            .rolling(days, min_periods=1, center=centering)
            .mean()
            .reset_index()
        )

    return data_nday