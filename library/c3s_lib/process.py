from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from typing import Union, Literal


# this does not account for dates that are outside of the range of the data such as januari 1-14 and december 18-31
def calculate_mean_gdf(gdf:gpd.GeoDataFrame, date_range:pd.core.arrays.datetimes.DatetimeArray,
                       value_col:str, padding:int=15, year_range:tuple[int, int]=None,
                       datetime_col:str="valid_time", group_by:list[str]=["longitude", "latitude", "geometry"]
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
                print(f"Skipping {center_date} as it is not in the data date range {date_min} to {date_max}")
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
        merged[value_col] = (merged[value_col] - merged[f"{value_col}_mean"]) / merged[f"{value_col}_mean"]
    else:
        raise ValueError("calcation must be 'subtract' or 'divide'")

    # Drop helper column
    merged.drop(columns=[f"{value_col}_mean"], inplace=True)

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=event_gdf.crs)





def n_day_accumulations_gdf(
    gdf: Union[pd.DataFrame, gpd.GeoDataFrame],
    value_col: str,
    padding: int,
    centering: bool = False,
    datetime_col: str = "valid_time",
    method: Literal["sum", "mean", "std", "quantile"] = "mean",
    quantile: float = 0.9,
    group_by: list[str] = None
):
    """
    Compute rolling n-day statistics (sum, mean, std, quantile) for each point
    or globally.

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
    quantile : float, optional
        Quantile to compute if method="quantile".
    group_by : list[str], optional
        Columns to group by before rolling. If None, roll globally.

    Returns
    -------
    DataFrame or GeoDataFrame
        Same as input, with rolled values in `value_col`.
    """

    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

    gdf = gdf[~((gdf[datetime_col].dt.month == 2) & (gdf[datetime_col].dt.day == 29))]

    if group_by is None:
        group_by = []  # roll globally

    def apply_roll(group):
        roller = group.set_index(datetime_col)[value_col].rolling(
            window=padding, min_periods=1, center=centering
        )
        if method == "sum":
            rolled = roller.sum()
        elif method == "mean":
            rolled = roller.mean()
        elif method == "std":
            rolled = roller.std()
        elif method == "quantile":
            rolled = roller.quantile(quantile)
        else:
            raise ValueError(f"Unsupported method: {method}")

        group[value_col] = rolled.values
        return group

    if group_by:
        # rolling separately for each group (e.g. per lat/lon/geometry)
        result = gdf.groupby(group_by, group_keys=False).apply(apply_roll)
    else:
        # single global time series
        roller = gdf.set_index(datetime_col)[value_col].rolling(
            window=padding, min_periods=1, center=centering
        )
        if method == "sum":
            rolled = roller.sum()
        elif method == "mean":
            rolled = roller.mean()
        elif method == "std":
            rolled = roller.std()
        elif method == "quantile":
            rolled = roller.quantile(quantile)
        else:
            raise ValueError(f"Unsupported method: {method}")

        result = rolled.reset_index()  # only datetime + value

    # preserve GeoDataFrame type if input was GeoDataFrame
    if isinstance(gdf, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)
    return result