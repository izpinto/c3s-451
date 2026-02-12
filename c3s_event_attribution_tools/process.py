from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import pandas as pd
import geopandas as gpd
from typing import Any, Union, Literal, Dict, List
import numpy as np

from .data import conversions
from .utils import Utils
import rasterio
from shapely.geometry import Polygon, shape
import xarray as xr
from shapely import contains_xy
import re
import regionmask
from scipy.stats import chi2
from rpy2.robjects import r


class Process:

    # Don't use
    @staticmethod
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


    @staticmethod
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
        if calcation == "absolute":
            merged[value_col] = merged[value_col] - merged[f"{value_col}_mean"]
        elif calcation == "relative":
            merged[value_col] = (merged[value_col] - merged[f"{value_col}_mean"]) / merged[f"{value_col}_mean"]
        else:
            raise ValueError("calcation must be 'absolute' or 'relative'")

        # Drop helper column
        merged.drop(columns=[f"{value_col}_mean"], inplace=True)

        return gpd.GeoDataFrame(merged, geometry="geometry", crs=event_gdf.crs)




    @staticmethod
    def calculate_rolling_n_days(
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

        if padding <= 1:
            return gdf

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



    @staticmethod
    def calculate_rolling_window(
        gdf: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_col: str,
        window: int,
        centering: bool = False,
        datetime_col: str = "valid_time",
        method: Literal["sum", "mean", "std", "quantile", "dispersion"] = "mean",
        quantile: float = 0.9,
        group_by: list[str] = None,
        min_periods: int|None = 1,
        remove_leap_days: bool = False,
        ci: bool = False,
        ci_level: float = 0.95,
    ):
        """
        Compute rolling statistics (sum, mean, std, quantile) over a fixed-size window.
        Assumes datetime_col is already at the desired temporal resolution (days, months, or years).

        Returns
        -------
        DataFrame or GeoDataFrame
            Same as input, with rolled values in `value_col`.
        """

        if window <= 1:
            return gdf

        gdf = gdf.copy()

        if remove_leap_days:
            # Ensure datetime_col is datetime (should already be if user promised so)
            gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

            # Drop Feb 29 for yearly data consistency (optional)
            gdf = gdf[~((gdf[datetime_col].dt.month == 2) & (gdf[datetime_col].dt.day == 29))]

        if group_by is None:
            group_by = []

        def apply_roll(group):
            group = group.sort_values(datetime_col)
            roller = group[value_col].rolling(window=window, min_periods=min_periods, center=centering)

            match method:
                case "sum":
                    rolled = roller.sum()
                case "mean":
                    rolled = roller.mean()
                case "std":
                    rolled = roller.std()
                case "quantile":
                    rolled = roller.quantile(quantile)
                case "dispersion":
                    rolled = roller.std() / roller.mean()
                case _:
                    raise ValueError(f"Unsupported method: {method}")

            group[value_col] = rolled
            
            if ci and method in ("std", "dispersion"):
                n = roller.count().reset_index(drop=True)
                alpha = 1 - ci_level
                var = roller.var().reset_index(drop=True)

                chi2_lower = chi2.ppf(1 - alpha / 2, n - 1)
                chi2_upper = chi2.ppf(alpha / 2, n - 1)
                std_lower = np.sqrt((n - 1) * var / chi2_lower)
                std_upper = np.sqrt((n - 1) * var / chi2_upper)

                if method == "std":
                    group[f"{value_col}_ci_lower"] = std_lower
                    group[f"{value_col}_ci_upper"] = std_upper

                elif method == "dispersion":
                    mean_vals = roller.mean().reset_index(drop=True)
                    mean_vals = mean_vals.replace(0, np.nan)
                    group[f"{value_col}_ci_lower"] = std_lower / mean_vals
                    group[f"{value_col}_ci_upper"] = std_upper / mean_vals

            return group

        if group_by:
            result = gdf.groupby(group_by, group_keys=False).apply(apply_roll)
        else:
            result = apply_roll(gdf)

        # Preserve GeoDataFrame type if input was one
        if isinstance(gdf, gpd.GeoDataFrame):
            return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)
        return result



    @staticmethod
    def calculate_climatology(gdf, value_col:str, event_date:datetime, padding:int=15, datetime_col:str="valid_time") -> gpd.GeoDataFrame:
        """
        Parameters:
        -----------
        data : ...
        pad : int
            Number of days on either side to include in the window (default: 15 → 30-day window)

        Returns:
        --------
        """

        gdf = gdf.copy()

        # Ensure 'time' column is datetime
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

        # Boolean mask: keep all rows NOT Feb 29
        mask = ~((gdf[datetime_col].dt.month == 2) & (gdf[datetime_col].dt.day == 29))

        # Apply mask
        gdf = gdf[mask]

        days = np.arange(1, 366)  # Days of year

        gdf['doy'] = gdf[datetime_col].dt.dayofyear

        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y

        result_list = []

        for day in days:
            # Build ±pad-day window (cyclically)
            window_days = [(day + offset - 1) % 365 + 1 for offset in range(-padding, padding + 1)]

            window_data = gdf[gdf['doy'].isin(window_days)]

            # Compute mean at each location
            daily_mean = (
                window_data.groupby(['longitude', 'latitude'])[value_col]
                .mean()
                .reset_index()
            )

            daily_mean['doy'] = day
            result_list.append(daily_mean)
            doy_mean_gdf = pd.concat(result_list, ignore_index=True)

        # Turn dataframe back into a gpd
        return_gdf = gpd.GeoDataFrame(doy_mean_gdf, geometry=gpd.points_from_xy(doy_mean_gdf.longitude, doy_mean_gdf.latitude), crs=gdf.crs)

        # turn doy (day of year) column back into datetime column
        return_gdf[datetime_col] = pd.to_datetime(f'{event_date.year}', format='%Y') + pd.to_timedelta(return_gdf['doy'] - 1, unit='D')
        #return_gdf["valid_time"] = pd.to_datetime(return_gdf["doy"], format="%j").dt.strftime("%m-%d")


        return return_gdf



    # apply a weight to each value based on latitude
    @staticmethod
    def weighted_values(df:gpd.GeoDataFrame|pd.DataFrame|xr.DataArray|xr.Dataset, value_col:str, lat_col:str='latitude'
                        ) -> gpd.GeoDataFrame|pd.DataFrame|Any:
        
        '''
        Calculates the weights for the values based on latitude

        Parameters:
            df (gpd.GeoDataFrame|pd.DataFrame|xr.DataArray|xr.Dataset):
                Dataframe
            value_col (str):
                Value column needed for 2D dataframes, can be whatever for 3D dataframes
            lat_col (str):
                Name for the latitude column/coordinate

        Returns:
            gpd.GeoDataFrame|pd.DataFrame|xr.core.weighted.Weighted:
                2D arrays return df with a new column 'value_col'+'_weighted'. xarray is returned as 'xarray.core.weighted.Weighted'
        '''

        if isinstance(df, (xr.DataArray, xr.Dataset)):
            if lat_col not in df.coords:
                raise ValueError(f"Latitude coordinate '{lat_col}' not found")
            
            weights = np.cos(np.deg2rad(df[lat_col]))

            return df.weighted(weights)
        
        if isinstance(df, (gpd.GeoDataFrame)):
            if value_col not in df.columns:
                raise ValueError(f"Column '{value_col}' not found")
            if lat_col not in df.columns:
                raise ValueError(f"Latitude coordinate '{lat_col}' not found")

            out_col = f"{value_col}_weighted"

            df = df.copy()
            weights = np.cos(np.deg2rad(df[lat_col]))
            df[out_col] = df[value_col] * weights
            df["_weight"] = weights

            return df

        raise TypeError(
            "weighted_values expects a GeoDataFrame, DataFrame, "
            "xarray DataArray, or xarray Dataset"
            )

    @staticmethod
    def calculate_mean(gdf:gpd.GeoDataFrame, value_col:str, groupby_col:str) -> gpd.GeoDataFrame:

        is_spatial = 'longitude' in groupby_col and 'latitude' in groupby_col and 'geometry' in groupby_col
        crs = gdf.crs if is_spatial else None

        if '_weight' in gdf.columns:
            gdf = (
                gdf.groupby(groupby_col)
                .apply(lambda x: x[f"{value_col}_weighted"].sum() / x["_weight"].sum())
                # .reset_index()
                .reset_index(name=value_col)
            )
        else:
            gdf = gdf.groupby(groupby_col)[value_col].mean().reset_index()

        if is_spatial:
            gdf = gpd.GeoDataFrame(gdf,geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude), crs=crs)

        return gdf
    
    # J: So I don't know if this works anymore for spatial data. Had to change this because the old function (above) wasn't working for the trend analysis for some reason
    # J: or if the output is even similar to the original
    @staticmethod
    def calculate_mean(gdf: gpd.GeoDataFrame, value_col: str, groupby_col: str) -> gpd.GeoDataFrame:

        # Check if GeoDataFrame
        is_spatial = 'longitude' in groupby_col and 'latitude' in groupby_col and 'geometry' in groupby_col
        crs = gdf.crs if is_spatial else None

        if '_weight' in gdf.columns:
            # Weighted mean
            weighted_mean = gdf.groupby(groupby_col).apply(
                lambda x: x[f"{value_col}_weighted"].sum() / x["_weight"].sum()
            )

            # Convert to DataFrame safely
            gdf_result = weighted_mean.reset_index()  # now each column is 1D
            gdf_result = gdf_result.rename(columns={0: value_col})

        else:
            # Unweighted mean
            gdf_result = gdf.groupby(groupby_col)[value_col].mean().reset_index()

        if is_spatial:
            # Recreate geometry if needed
            gdf_result = gpd.GeoDataFrame(
                gdf_result, 
                geometry=gpd.points_from_xy(gdf_result.longitude, gdf_result.latitude), 
                crs=crs
            )

        return gdf_result


    @staticmethod
    def calculate_max(gdf:gpd.GeoDataFrame, value_col:str, datetime_col:str, groupby_col:str) -> gpd.GeoDataFrame:

        return (
            gdf.loc[gdf.groupby(groupby_col)[value_col].idxmax(), [groupby_col, datetime_col, value_col]]
            .reset_index(drop=True)
        )

    @staticmethod
    def calculate_min(gdf:gpd.GeoDataFrame, value_col:str, datetime_col:str, groupby_col:str) -> gpd.GeoDataFrame:

        return (
            gdf.loc[gdf.groupby(groupby_col)[value_col].idxmin(), [groupby_col, datetime_col, value_col]]
            .reset_index(drop=True)
        )

    @staticmethod
    def calculate_yearly_value(gdf:gpd.GeoDataFrame, value_col:str, datetime_col:str,
                               yearly_value:str, month_range:tuple[int, int]|None=None,
                               padding:int=0, method:str='mean') -> gpd.GeoDataFrame:

        # if month_range is present select a subset of the gdf
        # if month_range is not None:
        #     start_date = pd.Timestamp(year=2001, month=month_range[0], day=1)
        #     end_date = pd.Timestamp(year=2001, month=month_range[1], day=1) + pd.offsets.MonthEnd(1)

        #     # if padding is > 1 add padding to subset
        #     if padding > 1:

        #         start_date = start_date - pd.Timedelta(days=padding)
        #         end_date = end_date + pd.Timedelta(days=padding)

        #     start_doy = start_date.timetuple().tm_yday
        #     end_doy = end_date.timetuple().tm_yday

        #     # subset the gdf
        #     gdf = subset_gdf(gdf=gdf, datetime_col=datetime_col, doy_range=(start_doy, end_doy))

        # calculate running mean. if padding == 1 gdf gets automatically returned
        rolled_gdf = Process.calculate_rolling_n_days(gdf=gdf, value_col=value_col, datetime_col=datetime_col,
                                            padding=padding, centering=True, method=method)

        # subset the gdf to remove potential padding
        if month_range is not None:
            rolled_gdf = Utils.subset_gdf(gdf=rolled_gdf, datetime_col=datetime_col, month_range=month_range)

        # add years to the gdf to calculate yearly values
        rolled_gdf = Utils.add_year_column(gdf=rolled_gdf, datetime_col=datetime_col)

        #change name to valid_time colum to f{padding}_day rolling date
        new_datetime_col = f"{padding}_day_rolling_date"
        rolled_gdf = rolled_gdf.rename(columns={datetime_col: new_datetime_col})


        match yearly_value:
            case 'mean':
                return Process.calculate_mean(gdf=rolled_gdf, value_col=value_col, groupby_col='year'), rolled_gdf
            case 'max':
                return Process.calculate_max(gdf=rolled_gdf, value_col=value_col, datetime_col=new_datetime_col, groupby_col='year'), rolled_gdf
            case 'min':
                return Process.calculate_min(gdf=rolled_gdf, value_col=value_col, datetime_col=new_datetime_col, groupby_col='year'), rolled_gdf
            case _:
                raise ValueError("calculation must be 'mean', 'max', or 'min'")

    @staticmethod
    def calculate_seasonal_cycle(clim31d: gpd.GeoDataFrame,
                            studyregion: gpd.GeoDataFrame | dict,
                            month_range: tuple[int, int],
                            value_col: str,
                            datetime_col: str,
                            event_end: pd.Timestamp
    ):
        gdf_sub = Utils.subset_gdf(gdf=clim31d, study_region=studyregion, month_range=month_range)
        gdf_sub[datetime_col] = (
            pd.to_datetime(f"{event_end.year}", format="%Y")
            + pd.to_timedelta(gdf_sub["doy"] - 1, unit="D")
        )
        gdf_weighted = Process.weighted_values(gdf_sub, value_col)
        ts_clim31d_studyregion = Process.calculate_mean(gdf_weighted, value_col=value_col, groupby_col=datetime_col)
        plot_df, labels, labelticks = Utils.get_seasonal_cycle_plot_values(
            ts_clim31d_studyregion, datetime_col=datetime_col, month_range=month_range
        )
        return ts_clim31d_studyregion, plot_df, labels, labelticks

    @staticmethod
    def filter_polygons_by_kg(
        polygons: List[Polygon],
        kg_da: xr.DataArray,
        category,
        invert: bool = True  # True → exclude category, False → keep only category
    ):
        """
        Filter polygons based on Köppen–Geiger classification raster.

        Parameters
        ----------
        polygons : list of shapely.Polygon
            Input polygons to filter.
        kg_da : xarray.DataArray
            Köppen–Geiger classification raster (integer-coded).
        category : str or list of str
            Category or categories to filter (e.g. 'Tropical', ['Arid','Temperate']).
        invert : bool, optional
            If True, exclude the given categories (keep everything else).
            If False, keep only the given categories.

        Returns
        -------
        adjusted_polygons : list of shapely.Polygon
            Polygons after Köppen–Geiger filtering.
        """
        KG_GROUPS = {
            "Tropical": [1,2,3],
            "Arid": [4,5,6,7],
            "Temperate": list(range(8,17)),
            "Cold": list(range(17,29)),
            "Polar": [29,30]
        }

        # Normalize category input
        if isinstance(category, str):
            category = [category]

        # Combine all selected codes
        codes = []
        for c in category:
            codes.extend(KG_GROUPS[c])

        adjusted_polygons = []

        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds

            # Subset Köppen raster to polygon extent
            kg_subset = kg_da.sel(
                lon=slice(minx-0.5, maxx+0.5),
                lat=slice(miny-0.5, maxy+0.5)
            ).squeeze()

            kg_vals = kg_subset.values

            # Ensure lat orientation is north-down
            if kg_subset.lat.values[0] < kg_subset.lat.values[-1]:
                kg_vals = kg_vals[::-1, :]
                lat = kg_subset.lat.values[::-1]
            else:
                lat = kg_subset.lat.values

            lon = kg_subset.lon.values
            lon2d, lat2d = np.meshgrid(lon, lat)

            inside_poly = contains_xy(poly, lon2d, lat2d)

            # Mask depending on invert
            if invert:
                # exclude given categories
                mask = ~np.isin(kg_vals, codes) & inside_poly
            else:
                # keep only given categories
                mask = np.isin(kg_vals, codes) & inside_poly

            transform = rasterio.transform.from_bounds(
                lon.min(), lat.min(),
                lon.max(), lat.max(),
                len(lon), len(lat)
            )

            for geom, val in rasterio.features.shapes(
                    mask.astype(np.uint8),
                    mask=mask,
                    transform=transform):

                if val == 1:
                    new_poly = shape(geom)
                    clipped_poly = new_poly.intersection(poly)

                    if not clipped_poly.is_empty:
                        if clipped_poly.geom_type == "Polygon":
                            adjusted_polygons.append(clipped_poly)
                        elif clipped_poly.geom_type == "MultiPolygon":
                            adjusted_polygons.extend(list(clipped_poly.geoms))

        return adjusted_polygons

    @staticmethod
    def calculate_yearly_value_xr(
        time_series: dict[str, xr.DataArray],
        yearly_value: str,              # options: "max", "mean", "min"
        month_range: tuple[int, int]|None = None,  # e.g., (6, 8) for June-August, None for all months
        padding: int=0,                   # n-day rolling window
        method: str = None              # "mean" or "sum"
    ) -> dict[str, xr.DataArray]:
        """
        Iterates over a dictionary of DataArrays, applies a rolling window and 
        resamples to yearly values (max, mean, or min) in a single step.
        """
        da_yearly_series = {}

        for name, da in time_series.items():
            # Determine method automatically for this specific DataArray if not provided
            current_method = method
            if current_method is None:
                # Check variable name safely
                var_name = da.name.lower() if da.name else ""
                if any(k in var_name for k in ["tp", "precip", "pr"]):
                    current_method = "sum"
                else:
                    current_method = "mean"

            # Rolling window on the daily series
            if padding is not None and padding > 1:
                if current_method == "mean":
                    da_rolled = da.rolling(time=padding, center=True).mean()
                elif current_method == "sum":
                    da_rolled = da.rolling(time=padding, center=True).sum()
                else:
                    raise ValueError(f"Method must be 'mean' or 'sum', got '{current_method}'")
            else:
                da_rolled = da

            # subset the gdf to remove potential padding
            if month_range is not None:
                start_month, end_month = month_range
                da_rolled = da_rolled.sel(time=da_rolled.time.dt.month.isin(range(start_month, end_month + 1)))

            # Compute yearly statistic on the rolled series
            if yearly_value == "max":
                da_year = da_rolled.resample(time="YE").max()
            elif yearly_value == "mean":
                da_year = da_rolled.resample(time="YE").mean()
            elif yearly_value == "min":
                da_year = da_rolled.resample(time="YE").min()
            else:
                raise ValueError(f"yearly_value must be: max, mean, min. Got '{yearly_value}'")

            # 4. Store result
            da_yearly_series[name] = da_year

        return da_yearly_series
    
    @staticmethod
    def build_cordex_model_pairs(catalog_file: Path, domain_input, experiments=("hist", "rcp85"), temporal_filter: str = "day"):
        """
        Finds models for specific domains or domain prefixes (e.g., 'EUR' finds 'EUR-11').
        Prioritizes highest resolution (lowest number) for each GCM-RCM pair.
        """
        # Normalize input to a list of uppercase prefixes
        if isinstance(domain_input, str):
            prefixes = [domain_input.upper().replace("-", "")]
        else:
            prefixes = [d.upper().replace("-", "") for d in domain_input]

        grouped = defaultdict(dict)

        with catalog_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                url = raw_line.strip()
                if not url or url.startswith("#"): continue

                descriptor = Path(url).parts[-2]
                try:
                    dom_code, exp_code, temp_res, remainder = descriptor.split("-", 3)
                except ValueError: continue

                # 1. Match Prefix (e.g., EUR-11 starts with EUR)
                clean_dom = dom_code.upper().replace("-", "")
                if not any(clean_dom.startswith(p) for p in prefixes):
                    continue

                if temporal_filter and temp_res.lower() != temporal_filter:
                    continue

                exp_code = exp_code.lower()
                if exp_code not in experiments:
                    continue

                try:
                    gcm_rcm, member = remainder.rsplit("-", 1)
                    gcm, rcm = gcm_rcm.split("-", 1)
                except ValueError: continue

                # Extract numeric resolution for priority (EUR11 -> 11, AFR44 -> 44)
                res_match = re.search(r'(\d+)$', clean_dom)
                res_val = int(res_match.group(1)) if res_match else 999

                key = (clean_dom, gcm, rcm, member, temp_res.lower(), res_val)
                grouped[key][exp_code] = url

        model_list = []
        for (dom, gcm, rcm, mem, temp, res), entries in grouped.items():
            if all(exp in entries for exp in experiments):
                model_list.append({
                    "domain": dom,
                    "gcm": gcm,
                    "rcm": rcm,
                    "member": mem,
                    "temporal": temp,
                    "res_value": res,  # Store for sorting/filtering
                    "hist_url": entries["hist"],
                    "rcp85_url": entries["rcp85"],
                })

        # Priority 1: Member Order
        member_order = ["r1i1p1", "r2i1p1", "r3i1p1", "r6i1p1", "r12i1p1", "r0i0p0"]
        mem_prio = {m: i for i, m in enumerate(member_order)}

        # 2. Filter logic: Highest Resolution (Lowest res_value) + Member Priority
        # We group by (GCM, RCM) to find the best available version across domains
        best_models = {}
        for entry in model_list:
            # Use GCM and RCM as the identifying key
            key = (entry["gcm"], entry["rcm"])
            
            if key not in best_models:
                best_models[key] = entry
            else:
                existing = best_models[key]
                
                # Condition A: Current has HIGHER resolution (e.g. 11 < 44)
                if entry["res_value"] < existing["res_value"]:
                    best_models[key] = entry
                
                # Condition B: Same resolution, but better member priority
                elif entry["res_value"] == existing["res_value"]:
                    if mem_prio.get(entry["member"], 999) < mem_prio.get(existing["member"], 999):
                        best_models[key] = entry

        return sorted(list(best_models.values()), 
                    key=lambda item: (item["domain"], item["gcm"], item["rcm"]))


    @staticmethod
    def compute_climate_indices(data_input, parameter, study_region, 
                            baseline_range=("1990", "2020"), padding=15, month_range: tuple[int, int]=None):
        """
        Unified processor for CMIP6 and CORDEX data.
        data_input: Can be a dict {name: ds} (CMIP6) or a list of dicts (CORDEX).
        """
        results = {
            "seasonal_cycles": {},
            "spatial_maps": {},
            "time_series": {},
            "processed": [],
            "dropped": []
        }

        # Normalize input to a common format: a list of (label, dataset, entry_metadata)
        items_to_process = []
        if isinstance(data_input, dict):
            # Format for CMIP6
            for name, ds in data_input.items():
                items_to_process.append((name, ds, name))
        else:
            # Format for CORDEX
            for entry in data_input:
                label = f"{entry['gcm']}\n{entry['rcm']}"
                items_to_process.append((label, entry["experiment_data"], entry))

        for label, ds, original_entry in items_to_process:
            print(f"Processing: {label}")
            try:
                # Extract Variable & Unit Conversion
                da = ds[parameter]
                da = Utils.wrap_lon(da)

                # Handle Dimensions & Coordinate Names
                # Rename CMIP6-style coords to standard names
                if "lat" in da.coords: da = da.rename({"lat": "latitude"})
                if "lon" in da.coords: da = da.rename({"lon": "longitude"})
                
                # Detect spatial dimensions for averaging (Native Grid Handling)
                if "rlon" in da.dims:
                    xdim, ydim = ["rlon", "rlat"]
                elif "x" in da.dims:
                    xdim, ydim = ["x", "y"]
                else:
                    xdim, ydim = ["longitude", "latitude"]

                da = da.sortby("time")

                gr_daily_clim = da.sel(time=slice(*baseline_range))

                # Product A: Spatial Climatology
                da_clim = gr_daily_clim.mean("time")

                # Spatial Masking
                mask = regionmask.mask_geopandas(study_region, da.longitude, da.latitude)
                ts_regional = da.where(mask == 0, drop=True)

                # Product B: Regional Time Series (Latitude Weighted)
                ts_weighted = Process.weighted_values(ts_regional, value_col=None, lat_col='latitude')
                ts_final = ts_weighted.mean([xdim, ydim]).sortby("time")

                # Product C: Seasonal Cycle

                # Remove leap days and apply 31-day rolling mean to smooth the climatology
                clim31d = gr_daily_clim.where(~((gr_daily_clim.time.dt.month == 2) & (gr_daily_clim.time.dt.day == 29)), drop=True)
                days = np.arange(1, 366)
                dayofyear = clim31d.time.dt.dayofyear
                result_list = []

                for day in days:
                    # Build ±pad-day window (cyclically)
                    window_days = [(day + offset - 1) % 365 + 1 for offset in range(-padding, padding + 1)]
                    mask_days = dayofyear.isin(window_days)
                    window_data = clim31d.sel(time=mask_days)

                    stat = window_data.mean("time")

                    result_list.append(stat)
                
                result = xr.concat(result_list, dim='dayofyear')
                clim31d = result.assign_coords(dayofyear=days)

                sc = clim31d.where(mask == 0, drop=True)

                if month_range is None:
                    start_month, end_month = 1, 12   # full year
                else:
                    start_month, end_month = month_range
                # Create a dummy non-leap year to map month → doy
                dummy_time = xr.cftime_range(start="2001-01-01", periods=365, freq="D")
                sc = sc.assign_coords(dayofyear_time=("dayofyear", dummy_time))

                if end_month >= start_month:
                    month_mask = sc.dayofyear_time.dt.month.isin(range(start_month, end_month+1))
                else:
                    month_mask = sc.dayofyear_time.dt.month.isin(
                        list(range(start_month, 13)) + list(range(1, end_month + 1))
                    )

                sc = sc.where(month_mask, drop=True)

                # Drop temporary coord
                sc = sc.drop_vars("dayofyear_time")

                # Re-sort the time axis for cross-year scenarios
                if month_range is not None and end_month < start_month:
                    doy = sc.dayofyear
                    sort_key = xr.where(doy < 32 * start_month, doy + 365, doy)
                    sc = sc.sortby(sort_key)
                sc = Process.weighted_values(sc, value_col=None, lat_col='latitude')
                sc = sc.mean([xdim, ydim])

                # Store Results

                sc_out = sc.compute()
                daclim_out = da_clim.compute()
                tsfinal_out = ts_final.compute()

                # Convert only now (small data)
                if parameter in ["tas", "tasmin", "tasmax"]:
                    sc_out.values = conversions.Conversions.convert_temperature(sc_out.values, "k", "c")
                    daclim_out.values = conversions.Conversions.convert_temperature(daclim_out.values, "k", "c")
                    tsfinal_out.values = conversions.Conversions.convert_temperature(tsfinal_out.values, "k", "c")
                elif parameter == "pr":
                    sc_out.values = sc_out.values * 86400
                    daclim_out.values = daclim_out.values * 86400
                    tsfinal_out.values = tsfinal_out.values * 86400
                    

                results["seasonal_cycles"][label] = sc_out
                results["spatial_maps"][label] = daclim_out
                results["time_series"][label] = tsfinal_out
                results["processed"].append(original_entry)
                
                print(f"✅ {label} Processed successfully.")

            except Exception as exc:
                results["dropped"].append(original_entry)
                print(f"❌ {label} Failed: {exc}")
                continue

        return results
    
    @staticmethod
    def compute_gmst_anomalies(gmst_dict, event_year, year_range=(1951, 2100), window=4):
        """
        Computes yearly GMST rolling anomalies relative to a specific event year.
        Compatible with CMIP5 and CMIP6 'tas' datasets.
        
        Parameters:
        - gmst_dict: Dictionary {model_name: xarray_dataset}
        - event_year: The year to set as 0.0 anomaly (e.g., 2025)
        - year_range: Tuple of (start_year, end_year)
        - window: Size of the rolling mean window
        """
        results = {}
        
        for model_name, ds in gmst_dict.items():
            print(f"Calculating GMST anomalies for: {model_name}")
            try:
                # Extract variable and convert units
                da = ds['tas']
                da = da - 273.15  # Convert from Kelvin to Celsius 

                # Standardize coordinate names for CMIP5/6 compatibility
                if "lat" in da.coords: da = da.rename({"lat": "latitude"})
                if "lon" in da.coords: da = da.rename({"lon": "longitude"})

                # Spatial Average (Latitude Weighted)
                gmst_monthly = Process.weighted_values(da, value_col=None, lat_col='latitude')
                gmst_monthly = gmst_monthly.mean(["longitude", "latitude"]).sortby("time")

                # Temporal Aggregation (Annual)
                gmst_yearly = gmst_monthly.groupby("time.year").mean().compute()
                
                # Convert to DataFrame and Clean
                df_yearly = gmst_yearly.to_dataframe(name='gmst').reset_index()
                # Ensure we only have necessary columns (handling optional 'height' or 'level' dims)
                df_yearly = df_yearly[['year', 'gmst']]

                # Apply Rolling Window
                df_rolled = Process.calculate_rolling_window(
                    gdf=df_yearly, value_col='gmst', datetime_col="year", 
                    window=window, min_periods=2, centering=True, method="mean"
                )

                # Subset to Study Period
                df_subset = Utils.subset_gdf(
                    gdf=df_rolled, datetime_col="year", 
                    date_range=(year_range[0], year_range[1])
                ).copy()

                # Calculate Anomaly relative to Event Year
                try:
                    ref_val = df_subset.loc[df_subset["year"] == event_year, "gmst"].values[0]
                    df_subset["gmst"] = df_subset["gmst"] - ref_val
                    results[model_name] = df_subset
                    print(f"✅ GMST anomaly calculated (Ref {event_year}: {ref_val:.2f}°C)")
                except IndexError:
                    print(f"⚠️ Event year {event_year} not found in model {model_name} range.")
                    continue

            except Exception as e:
                print(f"❌ Failed to process GMST for {model_name}: {e}")
                
        return results
    
    @staticmethod
    def analyze_extreme_scenario():
        r_code = """
        analyze_extreme_scenario <- function(model_name, rp, model_df, gmst_df, 
                                            y_start, y_end, y_now, nsamp,dGMST_target, 
                                            scenario_label, save_dir) {
            
            cat(paste0("   Scenario [", scenario_label, "]: Years ", y_start, "-", y_end, "\n"))
            
            # 1. Subset and Merge
            m_sub <- model_df[model_df$year >= y_start & model_df$year <= y_end, ]
            g_sub <- gmst_df[gmst_df$year >= y_start & gmst_df$year <= y_end, ]
            df <- merge_model_gmst(m_sub, g_sub)
            
            if (nrow(df) < 20) return(NULL)

            # 2. Fit Model
            mdl <- tryCatch({
                # Try first with the default optimization method
                fit_ns(dist = "gev", type = "shift", data = df, 
                    varnm = "value", covnm = "gmst", lower = FALSE)
            }, error = function(e) {
                # If default fails, try again using Nelder-Mead
                message(paste("      ⚠️ Default fit failed for", model_name, "- trying Nelder-Mead..."))
                tryCatch({
                    fit_ns(dist = "gev", type = "shift", data = df, 
                        varnm = "value", covnm = "gmst", lower = FALSE, 
                        method = "Nelder-Mead")
                }, error = function(e2) {
                    return(NULL) # If both fail, return NULL
                })
            })

            if (is.null(mdl)) return(NULL)

            # 3. Define Covariates (CRITICAL FIX: drop=F keeps it as a DataFrame)
            # This prevents the "incorrect number of dimensions" error
            cov_now <- gmst_df[gmst_df$year == y_now, "gmst", drop = F]
            
            if (nrow(cov_now) == 0) {
                # Fallback: ensure it remains a 1-column dataframe
                val <- tail(df$gmst, 1)
                cov_now <- data.frame(gmst = val)
            }
            
            # Math on dataframes preserves the dataframe structure in R
            cov_hist <- cov_now - 1.3
            cov_fut  <- cov_now + dGMST_target

            # 4. Extract Results
            res <- tryCatch({
                cmodel_results(mdl, rp = rp, 
                            cov_f = cov_now, 
                            cov_hist = cov_hist, 
                            cov_fut = cov_fut, 
                            y_now = y_now, y_start = y_start, y_fut = y_end, nsamp = nsamp)
            }, error = function(e) {
                cat("      ❌ Extraction failed:", e$message, "\n")
                return(NULL)
            })

            if (!is.null(res)) {
                res_df <- as.data.frame((unlist(res)))
                
                # Add identifiers
                res_df$scenario <- scenario_label
                res_df$model <- model_name
                
                # 5. Plotting
                tryCatch({
                    fname <- file.path(save_dir, paste0(model_name, "_", scenario_label, ".png"))
                    val_to_plot <- unlist(res)[,"rp_value"]
                    
                    # Use 'cov_hist' (cov_cf) exactly like your original loop
                    png(fname, width = 480, height = 360)
                    plot_returnlevels(mdl, cov_f = cov_now, cov_cf = cov_hist, 
                                    ev = val_to_plot, nsamp = 100, main = paste(model_name, scenario_label))
                    dev.off()
                }, error = function(e) if (dev.cur() > 1) dev.off())
                
                return(res_df)
            }
            return(NULL)
        }
            """
        r(r_code)

    @staticmethod
    def merge_model_gmst():
        r_code = """
        merge_model_gmst <- function(model_df, gmst_df) {
            
            # model_df has: time (POSIXct), value (tasmax)
            # gmst_df has: year, gmst
            # merge on year
            out <- merge(
                model_df[, c("year", "value")],
                gmst_df,
                by = "year",
                all = FALSE
            )
            
            return(out)
        }
        """
        r(r_code)