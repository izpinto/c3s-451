import pandas as pd
import xarray as xr
import geopandas as gpd
import beacon_api
from datetime import datetime
from cdsapi import Client
import tempfile
import glob
import numpy as np

class DataClient():
    def __init__(self, cds_key: str, beacon_cache_url: str | None = None, beacon_token: str | None = None):
        self.cds_client = CDSClient(cds_key)
        self.beacon_cache = None
        if beacon_cache_url:
            self.beacon_cache = BeaconDataClient(beacon_cache_url=beacon_cache_url, beacon_token=beacon_token)

    def _bbox_to_0_360(self, bbox: tuple[float,float,float,float], eps=1e-9) -> list[tuple[float,float,float,float]]:
        """
        Convert a bbox from [-180,180] lon to [0,360] lon.
        If it crosses 0° after conversion, return two bboxes that wrap around.
        
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat) with lon in [-180,180]
        eps  : small tolerance
        as_shapely : if True, return shapely boxes (requires shapely)
        
        Returns
        -------
        list of bboxes in (min_lon_360, min_lat, max_lon_360, max_lat) or shapely boxes
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Handle antimeridian-style input like (170, ..., -170, ...) meaning it crosses -180/180 already
        if min_lon > max_lon:
            max_lon += 360.0

        # Full-world guard
        if (max_lon - min_lon) >= 360.0 - eps:
            out = [(0.0, min_lat, 360.0, max_lat)]
        else:
            to360 = lambda x: x % 360.0
            lo = to360(min_lon)
            hi = to360(max_lon)

            if lo <= hi:  # no wrap in 0–360 space
                out = [(lo, min_lat, hi, max_lat)]
            else:
                # wraps across 0° in 0–360; split into two
                out = [
                    (lo,  min_lat, 360.0, max_lat),
                    (0.0, min_lat, hi,    max_lat),
                ]
        return out

    def _convert_temp(self, df: gpd.GeoDataFrame, from_unit="k", to_unit="c") -> gpd.GeoDataFrame:

        if from_unit not in ["k", "c", "f"]:
            raise ValueError(f"Invalid from_unit: {from_unit}. Must be 'k', 'c', or 'f'.")
            return df

        if from_unit == "k" and to_unit == "k":
            return df
        if from_unit == "c" and to_unit == "c":
            return df
        if from_unit == "f" and to_unit == "f":
            return df
        
        if from_unit == "k" and to_unit == "c":
            df['t2m'] = df['t2m'] - 273.15
        elif from_unit == "k" and to_unit == "f":
            df['t2m'] = (df['t2m'] - 273.15) * 9/5 + 32
        elif from_unit == "c" and to_unit == "k":
            df['t2m'] = df['t2m'] + 273.15
        elif from_unit == "c" and to_unit == "f":
            df['t2m'] = (df['t2m'] * 9/5) + 32
        elif from_unit == "f" and to_unit == "k":
            df['t2m'] = (df['t2m'] - 32) * 5/9 + 273.15
        elif from_unit == "f" and to_unit == "c":
            df['t2m'] = (df['t2m'] - 32) * 5/9

        return df
    
    def _convert_precipitation(self, df: gpd.GeoDataFrame, from_unit, to_unit) -> gpd.GeoDataFrame:
        if from_unit not in ["m", "m/h"]:
            raise ValueError(f"Invalid from_unit: {from_unit}. Must be 'm' or 'm/h'.")
            return df
        
        if from_unit == "m/h" and to_unit == "mm":
            df['tp'] = df['tp'] * 24000
            print("Converted from m/h to mm.")
        elif from_unit == "m" and to_unit == "mm":
            df['tp'] = df['tp'] * 1000
            print("Converted from m to mm.")
        
        return df
    
    def temperature_2m_min(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        """
        Fetches minimum temperature data for a given bounding box and time range.

        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.

        # Returns:
        - A pandas DataFrame containing the minimum temperature data.
        """
        # Implementation will go here
        df = self._convert_temp(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_minimum"), from_unit, to_unit)

        return df

    def temperature_2m_max(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        """
        Fetches maximum temperature data for a given bounding box and time range.

        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.

        # Returns:
        - A pandas DataFrame containing the maximum temperature data.
        """
        # Implementation will go here
        df = self._convert_temp(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_maximum"), from_unit, to_unit)

        return df
    
    def mean_sea_level_pressure(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        return self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['mean_sea_level_pressure'], bbox, time_range, daily_statistic="daily_mean")

    def z500_geopotential_mean(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        return self.cds_client._fetch_data_pressure_levels("derived-era5-pressure-levels-daily-statistics", ['geopotential'], bbox, time_range, levels=[500], daily_statistic="daily_mean")

    def temperature_2m_mean(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        """
        Fetches mean temperature data for a given bounding box and time range.

        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.
        
        # Returns:
        - A pandas DataFrame containing the mean temperature data.
        """
        
        # Fetch the data from the beacon client if has been defined. Then check the min and max date and compare. Whatever is missing should be requested via the cds api
        if self.beacon_cache:
            print("Fetching data from beacon cache...")
            beacon_bboxes = self._bbox_to_0_360(bbox)
            
            min_valid_time = None
            max_valid_time = None
            
            gdfs = []
            for beacon_bbox in beacon_bboxes:
                print("Beacon Bbox: "+  str(beacon_bbox))
                gdf = self.beacon_cache._fetch_temperature_data(bbox=beacon_bbox, time_range=time_range, columns=['t2m'])
                if not gdf.empty:
                    gdf = gdf.sort_values(['valid_time', 'longitude', 'latitude']).reset_index(drop=True)
                    # Get the min and max valid time from the DF and validate it covers the requested time range or else fill with era5 cds request
                    min_valid_time = gdf['valid_time'].min()
                    max_valid_time = gdf['valid_time'].max()
                    gdfs.append(gdf)
                    
            if min_valid_time == None or max_valid_time == None:
                print("No valid data found in beacon cache, fetching from CDS...")
                gdfs.append(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_mean"))
            else:
                print(f"Beacon cache covers time range: {min_valid_time} - {max_valid_time}")
                if min_valid_time > time_range[0]:
                    # Request missing data from CDS
                    print(f"Requesting missing data from CDS for range: {time_range[0]} - {min_valid_time}")
                    gdfs.append(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, (time_range[0], min_valid_time), daily_statistic="daily_mean"))

                if max_valid_time < time_range[1]:
                    # Request missing data from CDS
                    print(f"Requesting missing data from CDS for range: {max_valid_time} - {time_range[1]}")
                    gdfs.append(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, (max_valid_time, time_range[1]), daily_statistic="daily_mean"))

            # Concatenate all GeoDataFrames
            final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

            return self._convert_temp(final_gdf, from_unit, to_unit)
        
        # If no beacon cache, fetch directly from CDS
        print("Fetching data from CDS...")

        # Implementation will go here
        df = self._convert_temp(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_mean"), from_unit, to_unit)

        return df
    
    def total_precipitation(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str = "m", to_unit:str = "mm") -> gpd.GeoDataFrame:
        """
        Fetches precipitation data for a given bounding box and time range.
        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.

        # Returns:
        - A pandas DataFrame containing the precipitation data.
        - precipitation is in L/m^2/day
        """
        # Implementation will go here
        if self.beacon_cache:
            print("Fetching data from beacon cache...")
            beacon_bboxes = self._bbox_to_0_360(bbox)
            
            min_valid_time = None
            max_valid_time = None
            
            gdfs = []
            for beacon_bbox in beacon_bboxes:
                print("Beacon Bbox: "+  str(beacon_bbox))
                gdf = self.beacon_cache._fetch_total_precipitation_data(bbox=beacon_bbox, time_range=time_range)
                gdf = self._convert_precipitation(gdf, "m/h", "mm")
                if not gdf.empty:
                    gdf = gdf.sort_values(['valid_time', 'longitude', 'latitude']).reset_index(drop=True)
                    # Get the min and max valid time from the DF and validate it covers the requested time range or else fill with era5 cds request
                    min_valid_time = gdf['valid_time'].min()
                    max_valid_time = gdf['valid_time'].max()
                    gdfs.append(gdf)
                if min_valid_time == None or max_valid_time == None:
                    print("No valid data found in beacon cache, fetching from CDS...")
                    gdfs.append(self._convert_precipitation(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['total_precipitation'], bbox, time_range, daily_statistic="daily_sum"), "m", "mm"))
                else:
                    print(f"Beacon cache covers time range: {min_valid_time} - {max_valid_time}")
                    if min_valid_time > time_range[0]:
                        # Request missing data from CDS
                        print(f"Requesting missing data from CDS for range: {time_range[0]} - {min_valid_time}")
                        gdfs.append(self._convert_precipitation(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['total_precipitation'], bbox, (time_range[0], min_valid_time), daily_statistic="daily_sum"), "m", "mm"))
                    if max_valid_time < time_range[1]:
                        # Request missing data from CDS
                        print(f"Requesting missing data from CDS for range: {max_valid_time} - {time_range[1]}")
                        gdfs.append(self._convert_precipitation(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['total_precipitation'], bbox, (max_valid_time, time_range[1]), daily_statistic="daily_sum"), "m", "mm"))
        
            # Concatenate all GeoDataFrames
            final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
            return final_gdf
        
        print("Fetching data from CDS...")
        return self._convert_precipitation(self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['total_precipitation'], bbox, time_range, daily_statistic="daily_sum"), "m", "mm")
    
    def GET(self, parameter:str, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str|None = None, to_unit:str|None = None) -> gpd.GeoDataFrame:
        
        parameter = parameter.lower()

        # Build kwargs only if explicitly set
        unit_kwargs = {}
        if from_unit is not None:
            unit_kwargs["from_unit"] = from_unit
        if to_unit is not None:
            unit_kwargs["to_unit"] = to_unit

        match parameter:
            case 'tmean':
                return self.temperature_2m_mean(bbox, time_range, **unit_kwargs)
            case 'tmin':
                return self.temperature_2m_min(bbox, time_range, **unit_kwargs)
            case 'tmax':
                return self.temperature_2m_max(bbox, time_range, **unit_kwargs)
            case 'precipitation':
                return self.total_precipitation(bbox, time_range, **unit_kwargs)
            case 'z500': # 'z500_geopotential_mean':
                return self.z500_geopotential_mean(bbox, time_range, **unit_kwargs)
            case 'slp': # 'mean_sea_level_pressure':
                return self.mean_sea_level_pressure(bbox, time_range, **unit_kwargs)
            case _:
                return ValueError(f"Unsupported parameter: {parameter}")
    
class CDSClient():
    CDS_API_URL = "https://cds.climate.copernicus.eu/api"
    
    def __init__(self, cds_key: str):
        self.cds_client = Client(self.CDS_API_URL, key=cds_key)
        
    def _split_time_range_by_year(
        self,
        start: datetime,
        end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """
        Split a time range into sub-ranges, each within a single calendar year.
        """
        ranges = []
        current_start = start
        # iterate until within same year as end
        while current_start.year < end.year:
            year_end = datetime(year=current_start.year, month=12, day=31)
            ranges.append((current_start, year_end))
            current_start = datetime(year=current_start.year + 1, month=1, day=1)
        ranges.append((current_start, end))
        return ranges
    
    def _build_request_pressure_levels(self, variables: list[str], bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], levels: list[int], daily_statistic: str = "daily_mean") -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        year = str(start.year)
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})

        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": months,
            "day": days,
            "pressure_level": levels,
            "daily_statistic": daily_statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        return request

    def _build_request_single_levels(
        self,
        variables: list[str],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        daily_statistic: str = "daily_mean"
    ) -> dict:
        """
        Build the CDS API request dictionary for ERA5 daily statistics, limited to one year.

        variables: list of variable names for CDS (e.g. ['2m_temperature'])
        bbox: (min_lon, min_lat, max_lon, max_lat)
        time_range: (start_datetime, end_datetime) within a single year
        """
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        year = str(start.year)
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})

        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": months,
            "day": days,
            "daily_statistic": daily_statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        return request
    
    def _fetch_data_pressure_levels(
        self,
        dataset: str,
        variables: list[str],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
        daily_statistic: str = "daily_mean"
    ) -> gpd.GeoDataFrame:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.
        Saves each year's netCDF to a temp file, converts entire xarray dataset
        straight to a pandas DataFrame, then to a GeoDataFrame.
        Returns a list of GeoDataFrames, one per year.
        """
        yearly_ranges = self._split_time_range_by_year(*time_range)
        gdfs = []
        for start_dt, end_dt in yearly_ranges:
            req = self._build_request_pressure_levels(variables, bbox, (start_dt, end_dt), levels, daily_statistic=daily_statistic)
            with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)
                # open dataset
                ds = xr.open_dataset(tmp.name)
                # convert entire dataset to DataFrame
                df = ds.to_dataframe().reset_index()
                # create geometry
                df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
                gdfs.append(gdf)
                
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        # filter on given time range instead of taking the entire month
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= start_dt) & (combined_gdf['valid_time'] <= end_dt)]

        return combined_gdf

    def _fetch_data_single_levels(
        self,
        dataset: str,
        variables: list[str],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        daily_statistic: str = "daily_mean"
    ) -> gpd.GeoDataFrame:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.
        Saves each year's netCDF to a temp file, converts entire xarray dataset
        straight to a pandas DataFrame, then to a GeoDataFrame.
        Returns a list of GeoDataFrames, one per year.
        """
        yearly_ranges = self._split_time_range_by_year(*time_range)
        gdfs = []
        for start_dt, end_dt in yearly_ranges:
            req = self._build_request_single_levels(variables, bbox, (start_dt, end_dt), daily_statistic=daily_statistic)
            with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)
                # open dataset
                ds = xr.open_dataset(tmp.name)
                # convert entire dataset to DataFrame
                df = ds.to_dataframe().reset_index()
                # create geometry
                col_to_drop = 'number'
                if col_to_drop in df.columns:
                    df = df.drop(columns=[col_to_drop])
                df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
                gdfs.append(gdf)
                
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        # filter on given time range instead of taking the entire month
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= start_dt) & (combined_gdf['valid_time'] <= end_dt)]

        return combined_gdf
    
    
class BeaconDataClient():
    def __init__(self, beacon_cache_url: str, beacon_token: str | None = None) -> None:
        self.beacon_cache_url = beacon_cache_url
        self.beacon_token = beacon_token

        self.beacon_client = beacon_api.Client(
            url=self.beacon_cache_url,
            jwt_token=self.beacon_token
        )
        
        # Do a check for connecting to the Beacon Cache Layer
        self.beacon_client.check_status()

    def _fetch_temperature_data(self, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], columns: list[str]) -> gpd.GeoDataFrame:
        
        era5_table = self.beacon_client.list_tables()['era5_daily_mean_2m_temperature']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        if columns:
            for col in columns:
                query = query.add_select_column(col)

        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        df['longitude'] = (df['longitude'] + 180) % 360 - 180
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    
    def _fetch_total_precipitation_data(self, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        era5_table = self.beacon_client.list_tables()['era5_daily_total_precipitation']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column('tp')
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        df['longitude'] = (df['longitude'] + 180) % 360 - 180
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    
class CacheStatus():
    def __init__(self) -> None:
        self.time_range_covered = None

    def update_time_range(self, time_range: tuple[datetime, datetime]) -> None:
        self.time_range_covered = time_range

class LocalFileCache():
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.cache_status = CacheStatus()
        
        # Check which time ranges are covered by the cache. (Make sure they are successive in years or days.)
        files = sorted(glob.glob("./*.nc"))
        all_times = []
        for f in files:
            ds = xr.open_dataset(f)
            times = ds["valid_time"].values  # extract valid_time variable
            all_times.append(times)
            ds.close()
            
        # Flatten into one array
        all_times = np.concatenate(all_times)

        # Find min and max
        time_min = all_times.min()
        time_max = all_times.max()
        
    def get_cache_status(self) -> CacheStatus:
        return self.cache_status

    def _fetch_mean_2m_temperature(self, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        """
        Fetch mean 2m temperature data from local NetCDF files.
        """
        # Construct file path pattern
        files = sorted(glob.glob("./*.nc"))

        # Open all matching NetCDF files
        ds = xr.open_mfdataset(files, combine="by_coords")

        min_lon_index = bbox[0] * 4 # ERA5 using 0.25 degrees grid
        min_lat_index = bbox[1] * 4
        max_lon_index = bbox[2] * 4
        max_lat_index = bbox[3] * 4

        def slice_region(ds):
            return ds.isel(
                longitude=slice(min_lon_index, max_lon_index),
                latitude=slice(min_lat_index, max_lat_index),
            )

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            preprocess=slice_region,
            chunks={"valid_time": -1},      # let xarray pick time chunk size
            parallel=True,
        )
        
        df = ds.to_dataframe().reset_index()

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

        return gdf