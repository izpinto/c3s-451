import subprocess
import pandas as pd
import xarray as xr
import geopandas as gpd
import beacon_api
from datetime import datetime, timedelta
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
    
    def mean_sea_level_pressure(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
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

    def _fetch_from_era5_zarr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> gpd.GeoDataFrame:
        """
        Fetch data from ERA5 Zarr dataset in Beacon Cache.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['tmean', 'tmax', 'tmin', 'tp']

        Returns
        -------
        GeoDataFrame with data
        """
        era5_table = self.beacon_client.list_tables()['era5_zarr']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable)
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

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

class MarsClient():
    def __init__(self, key: str):
        from ecmwfapi import ECMWFService
        self.key = key
        self.server = ECMWFService("mars", key=key, url='https://api.ecmwf.int/v1')
        self.check_cdo()
    
    def check_cdo(self):
        import shutil, subprocess
        if shutil.which("cdo") is None:
            raise EnvironmentError("❌ CDO not found. Please install it (e.g. `sudo apt install cdo`).")

        try:
            v = subprocess.run(["cdo", "-V"], capture_output=True, text=True, check=True)
            print(f"✅ {v.stdout.splitlines()[0]}")
        except Exception as e:
            raise EnvironmentError(f"⚠️ CDO check failed: {e}")
    
    def get_date_list_past_7_days(self) -> list[str]:
        # Get current system date
        given_date = datetime.now()

        # Generate list of 7 days before the current date
        date_list = [(given_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2, 8)]

        # Sort ascending (oldest to newest)
        date_list.reverse()
        return date_list    
    
    def get_temp_path(self) -> str:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
        return temp_file.name
    
    def fetch_t2m_mean_operational_data(self) -> gpd.GeoDataFrame:
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m data for past 7 days from MARS: {self.get_date_list_past_7_days()}")
        request = {
            "class": "od",
            "date": self.get_date_list_past_7_days(),
            "expver": "1",
            "param": self.find_param_code("t2m"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymean", f"-shifttime,3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, t2m)
        df = ds[['longitude', 'latitude', 'time', 't2m']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        # Rename t2m to tmean for clarity
        out_daily = out_daily.rename(columns={"t2m": "tmean"})
        return out_daily
    
    def fetch_t2m_min_operational_data(self) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m data for past 7 days from MARS: {self.get_date_list_past_7_days()}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list_past_7_days(),
            "expver": "1",
            "param": self.find_param_code("tmin"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily min
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymin", f"-shifttime,-3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        print(ds)
        # Convert to dataframe but only keep (lon, lat, time, tmin)
        df = ds[['longitude', 'latitude', 'time', 'mn2t6']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        # Rename mn2t6 to tmin for clarity
        out_daily = out_daily.rename(columns={"mn2t6": "tmin"})
        return out_daily
    
    def fetch_t2m_max_operational_data(self) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m data for past 7 days from MARS: {self.get_date_list_past_7_days()}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list_past_7_days(),
            "expver": "1",
            "param": self.find_param_code("tmax"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily max
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymax", f"-shifttime,-3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, tmax)
        df = ds[['longitude', 'latitude', 'time', 'mx2t6']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        # Rename mx2t6 to tmax for consistency
        out_daily = out_daily.rename(columns={"mx2t6": "tmax"})
        return out_daily
    
    def fetch_total_precipitation_operational_data(self) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching tp data for past 7 days from MARS: {self.get_date_list_past_7_days()}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list_past_7_days(),
            "expver": "1",
            "param": self.find_param_code("tp"), # total precipitation
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily sum
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daysum", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, tp)
        df = ds[['longitude', 'latitude', 'time', 'tp']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return out_daily
    
    def fetch_t2m_mean_forecast_data(self) -> gpd.GeoDataFrame:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching t2m forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("t2m"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymean", f"-shifttime,3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, t2m)
        df = ds[['longitude', 'latitude', 'time', 't2m']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return out_daily
    
    def find_param_code(self, name: str) -> str | None:
        """
        Find the ECMWF MARS parameter code for a given variable name.
        Parameters: t2m, z500, mslp, tp, tmin, tmax        
        """
        param_codes = {
            "t2m": "167.128",
            "z500": "129.128",
            "mslp": "151.128",
            "tp": "228.128",
            "tmin": "122.128",
            "tmax": "121.128",
            # Add more mappings as needed
        }
        return param_codes.get(name)
    
class CordexClient():
    def __init__(self):
        pass