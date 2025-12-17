import beacon_api
import geopandas as gpd
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import tempfile

# Import iris only on supported platforms
if __import__('sys').platform in ['linux', 'darwin']:
    import iris # type: ignore

class BeaconDataClient():
    def __init__(self, beacon_cache_url: str, beacon_token: str | None = None) -> None:
        self.beacon_cache_url = beacon_cache_url
        self.beacon_token = beacon_token

        self.beacon_client = beacon_api.Client(
            url=self.beacon_cache_url,
            jwt_token=self.beacon_token
        )

    # copied for CDSClient so not the best practice but quick solution
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


    def _split_time_rangbe_by_year_and_months(
            self,
            start: datetime,
            end: datetime,
            months: list[str]|list[int]
        ) -> list[tuple[datetime, datetime]]:
        
        result = []

        def last_day_of_month(dt: datetime) -> datetime:
            next_month = dt.replace(day=28) + timedelta(days=4)  # always moves to the next month
            return next_month.replace(day=1) - timedelta(days=1)    # always returns back to the current month
        
        current = datetime(start.year, start.month, start.day)

        while current <= end:
            if current.month in months:
                month_start = current
                month_end = last_day_of_month(current)

                actual_start = max(month_start, start)
                actual_end = min(month_end, end)

                if actual_start <= actual_end:
                    result.append((actual_start, actual_end))
                
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return result

    
    def fetch_from_era5_daily_query(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> beacon_api.JSONQuery:
        """
        Create a query to fetch data from ERA5 Daily Zarr dataset in Beacon Cache.
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 't2m_max', 't2m_tmin', 'total_precipitation']
        Returns
        -------
        beacon_api.JSONQuery with data
        """

        era5_table = self.beacon_client.list_tables()['daily']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable)
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        return query

    def fetch_from_era5_daily_zarr_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> xr.Dataset:
        """
        Fetch data from ERA5 Daily Zarr dataset in Beacon Cache.
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 't2m_max', 't2m_tmin', 'total_precipitation']
        Returns
        -------
        xarray.Dataset with data
        """

        query = self.fetch_from_era5_daily_query(bbox, time_range, variable)

        ds = query.to_xarray_dataset(dimension_columns=['longitude','latitude','valid_time'], force=True)
        return ds    

    def fetch_from_era5_daily_zarr_iris(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str):
        """
        Fetch data from ERA5 Daily Zarr dataset in Beacon Cache.
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 't2m_max', 't2m_tmin', 'total_precipitation']
        Returns
        -------
        Iris cube with data
        """
        
        # Check platform
        if __import__('sys').platform not in ['linux', 'darwin']:
            raise RuntimeError("Iris cubes is only supported on Linux and macOS platforms.")

        query = self.fetch_from_era5_daily_query(bbox, time_range, variable)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        query.to_nd_netcdf(temp_path, ['longitude','latitude','valid_time'], force=True)

        iris_cube = iris.iris.load_cube(temp_path) # type: ignore # we have checked platform above
        return iris_cube

    def fetch_from_era5_daily_zarr_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> gpd.GeoDataFrame:
        """
        Fetch data from ERA5 Zarr dataset in Beacon Cache.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 't2m_max', 't2m_tmin', 'total_precipitation']

        Returns
        -------
        GeoDataFrame with data
        """
        query = self.fetch_from_era5_daily_query(bbox, time_range, variable)

        df = query.to_pandas_dataframe()
        
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time', variable], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
