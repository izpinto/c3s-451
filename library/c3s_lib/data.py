import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import datetime
from cdsapi import Client
import tempfile

class DataClient():
    def __init__(self, cds_key: str):
        self.cds_client = CDSClient(cds_key)

    def temperature_2m_min(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetches minimum temperature data for a given bounding box and time range.

        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.

        # Returns:
        - A pandas DataFrame containing the minimum temperature data.
        """
        # Implementation will go here
        return self.cds_client._fetch_data("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_minimum")

    def temperature_2m_max(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetches maximum temperature data for a given bounding box and time range.

        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.

        # Returns:
        - A pandas DataFrame containing the maximum temperature data.
        """
        # Implementation will go here
        return self.cds_client._fetch_data("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_maximum")

    def temperature_2m_mean(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetches mean temperature data for a given bounding box and time range.

        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.
        
        # Returns:
        - A pandas DataFrame containing the mean temperature data.
        """
        # Implementation will go here
        return self.cds_client._fetch_data("derived-era5-single-levels-daily-statistics", ['2m_temperature'], bbox, time_range, daily_statistic="daily_mean")
        
    
    
    def total_precipitation(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetches precipitation data for a given bounding box and time range.
        # Parameters:
        - bbox: A tuple of (min_longitude, min_latitude, max_longitude, max_latitude).
        - time_range: A tuple of (start_time, end_time) as datetime objects. Inclusive range.

        # Returns:
        - A pandas DataFrame containing the precipitation data.
        """
        # Implementation will go here
        return self.cds_client._fetch_data("derived-era5-single-levels-daily-statistics", ['total_precipitation'], bbox, time_range, daily_statistic="daily_sum")
    
    
    
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

    def _build_request(
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

    def _fetch_data(
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
            req = self._build_request(variables, bbox, (start_dt, end_dt), daily_statistic=daily_statistic)
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
        return combined_gdf