from cdsapi import Client
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
import xarray as xr
import tempfile
import zipfile

from c3s_lib.data.variables import Variable

if __import__('sys').platform in ['linux']:
    import iris # type: ignore

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
    
    def _split_time_range_by_year_and_months(
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
    
    def _build_request_monthly_averaged(self, variable: str, time_range: tuple[datetime, datetime], bbox: tuple[float, float, float, float]) -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='MS')

        years = sorted({d.strftime('%Y') for d in dates})
        months = sorted({d.strftime('%m') for d in dates})

        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [variable],
            "year": years,
            "month": months,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],  # CDS expects [north, west, south, east]
        }
        return request
    
    def _fetch_data_monthly_netcdf(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> str:
        dataset = "reanalysis-era5-single-levels-monthly-means"
        
        req = self._build_request_monthly_averaged(variable, time_range, bbox)
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            self.cds_client.retrieve(
                dataset,
                req
            ).download(target=tmp.name)
            # open dataset
            return tmp.name
        
    def _fetch_data_monthly_averaged_xr(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> xr.Dataset:
        file = self._fetch_data_monthly_netcdf(variable, bbox, time_range)
        ds = xr.open_dataset(file)
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        return ds
    
    def _fetch_data_monthly_averaged_gpd(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        file = self._fetch_data_monthly_netcdf(variable, bbox, time_range)
        # open dataset
        ds = xr.open_dataset(file)
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        # filter time to exact range
        gdf = gdf[(gdf['valid_time'] >= time_range[0]) & (gdf['valid_time'] <= time_range[1])]

        return gdf
    
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
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> list[str]:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.
        Saves each year's netCDF to a temp file, converts entire xarray dataset
        straight to a pandas DataFrame, then to a GeoDataFrame.
        Returns a list of file paths to the saved netCDF files.
        """
        ranges = self._split_time_range_by_year(*time_range)
        files = []
        for start_dt, end_dt in ranges:
            req = self._build_request_pressure_levels([variable.cds_name()], bbox, (start_dt, end_dt), levels, daily_statistic=variable.cds_daily_statistic())
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)
                files.append(tmp.name)

        return files
    
    def fetch_data_pressure_levels_gpd(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> gpd.GeoDataFrame:
        files = self._fetch_data_pressure_levels(
            dataset,
            variable,
            bbox,
            time_range,
            levels,
        )
        
        gdfs = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            # create geometry
            df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            gdfs.append(gdf)
        
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
        
        # filter time to exact range
        min_start, max_end = time_range
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= min_start) & (combined_gdf['valid_time'] <= max_end)]

        return combined_gdf
    
    def fetch_data_pressure_levels_xr(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> xr.Dataset:
        files = self._fetch_data_pressure_levels(
            dataset,
            variable,
            bbox,
            time_range,
            levels,
        )
        
        dss = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            dss.append(ds)
        
        ds = xr.open_mfdataset(
            files,
            combine="by_coords"
        )
        
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        return ds
    
    def fetch_data_pressure_levels_iris(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> xr.Dataset:
        if __import__('sys').platform not in ['linux']:
            raise RuntimeError("Iris cubes is only supported on Linux platforms.")
        
        files = self._fetch_data_pressure_levels(
            dataset,
            variable,
            bbox,
            time_range,
            levels,
        )
        
        dss = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            dss.append(ds)
        
        ds = xr.open_mfdataset(
            files,
            combine="by_coords"
        )
        
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        # Write the netcdf to local path and open
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            ds.to_netcdf(tmp.name)
            iris_cube = iris.iris.load_cube(tmp.name) # type: ignore # we have checked platform above
            return iris_cube

    def _fetch_data_single_levels(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        months: list[int]|None = None
    ) -> list[str]:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.
        Saves each year's netCDF to a temp file, converts entire xarray dataset
        straight to a pandas DataFrame, then to a GeoDataFrame.
        Returns a list of GeoDataFrames, one per year.
        """
        ranges = self._split_time_range_by_year_and_months(time_range[0], time_range[1], months) if months is not None else self._split_time_range_by_year(*time_range)

        files = []
        for start_dt, end_dt in ranges:
            req = self._build_request_single_levels([variable.cds_name()], bbox, (start_dt, end_dt), daily_statistic=variable.cds_daily_statistic())
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)

                files.append(tmp.name)
                
        return files
    
    def fetch_data_single_levels_gpd(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> gpd.GeoDataFrame:
        files = self._fetch_data_single_levels(
            dataset,
            variable,
            bbox,
            time_range,
        )
        min_start, max_end = time_range
        gdfs = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            # create geometry
            df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            gdfs.append(gdf)
        
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
        
        # filter on given time range instead of taking the entire month
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= min_start) & (combined_gdf['valid_time'] <= max_end)]
        
        return combined_gdf
    
    def fetch_data_single_levels_xr(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> xr.Dataset:
        files = self._fetch_data_single_levels(
            dataset,
            variable,
            bbox,
            time_range,
        )
        
        dss = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            dss.append(ds)
        
        ds = xr.open_mfdataset(
            files,
            combine="by_coords"
        )
        
        # filter on given time range instead of taking the entire month
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        return ds
    
    def fetch_data_single_levels_iris(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> xr.Dataset:
        if __import__('sys').platform not in ['linux']:
            raise RuntimeError("Iris cubes is only supported on Linux platforms.")
        
        files = self._fetch_data_single_levels(
            dataset,
            variable,
            bbox,
            time_range,
        )
        
        dss = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            dss.append(ds)
        
        ds = xr.open_mfdataset(
            files,
            combine="by_coords"
        )
        
        # filter on given time range instead of taking the entire month
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        # Write the netcdf to local path and open
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            ds.to_netcdf(tmp.name)
            iris_cube = iris.iris.load_cube(tmp.name) # type: ignore # we have checked platform above
            return iris_cube      

    def _build_request_cmip6(self, variable: str, model:str, time_range: tuple[datetime, datetime], bbox: tuple[float, float, float, float], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        year = str(start.year)
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})

        request = {
            "temporal_resolution": temporal_resolution,
            "experiment": experiment,
            "variable": variable,
            "model": model,
            "year": [year],
            "month": months,
            "day": days,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        
        return request

    def _fetch_data_cmip6_netcdf(self, variable: str, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> str:
        dataset = "projections-cmip6"
        req = self._build_request_cmip6(variable, model, time_range, bbox, experiment, temporal_resolution)
        temp_dir = tempfile.TemporaryDirectory(delete=False)
        with temp_dir:
            zip_path = f"{temp_dir.name}/cmip6_data.zip"
            self.cds_client.retrieve(
                dataset,
                req
            ).download(target=zip_path)
            # unzip the file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir.name)
            # find the netcdf file in the extracted files
            for file_name in zip_ref.namelist():
                if file_name.endswith('.nc'):
                    return f"{temp_dir.name}/{file_name}"
            raise FileNotFoundError("No netCDF file found in the downloaded CMIP6 data.")
        
    def fetch_cmip6_xr(self, variable: str, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> xr.Dataset:
        file = self._fetch_data_cmip6_netcdf(variable, model, bbox, time_range, experiment, temporal_resolution)
        ds = xr.open_dataset(file)
        # filter time to exact range
        ds = ds.sel(time=slice(time_range[0], time_range[1]))
        return ds
    
    def fetch_cmip6_gpd(self, variable: str, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> gpd.GeoDataFrame:
        ds = self.fetch_cmip6_xr(variable, model, bbox, time_range, experiment, temporal_resolution)
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf