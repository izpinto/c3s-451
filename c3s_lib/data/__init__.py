from .beacon_client import *
from .cds_client import *
from .cordex_client import *
from .conversions import *
from .variables import *

if __import__('sys').platform in ['linux']:
    from .mars_client import *
    import iris # type: ignore

class DataClient():
    """
    DataClient is a class that provides methods to fetch and process climate data
    from various sources including the Climate Data Store (CDS), Beacon Cache, and MARS.
    
    Parameters:
        cds_key (str): The API key for the Climate Data Store (CDS).
        beacon_cache_url (str | None): Optional URL for the Beacon Cache service.
        beacon_token (str | None): Optional token for accessing the Beacon Cache.
        mars_key (str | None): Optional API key for accessing MARS data.
        
    
    """
    def __init__(self, cds_key: str, beacon_cache_url: str | None = None, beacon_token: str | None = None, mars_key: str | None = None) -> None:
        self.cds_client = CDSClient(cds_key)
        self.beacon_cache = None
        self.mars_client = None
        if beacon_cache_url:
            self.beacon_cache = BeaconDataClient(beacon_cache_url=beacon_cache_url, beacon_token=beacon_token)
        if mars_key:
            self.mars_client = MarsClient(key=mars_key)  # type: ignore for mars on non-linux platforms
    
    
    def get_beacon_cache_daily_gpd(
        self, 
        bbox: tuple[float,float,float,float], 
        time_ranges: list[tuple[datetime,datetime]],
        variable: Variable) -> gpd.GeoDataFrame | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        
        gdfs = []
        for range in time_ranges:
            for adj_bbox in adjusted_bboxes:
                gdf = self.beacon_cache.fetch_from_era5_daily_zarr_gpd(adj_bbox, range, variable.beacon_name())
                gdfs.append(gdf)
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    def get_beacon_cache_daily_xr(
        self,
        bbox: tuple[float,float,float,float],
        time_ranges: list[tuple[datetime,datetime]],
        variable: Variable) -> xr.Dataset | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        dss = []
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        for range in time_ranges:
            for adj_bbox in adjusted_bboxes:
                ds = self.beacon_cache.fetch_from_era5_daily_zarr_xr(adj_bbox, range, variable.beacon_name())
                dss.append(ds)

        return xr.concat(dss, dim='time') # type: ignore
    
    # def get_beacon_cache_daily_iris(
    #     self,
    #     bbox: tuple[float,float,float,float],
    #     time_ranges: list[tuple[datetime,datetime]],
    #     variable: str) -> iris.cube.Cube | None:
    #     if self.beacon_cache is None:
    #         print("Beacon Cache client not initialized")
    #         return None
        
    #     cubes = []
    #     adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
    #     for range in time_ranges:
    #         for adj_bbox in adjusted_bboxes:
    #             cube = self.beacon_cache.fetch_from_era5_daily_zarr_iris(adj_bbox, range, variable)
    #             cubes.append(cube)

    #     return iris.cube.CubeList(cubes).concatenate()
    
    def get_cds_daily_data_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> gpd.GeoDataFrame:
        
        gdfs = []
        for time_range in time_ranges:
            print(f"Fetching data from CDS for range: {time_range[0]} - {time_range[1]}")
            gdf = self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=time_range, variable=variable)
            gdfs.append(gdf)
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    def get_cds_daily_data_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> xr.Dataset:

        dss = []
        for time_range in time_ranges:
            print(f"Fetching data from CDS for range: {time_range[0]} - {time_range[1]}")
            ds = self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=time_range, variable=variable)
            dss.append(ds)
        return xr.concat(dss, dim='time') # type: ignore

    # def get_cds_daily_data_iris(
    #     self,
    #     bbox:tuple[float, float, float, float],
    #     time_ranges: list[tuple[datetime, datetime]],
    #     variable: str,
    #     statistic: str = "daily_mean",
    #     ) -> iris.cube.Cube:
    #     cubes = []
    #     for time_range in time_ranges:
    #         print(f"Fetching data from CDS for range: {time_range[0]} - {time_range[1]}")
    #         cube = self.cds_client.fetch_data_single_levels_iris("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=time_range, variable=variable, daily_statistic=statistic)
    #         cubes.append(cube)
    #     return iris.cube.CubeList(cubes).concatenate()    
    
    def get_daily_data_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> gpd.GeoDataFrame | None:
        
        all_gdfs = []
        for time_range in time_ranges:
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            gdfs = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                gdf = self.get_beacon_cache_daily_gpd(bbox, [time_range], variable)
                if gdf is not None and not gdf.empty:
                    gdfs.append(gdf)
                    min_retrieved_time = gdf['valid_time'].min()
                    max_retrieved_time = gdf['valid_time'].max()
            
            if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
                # No valid data found in beacon cache or we are missing data for the requested time range
                print("Missing data in beacon cache, fetching missing data from CDS...")

                if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                    # Request missing data from CDS
                    fetch_start = time_range[0]
                    fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                    gdf_cds = self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    gdf_cds = gdf_cds.rename(columns=variable.cds_variable_renames())
                    max_retrieved_time = gdf_cds['valid_time'].max()
                    gdfs.append(gdf_cds)
                
                if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                    # Request missing data from CDS
                    fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    gdf_cds = self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    gdf_cds = gdf_cds.rename(columns=variable.cds_variable_renames())
                    gdfs.append(gdf_cds)
        
            all_gdfs.append(gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)))
            
        # ToDo re-implement MARS data fetching if needed
            
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        
    def get_daily_data_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> xr.Dataset | None:
        
        all_dss = []
        for time_range in time_ranges:
            
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            dss = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                ds = self.get_beacon_cache_daily_xr(bbox, [time_range], variable)
                if ds is not None and 'valid_time' in ds:
                    dss.append(ds)
                    min_retrieved_time = ds['valid_time'].values.min()
                    max_retrieved_time = ds['valid_time'].values.max()
            
            if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > np.datetime64(time_range[0]) or max_retrieved_time < np.datetime64(time_range[1]):
                # No valid data found in beacon cache or we are missing data for the requested time range
                print("Missing data in beacon cache, fetching missing data from CDS...")

                if min_retrieved_time is None or min_retrieved_time > np.datetime64(time_range[0]):
                    # Request missing data from CDS
                    fetch_start = time_range[0]
                    fetch_end = pd.to_datetime(min_retrieved_time).to_pydatetime() if min_retrieved_time is not None else time_range[1]
                    ds_cds = self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    dss.append(ds_cds)
                
                if max_retrieved_time is None or max_retrieved_time < np.datetime64(time_range[1]):
                    # Request missing data from CDS
                    fetch_start = pd.to_datetime(max_retrieved_time).to_pydatetime() if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    ds_cds = self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    dss.append(ds_cds)
        
            all_dss.append(xr.concat(dss, dim='valid_time')) # type: ignore
        
        return xr.concat(all_dss, dim='valid_time') # type: ignore
    
    def temperature_2m_mean_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame | None:
        """
        Fetch mean 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily mean 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing mean 2m temperature data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.t2mean)
        if gdf is not None and not gdf.empty:
            gdf['t2m'] = Conversions.convert_temperature(gdf['t2m'], from_unit, to_unit)
        return gdf
    
    def temperature_2m_mean_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> xr.Dataset | None:
        """
        Fetch mean 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily mean 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            xr.Dataset: An xarray Dataset containing mean 2m temperature data with
            spatial and temporal information.
        """
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.t2mean)
        if ds is not None and 't2m' in ds:
            ds['t2m'].values = Conversions.convert_temperature(ds['t2m'].values, from_unit, to_unit)
        return ds
        
    def temperature_2m_min_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame | None:
        """
        Fetch minimum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily minimum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing minimum 2m temperature data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.t2min)
        if gdf is not None and not gdf.empty:
            gdf['t2m_min'] = Conversions.convert_temperature(gdf['t2m_min'], from_unit, to_unit)
        return gdf
    
    def temperature_2m_min_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> xr.Dataset | None:
        """
        Fetch minimum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily minimum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            xr.Dataset: An xarray Dataset containing minimum 2m temperature data with
            spatial and temporal information.
        """
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.t2min)
        if ds is not None and 't2m_min' in ds:
            ds['t2m_min'].values = Conversions.convert_temperature(ds['t2m_min'].values, from_unit, to_unit)
        return ds
    
    def temperature_2m_max_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame | None:
        """
        Fetch maximum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily maximum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing maximum 2m temperature data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.t2max)
        if gdf is not None and not gdf.empty:
            gdf['t2m_max'] = Conversions.convert_temperature(gdf['t2m_max'], from_unit, to_unit)
        return gdf
    
    def temperature_2m_max_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> xr.Dataset | None:
        """
        Fetch maximum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily maximum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            xr.Dataset: An xarray Dataset containing maximum 2m temperature data with
            spatial and temporal information.
        """ 
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.t2max)
        if ds is not None and 't2m_max' in ds:
            ds['t2m_max'].values = Conversions.convert_temperature(ds['t2m_max'].values, from_unit, to_unit)
        return ds
    
    def total_precipitation_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]]) -> gpd.GeoDataFrame | None:
        """
        Fetch total precipitation data for a specified bounding box and time ranges.

        This method retrieves daily total precipitation data from the ERA5
        single levels daily statistics dataset via the CDS client.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing total precipitation data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.tp)
        return gdf
    
    def total_precipitation_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]]) -> xr.Dataset | None:
        """
        Fetch total precipitation data for a specified bounding box and time ranges.

        This method retrieves daily total precipitation data from the ERA5
        single levels daily statistics dataset via the CDS client.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
        Returns:
            xr.Dataset: An xarray Dataset containing total precipitation data with
            spatial and temporal information.
        """
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.tp)
        return ds

    def mean_sea_level_pressure(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetch mean sea level pressure data for a specified bounding box and time range.

        This method retrieves daily mean sea level pressure statistics from the ERA5
        single levels daily statistics dataset via the CDS client.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Time range as (start_date, end_date) defining the period for which to fetch data.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing mean sea level pressure data with
            spatial and temporal information.
        """
        return self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", Variable.mslp, bbox, time_range)

    def z500_geopotential_mean(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetch daily mean geopotential data at 500 hPa pressure level from ERA5.

        This method retrieves geopotential height data at the 500 hPa pressure level,
        which is commonly used in meteorological analysis to identify atmospheric patterns
        and pressure systems.

        Parameters:
            bbox (tuple[float, float, float, float]): 
                Bounding box coordinates in the format (min_longitude, min_latitude, max_longitude, max_latitude) defining the geographical area of interest.
            time_range (tuple[datetime, datetime]): 
                Time range as a tuple of two datetime objects (start_date, end_date) defining the temporal extent of the data.

        Returns:
            gpd.GeoDataFrame: 
                A GeoDataFrame containing the daily mean geopotential data
                at 500 hPa pressure level for the specified spatial and temporal extent.
        """
        return self.cds_client.fetch_data_pressure_levels_gpd("derived-era5-pressure-levels-daily-statistics", Variable.geopotential, bbox, time_range, levels=[500])