from ecmwfapi import ECMWFService
from datetime import datetime, timedelta
import shutil, subprocess
import geopandas as gpd
import xarray as xr
import iris # type: ignore

class MarsClient():
    def __init__(self, key: str):
        self.key = key
        self.server = ECMWFService("mars", key=key, url='https://api.ecmwf.int/v1')
        self.check_cdo()
    
    def check_cdo(self):
        if shutil.which("cdo") is None:
            raise EnvironmentError("❌ CDO not found. Please install it (e.g. `sudo apt install cdo`).")

        try:
            v = subprocess.run(["cdo", "-V"], capture_output=True, text=True, check=True)
            print(f"✅ {v.stdout.splitlines()[0]}")
        except Exception as e:
            raise EnvironmentError(f"⚠️ CDO check failed: {e}")
    
    def get_date_list(self, min_date: datetime, max_date: datetime) -> list[str]:
        # Get current system date
        given_date = datetime.now() - timedelta(days=1)  # Yesterday
        
        # Ensure given_date is within min_date and max_date
        if max_date > given_date:
            max_date = given_date

        # Generate list of days before the current date
        date_list = []
        while min_date <= max_date:
            date_list.append(min_date.strftime("%Y-%m-%d"))
            min_date += timedelta(days=1)

        return date_list    
    
    def get_temp_path(self) -> str:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
        return temp_file.name
    
    def fetch_t2m_mean_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m data for past 7 days from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "date": self.get_date_list(min_date, max_date),
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
        
        # Rename time to valid_time for clarity
        out_daily = out_daily.rename(columns={"time": "valid_time"})
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        return out_daily
    
    def fetch_t2m_min_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m min data for past 7 days from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
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
        out_daily = out_daily.rename(columns={"mn2t6": "t2m", "time": "valid_time"})
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        
        return out_daily
    
    def fetch_t2m_max_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m max data from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
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
        out_daily = out_daily.rename(columns={"mx2t6": "t2m", "time": "valid_time"})
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        
        return out_daily
    
    def fetch_total_precipitation_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching tp data for past 7 days from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
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
        
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        
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