from enum import Enum

class Variable:
    class CMIP5Monthly(Enum):
        temperature_2m = 1 # 2 meter temperature
        temperature_2m_max = 2 # 2 meter temperature max
        temperature_2m_min = 3 # 2 meter temperature min
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.CMIP5Monthly.temperature_2m: '2m_temperature',
                Variable.CMIP5Monthly.temperature_2m_max: '2m_temperature',
                Variable.CMIP5Monthly.temperature_2m_min: '2m_temperature',
            }
            return translation_dict[self]

    class CMIP6(Enum):
        near_surface_air_temperature = 1 # near surface air temperature
        precipitation = 2 # precipitation
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.CMIP6.near_surface_air_temperature: '2m_temperature',
                Variable.CMIP6.precipitation: 'total_precipitation',
            }
            return translation_dict[self]
        
    class ERA5DailySingleLevel(Enum):
        temperature_2m_mean = 1 # 2 meter temperature mean
        temperature_2m_max = 2 # 2 meter temperature max
        temperature_2m_min = 3 # 2 meter temperature min
        total_precipitation = 4 # total precipitation
        mean_sea_level_pressure = 5 # mean sea level pressure
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: '2m_temperature',
                Variable.ERA5DailySingleLevel.temperature_2m_max: '2m_temperature',
                Variable.ERA5DailySingleLevel.temperature_2m_min: '2m_temperature',
                Variable.ERA5DailySingleLevel.total_precipitation: 'total_precipitation',
                Variable.ERA5DailySingleLevel.mean_sea_level_pressure: 'mean_sea_level_pressure',
            }
            return translation_dict[self]
        
    class ERA5DailyPressureLevel(Enum):
        geopotential = 1 # geopotential
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.ERA5DailyPressureLevel.geopotential: 'geopotential',
            }
            return translation_dict[self]
    


class TempVariable(Enum):
    tp = 1 # total precipitation
    t2mean = 2 # mean daily temperature at 2 meters
    t2min = 3 # min daily temperature at 2 meters
    t2max = 4 # max daily temperature at 2 meters
    mslp = 5 # mean sea level pressure
    geopotential = 6 # geopotential
    
    def cds_name(self) -> str:
        """
        Get the corresponding CDS variable name.

        Returns:
            str: The CDS variable name.
        """
        translation_dict = {
            TempVariable.t2mean: '2m_temperature',
            TempVariable.tp: 'total_precipitation',
            TempVariable.t2min: '2m_temperature',
            TempVariable.t2max: '2m_temperature',
            TempVariable.mslp: 'mean_sea_level_pressure',
            TempVariable.geopotential: 'geopotential',
        }
        return translation_dict[self]
    
    def cds_daily_statistic(self) -> str:
        """
        Get the corresponding CDS daily statistic name, if applicable.

        Returns:
            str | None: The CDS daily statistic name, or None if not applicable.
        """
        translation_dict = {
            TempVariable.t2mean: 'daily_mean',
            TempVariable.tp: 'daily_sum',
            TempVariable.t2min: 'daily_minimum',
            TempVariable.t2max: 'daily_maximum',
            TempVariable.mslp: 'daily_mean',
            TempVariable.geopotential: 'daily_mean',
        }
        return translation_dict[self]
    
    def cds_variable_renames(self) -> dict[str, str]:
        """
        Get the renaming dictionary for CDS variable names to standard names.

        Returns:
            dict: A dictionary mapping CDS variable names to standard names.
        """
        translation_dict = {
            TempVariable.t2mean: {'t2m': 't2m'},
            TempVariable.tp: {'tp': 'tp'},
            TempVariable.t2min: {'t2m': 't2m_min'},
            TempVariable.t2max: {'t2m': 't2m_max'},
        }
        return translation_dict[self]
    
    def beacon_name(self) -> str:
        """
        Get the corresponding Beacon variable name.

        Returns:
            str: The Beacon variable name.
        """
        translation_dict = {
            TempVariable.t2mean: 't2m',
            TempVariable.tp: 'tp',
            TempVariable.t2min: 't2m_min',
            TempVariable.t2max: 't2m_max',
        }
        return translation_dict[self]