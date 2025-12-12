# Functions for analogues

# import subprocess
import iris
import iris.coord_categorisation # type: ignore
# from iris.coord_categorisation import add_season_membership
import numpy as np
# import cartopy
import cartopy.crs as ccrs
# import glob
# import matplotlib.cm as mpl_cm
import os
# import sys
# import scipy.stats as sps
# from scipy.stats import genextreme as gev
# import random
# import scipy.io
# import xarray as xr
# import netCDF4 as nc
# import iris.coords # type: ignore
import iris.util # type: ignore
# from iris.util import equalise_attributes # type: ignore
# from iris.util import unify_time_units # type: ignore
# from scipy.stats.stats import pearsonr
import scipy.stats as stats
import calendar
# import random
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris.analysis
import cartopy.feature as cfeature
from datetime import datetime
import warnings
from shapely.geometry import Polygon


class Analogues:

    ERA5FILESUFFIX : str = "_daily" 

    def __init__(self) -> None:
        Analogues.ERA5FILESUFFIX : str = "_daily"    

    @staticmethod
    def reanalysis_file_location(self) -> str:
        '''
        Return the location of the ERA5 data
        '''
        #CEX path 
        #return os.path.join(os.environ["CLIMEXP_DATA"],"ERA5")
        #local wrkstation path 
        return '/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5-CX-READY/'

    @staticmethod
    def find_reanalysis_filename(var : str, daily : bool = True) -> str:
        '''
        Return the field filename for a given variable
        '''
        suffix : str = Analogues.ERA5FILESUFFIX
        path : str = os.path.join(Analogues.reanalysis_file_location(),"era5_{0}{1}.nc".format(var,suffix))
        return path

    @staticmethod
    def event_data_era(event_data, date: list, ana_var: str) -> list:
        '''
        Get ERA data for a defined list of variables on a given event date
        '''
        event_list = []
        # TODO: Convert variable list into a non-local (possibly with a class?)
        if event_data == 'extended':
            for variable in [ana_var, 'tp', 't2m', 't2m']:
                event_list.append(Analogues.reanalysis_data_single_date(variable, date))
        else:
            for variable in [ana_var, 'tp', 't2m', 'sfcWind']:
                event_list.append(Analogues.reanalysis_data_single_date(variable, date))
        return event_list

    @staticmethod
    def composite_dates_anomaly(P1_field, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
          P1_field = list of cubes, 1 per year - as used to calc D/date_list
          date_list = list of events to composite
        '''
        P1_field = P1_field - P1_field.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        n = len(date_list)
        FIELD = 0
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(P1_field, year, month, day)
            if NEXT_FIELD == None:
                print('Field failure for: ',+each)
                n = n-1
            else:
                if FIELD == 0:
                    FIELD = NEXT_FIELD
                else:
                    FIELD = FIELD + NEXT_FIELD
        return FIELD/n

    @staticmethod
    def ED_similarity(event, P_cube, region, method):
        '''
        Returns similarity values based on euclidean distance
        '''
        E = Analogues.extract_region(event, region)
        P = Analogues.extract_region(P_cube, region)
        D = []
        for yx_slice in P.slices(['grid_latitude', 'grid_longitude']):
            if method == 'ED':
                D.append(Analogues.euclidean_distance(yx_slice, E))
            elif method == 'CC':
                D.append(Analogues.correlation_coeffs(yx_slice, E))
        ED_max = np.max(np.max(D))
        S = [(1-x / ED_max) for x in D]
        return S

    @staticmethod
    def regrid(original, new):
        ''' Regrids onto a new grid '''
        mod_cs = original.coord_system(iris.coord_systems.CoordSystem)
        new.coord(axis='x').coord_system = mod_cs
        new.coord(axis='y').coord_system = mod_cs
        new_cube = original.regrid(new, iris.analysis.Linear())
        return new_cube

    @staticmethod
    def extract_region(cube_list, R1, lat:str='latitude', lon:str='longitude'):
        '''
        Extract Region (defaults to Europe)
        '''
        const_lat = iris.Constraint(latitude = lambda cell:R1[1] < cell < R1[0])
        if isinstance(cube_list, iris.cube.Cube):
            reg_cubes_lat = cube_list.extract(const_lat)
            reg_cubes = reg_cubes_lat.intersection(longitude=(R1[3], R1[2]))
        elif isinstance(cube_list, iris.cube.CubeList):
            reg_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                # print(each)
                subset = cube_list[each].extract(const_lat)
                reg_cubes.append(subset.intersection(longitude=(R1[3], R1[2])))
                
        return Analogues.guess_bounds(reg_cubes, lat=lat, lon=lon)

    @staticmethod
    def euclidean_distance(field, event):
        '''
        Returns list of D
        Inputs required:
          field = single cube of analogues field.
          event = cube of single day of event to match.
          BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
          AREA WEIGHTING APPLIED
        '''
        D= [] # to be list of all euclidean distances
        if event.coord('latitude').has_bounds():
            pass
        else:
            event.coord('latitude').guess_bounds()
        if event.coord('longitude').has_bounds():
            pass
        else:
            event.coord('longitude').guess_bounds()
        weights=iris.analysis.cartography.area_weights(event)
        event=event*weights
        a, b, c =np.shape(field) 
        field=field*np.array([weights]*a) 
        XA= event.data.reshape(b*c,1)
        XB = field.data.reshape(np.shape(field.data)[0],b*c,1)
        for Xb in XB:
            D.append(np.sqrt(np.sum(np.square(XA-Xb)))) 
        return D

    @staticmethod
    def reanalysis_data(var, Y1=1950, Y2=2023, months='[Jan]'):
        '''
        Loads in reanalysis daily data
        VAR can be psi250, msl, or tp (to add more)
        '''
        cubes = iris.load(Analogues.find_reanalysis_filename(var), var)
        try:
            cube = cubes[0]
        except:
            print("Error reading cubes for %s", var)
            raise FileNotFoundError
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell < Y2))
        iris.coord_categorisation.add_month(cube, 'time')
        cube = cube.extract(iris.Constraint(month=months))
        return cube

    @staticmethod
    def anomaly_period_outputs(Y1, Y2, ana_var, N, date, months, R1):
        '''
        Function to identify the N closest analogues of (N: number)
        date between (date format: [YYYY, 'Mon', DD], e.g. [2021, 'Jul', 14])
        Y1 and Y2, (year between 1950 and 2023)
        for 'event' 
        '''
        P1_msl = Analogues.reanalysis_data(ana_var, Y1, Y2, months) # Get ERA5 data, Y1 to Y2, for var and season chosen. Global.
        P1_field = Analogues.extract_region(P1_msl, R1) # Extract the analogues domain (R1) from global field
        ### difference for anomaly version
        P1_spatialmean = P1_field.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)   # Calculate spatial mean for each day
        P1_field = P1_field - P1_spatialmean # Remove spatial mean from each day
        event = Analogues.reanalysis_data_single_date(ana_var, date)
        E = Analogues.extract_region(event, R1) # Extract domain for event field
        E = E - E.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # remove spatial mean for event field
        ###
        P1_dates = Analogues.analogue_dates_v2(E, P1_field, R1, N*5)[:N] # calculate the closest analogues
        if str(date[0])+str("{:02d}".format(list(calendar.month_abbr).index(date[1])))+str(date[2]) in P1_dates: # Remove the date being searched for
            P1_dates.remove(str(date[0])+str("{:02d}".format(list(calendar.month_abbr).index(date[1])))+str(date[2]))
        return P1_dates

    @staticmethod
    def cube_date(cube):
        '''
        Returns date of cube (assumes cube single day)
        '''
        if len(cube.coords('year')) > 0:
           pass
        else:
           iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
           pass
        else:
           iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
           pass
        else:
           iris.coord_categorisation.add_day_of_month(cube, 'time')
        if len(cube.coords('day_of_year')) > 0:
           pass
        else:
           iris.coord_categorisation.add_day_of_year(cube, 'time')
        year = cube.coord('time').units.num2date(cube.coord('time').points)[0].year
        month = cube.coord('time').units.num2date(cube.coord('time').points)[0].month
        day = cube.coord('time').units.num2date(cube.coord('time').points)[0].day
        time = cube.coord('time').points[0]
        return year, month, day, time

    @staticmethod
    def date_list_checks(date_list, days_apart=5):
        '''
        Takes date_list and removes:
         1) the original event (if present)
         2) any days within 5 days of another event
        '''
        import datetime
        dates = []
        for each in date_list:
            dates.append(datetime.date(int(each[:4]), int(each[4:6]), int(each[6:])))
        new_dates = dates.copy()
        for i, each in enumerate(dates): # for each date in turn
            for other in dates[i+1:]: # go through all the rest of the dates
                if other in new_dates:
                    if abs((each-other).days) < days_apart:
                        new_dates.remove(other)
        new_dates_list = []
        for each in new_dates:
            new_dates_list.append(str(each.year)+str("{:02d}".format(each.month))+str("{:02d}".format(each.day)))
        return new_dates_list

    @staticmethod
    def eucdist_of_datelist(event, reanalysis_cubelist, date_list, region):
        ED_list = []
        E = Analogues.extract_region(event, region)
        if E.coord('latitude').has_bounds():
            pass
        else:
            E.coord('latitude').guess_bounds()
        if E.coord('longitude').has_bounds():
            pass
        else:
            E.coord('longitude').guess_bounds()
        weights = iris.analysis.cartography.area_weights(E)
        E = E*weights
        for i, each in enumerate(date_list):
            yr = int(date_list[i][:4])
            mon = calendar.month_abbr[int(date_list[i][4:-2])]
            day = int(date_list[i][-2:])
            field = Analogues.extract_region(Analogues.pull_out_day_era(reanalysis_cubelist, yr, mon, day), region)
            field = field*weights
            b, c = np.shape(field)
            XA = E.data.reshape(b*c,1)
            XB = field.data.reshape(b*c, 1)
            D = np.sqrt(np.sum(np.square(XA - XB)))
            ED_list.append(D)
        return ED_list

    @staticmethod
    def analogue_dates_v2(event, reanalysis_cube, region, N):
        '''
        '''
        def cube_date_to_string(cube_date : tuple) -> tuple:
            year,month,day,time = cube_date
            return str(year)+str(month).zfill(2)+str(day).zfill(2), time
        E = Analogues.extract_region(event, region)
        reanalysis_cube = Analogues.extract_region(reanalysis_cube, region)
        D = Analogues.euclidean_distance(reanalysis_cube, E)
        date_list = []
        time_list = []
        for i in np.arange(N):
            #print(i)
            I = np.sort(D)[i]
            for n, each in enumerate(D):
                if I == each:
                    a1 = n
            date, time = cube_date_to_string(Analogues.cube_date(reanalysis_cube[a1,...]))
            date_list.append(date)
            date_list2 = Analogues.date_list_checks(date_list, days_apart=5)
        return date_list2

    @staticmethod
    def pull_out_day_era(psi, sel_year, sel_month, sel_day):
        if type(psi)==iris.cube.Cube:
            psi_day = Analogues.extract_date(psi, sel_year, sel_month, sel_day)
        else:
            for each in psi:
                if len(each.coords('year')) > 0:
                    pass
                else:
                    iris.coord_categorisation.add_year(each, 'time')
                if each.coord('year').points[0]==sel_year:
                    psi_day = Analogues.extract_date(each, sel_year, sel_month, sel_day)
                else:
                    pass
        try:
            return psi_day
        except NameError:
            print('ERROR: Date not in data')
            return

    @staticmethod
    def extract_date(cube, yr, mon, day):
       '''
       Extract specific day from cube of a single year
       '''
       if len(cube.coords('year')) > 0:
           pass
       else:
           iris.coord_categorisation.add_year(cube, 'time')
       if len(cube.coords('month')) > 0:
           pass
       else:
           iris.coord_categorisation.add_month(cube, 'time')
       if len(cube.coords('day_of_month')) > 0:
           pass
       else:
           iris.coord_categorisation.add_day_of_month(cube, 'time')
       return cube.extract(iris.Constraint(year=yr, month=mon, day_of_month=day))

    @staticmethod
    def composite_dates(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
          psi = list of cubes, 1 per year - as used to calc D/date_list
          date_list = list of events to composite
        '''
        n = len(date_list)
        FIELD = 0
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(psi, year, month, day)
            if NEXT_FIELD == None:
                print('Field failure for: ',+each)
                n = n-1
            else:
                if FIELD == 0:
                    FIELD = NEXT_FIELD
                else:
                    FIELD = FIELD + NEXT_FIELD
        return FIELD/n

    @staticmethod
    def reanalysis_data_single_date(var : str, date : list):
        '''
        Loads in reanalysis daily data
        VAR can be: msl, or tp (to add more)
        '''
        filename = Analogues.find_reanalysis_filename(var)
        # print("Read file: {} for date {}".format(filename,date))
        cube = iris.load(filename, var)[0]
        cube = Analogues.extract_date(cube,date[0],date[1],date[2])
        return cube

    @staticmethod
    def quality_analogs(field, date_list, N, analogue_variable, region):
        '''
        For the 30 closest analogs of the event day, calculates analogue quality
        '''
        Q = []
        filename = Analogues. find_reanalysis_filename(analogue_variable)
        cube = iris.load(filename, analogue_variable)[0]
        for i, each in enumerate(date_list):
            year = int(each[:4])
            month = calendar.month_abbr[int(each[4:-2])]
            day = int(each[-2:])
            cube = Analogues.extract_date(cube,year,month,day)
            cube = Analogues.extract_region(cube,region)
            D = Analogues.euclidean_distance(field, cube) # calc euclidean distances
            Q.append(np.sum(np.sort(D)[:N]))
        return Q

    @staticmethod
    def diff_significance(field1, dates1, field2, dates2):
        '''
        Returns single composite of all dates
        Inputs required:
          field1 / 2 = cube of variable in period 1 /2
          dates1 / 2 = list of dates in period 1/2
        '''    
        n = len(dates1)
        field_list1 = iris.cube.CubeList([])
        for each in range(n):
            year = int(dates1[each][:4])
            month = calendar.month_abbr[int(dates1[each][4:-2])]
            day = int(dates1[each][-2:])
            field_list1.append(Analogues.pull_out_day_era(field1, year, month, day))
        n = len(dates2)
        field_list2 = iris.cube.CubeList([])
        for each in range(n):
            year = int(dates2[each][:4])
            month = calendar.month_abbr[int(dates2[each][4:-2])]
            day = int(dates2[each][-2:])
            field_list2.append(Analogues.pull_out_day_era(field2, year, month, day))    
        sig_field = field_list1[0].data
        a, b = np.shape(field_list1[0].data)
        for i in range(a):
            # print(i)
            for j in range(b):
                loc_list1 = []; loc_list2 = []
                for R in range(n):
                    loc_list1.append(field_list1[R].data[i,j])
                    loc_list2.append(field_list2[R].data[i,j])
                    # Trap and avoid precision loss warning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        u, p = stats.ttest_ind(loc_list1, loc_list2, equal_var=False, alternative='two-sided')
                if p < 0.05:
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list1[0]
        result_cube.data = sig_field
        return result_cube

    @staticmethod
    def composite_dates_ttest(field, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
          field = cube of field
          date_list = list of events to composite
        '''
        n = len(date_list)
        field_list = iris.cube.CubeList([])
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            field_list.append(Analogues.pull_out_day_era(field, year, month, day))
        sig_field = field_list[0].data
        a, b = np.shape(field_list[0].data)
        for i in range(a):
            # print(i)
            for j in range(b):
                loc_list = []
                for R in range(n):
                    loc_list.append(field_list[R].data[i,j])
                t_stat, p_val = stats.ttest_1samp(loc_list, 0)
                if p_val < 0.05:
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list[0]
        result_cube.data = sig_field
        return result_cube

    @staticmethod
    def impact_index(cube, II_domain):
        '''
        Calculates the impact index over the cube
        II_domain: spatial extent of index
        '''
        cube = Analogues.extract_region(cube, II_domain)
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        cube.data = ma.masked_invalid(cube.data)
        return cube.collapsed(('longitude','latitude'),iris.analysis.MEAN,weights=grid_areas).data

    @staticmethod
    def analogues_list(cube, date_list):
        '''
        Takes single cube of single variable that includes dates in date_list
        Pulls out single days of date
        Returns as list of cubes
        cube: single cube of a single variable
        date_list: list of dates in format 'YYYYMMDD'
        '''
        n = len(date_list)
        x = []
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(cube, year, month, day)
            if NEXT_FIELD == None:
                print('Field failure for: ',+each)
                n = n-1
            x.append(NEXT_FIELD)
        return x

    @staticmethod
    def plot_analogue_months(PAST, PRST):
        '''
        Produces histogram of number of analogues in each calendar month
        Inputs:
        PAST = list of dates of past period analogues, format ['19580308', ...
        PRST = list of dates of present period analogues, format ['19580308', ...
        '''
        # List of months (as number)
        PAST_MONTH = []
        for each in PAST:
            PAST_MONTH.append(int(each[4:6]))
        PRST_MONTH = []
        for each in PRST:
            PRST_MONTH.append(int(each[4:6]))
        plt.hist(PAST_MONTH, np.arange(.5, 13, 1), alpha=.5, label='Past (1950-1980)')
        plt.hist(PRST_MONTH, np.arange(.5, 13, 1), color='r', alpha=.5, label='Present (1994-2024)')
        plt.xticks(np.arange(1, 13, 1),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        plt.legend()
        plt.xlim([0.5, 12.5])
        #plt.yticks(np.arange(0, 25, 5))
        plt.ylabel('Frequency')
        plt.xlabel('Month')
        plt.title('Monthly distribution of top circulation analogues')

    @staticmethod
    def plot_box(axs, bdry):
        axs.plot([bdry[3], bdry[2]], [bdry[1], bdry[1]],'k')
        axs.plot([bdry[3], bdry[2]], [bdry[0], bdry[0]],'k')
        axs.plot([bdry[3], bdry[3]], [bdry[1], bdry[0]],'k')
        axs.plot([bdry[2], bdry[2]], [bdry[1], bdry[0]],'k')
        return

    @staticmethod
    def set_coord_system(cube, chosen_system = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS):
        '''
        This is used to prevent warnings that no coordinate system defined
        Defaults to DEFAULT_SPHERICAL_EARTH_RADIUS
        '''
        cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(chosen_system)
        cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(chosen_system)
        return cube



    ######################################################################################
    # MARIS added functions
    # most of these are repeated functions in the analogues
    ######################################################################################

    # Analogue rewrites
    ######################################################################################

    @staticmethod
    def analogue_months(event_date:list) -> list[str]:
        '''
        Return the months surrounding the event month
        
        Parameters:
            event_date: [year, month, day] e.g. [2024, 'Mar', 10]

        Returns:
            list[str]: Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr'].
        '''

        X = list(calendar.month_abbr)
        i = event_date if type(event_date) is int else X.index(event_date)
        if 1<i<12:
            months = [X[i-1], X[i], X[i+1]]
        elif i == 1:
            months = [X[12], X[i], X[i+1]]
        elif i == 12:
            months = [X[i-1], X[i], X[1]]

        return months

    @staticmethod
    def number_of_analogues(Y1:int, Y2:int, months:list[str]) -> int:
        '''
        Return the number of analogues in period Y1 to Y2 for the months given

        Parameters:
            Y1 (int): First year period
            Y2 (int): Last year period
            months (list[str]): Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr']
        
        Returns:
            int: Number of analouges to be used for calculations
        '''

        return int(((Y2-Y1)*len(months)*30)/100)

    @staticmethod
    def find_reanalysis_filename_v2(var:str, daily:bool = True) -> str:
        '''
        Return the field filename for a given variable

        Parameters:
            var (str): Variable
            daily (bool): ???

        Returns:
            str: Relative file location
        '''

        suffix : str = Analogues.ERA5FILESUFFIX
        path : str = os.path.join("", "era5_{0}{1}.nc".format(var,suffix))
        return path

    @staticmethod
    def reanalysis_data_v2(var:str, Y1:int=1950, Y2:int=2023, months:list[str]='[Jan]') -> iris.cube.Cube:
        '''
        Loads in reanalysis daily data

        Parameters:
            var (str): can be msl, or tp (to add more)
            Y1 (int): First year period
            Y2 (int): Last year period
            months (list[str]): Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr']

        Returns
            iris.cube.Cube: Loaded iris cube from NetCDF from Y1 to Y2 for selected months
        '''

        cubes = iris.load(Analogues.find_reanalysis_filename_v2(var), var)
        try:
            cube = cubes[0]
        except:
            print("Error reading cubes for %s", var)
            raise FileNotFoundError
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell < Y2))
        iris.coord_categorisation.add_month(cube, 'time')
        cube = cube.extract(iris.Constraint(month=months))
        return cube

    @staticmethod
    def reanalysis_data_single_date_v2(var:str, date:list) -> iris.cube.Cube:
        '''
        Loads in reanalysis daily data

        Parameters:
            var (str): Can be msl, or tp (to add more)
            date (list): Single date for extracting the cube
                [year, month, day] e.g. [2024, 'Mar', 10]

        Returns:
            iris.cube.Cube: Loaded iris cube from NetCDF for selected date
        '''

        filename = Analogues.find_reanalysis_filename_v2(var)
        # print("Read file: {} for date {}".format(filename,date))
        cube = iris.load(filename, var)[0]
        cube = Analogues.extract_date(cube,date[0],date[1],date[2])
        return cube

    @staticmethod
    def extract_region_shape(cube:iris.cube.Cube, shape:Polygon) -> iris.cube.Cube:
        '''
        Extract Region using a shape e.g. shapefile or polygon

        Parameters:
            cube (iris.cube.Cube): Single cube
            shape (Polygon): Polygon to mask cube with

        Returns
            iris.cube.Cube: Masked cube
        '''

        masked_cube = iris.util.mask_cube_from_shape(cube=cube, shape=shape)
        return masked_cube

    @staticmethod
    def event_data_era_v2(event_data:str, date:list, ana_var:str) -> list:
        '''
        Get ERA data for a defined list of variables on a given event date

        Parameters:
            event_data (str): Daily or extended
            date (list): Selected date
            ana_var (str): Variable to import

        Returns
            list: List of cubes for selected date
        '''

        event_list = []
        # TODO: Convert variable list into a non-local (possibly with a class?)
        if event_data == 'extended':
            for variable in [ana_var, 'tp', 't2m', 't2m']:
                event_list.append(Analogues.reanalysis_data_single_date_v2(variable, date))
        else:
            for variable in [ana_var, 'tp', 't2m', 'sfcWind']:
                event_list.append(Analogues.reanalysis_data_single_date_v2(variable, date))
        return event_list

    @staticmethod
    def extract_year(cube:iris.cube.Cube, Y1:int, Y2:int) -> iris.cube.Cube:
        '''
        Subsets the cube for given year range

        Parameters:
            cube (iris.cube.Cube): Cube to subset
            Y1 (int): First year period
            Y2 (int): Last year period
        
        Returns:
            iris.cube.Cube: Subsetted cube
        '''

        if not any(coord.name() == "year" for coord in cube.coords()):
            iris.coord_categorisation.add_year(cube, 'time')

        # is this actually checking if the Y1 and Y2 are in cube year range?
        rCube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell < Y2))
        return rCube

    @staticmethod
    def extract_date_v2(cube:iris.cube.Cube, date:list) -> iris.cube.Cube:
        '''
        Subset cube for specific date (single day)

        Paramters:
            cube (iris.cube.Cube): Cube to subset
            date (list): Date
        
        Returns:
            iris.cube.Cube: Subsetted cube
        '''

        yr = date[0]
        mon = date[1]
        day = date[2]

        if len(cube.coords('year')) > 0:
            pass
        else:
            iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_day_of_month(cube, 'time')
        return cube.extract(iris.Constraint(year=yr, month=mon, day_of_month=day))

    # J: missing description on what this function does
    @staticmethod
    def analogue_dates_v3(daily_cube:iris.cube.Cube, event_cube:iris.cube.Cube, N:int) -> list:
        '''
        To be added

        Parameters:
            daily_cube (iris.cube.Cube): Cube with daily values
            event_cube (iris.cube.Cube): Cube of just a single day
            N (int): Number of analogues
        
        Returns:
            list: To be added
        '''

        def cube_date_to_string(cube_date : tuple) -> tuple:
            year,month,day,time = cube_date
            return str(year)+str(month).zfill(2)+str(day).zfill(2), time

        D = Analogues.euclidean_distance(daily_cube, event_cube)
        date_list = []

        for i in np.arange(N):
            #print(i)
            I = np.sort(D)[i]
            for n, each in enumerate(D):
                if I == each:
                    a1 = n
            date, time = cube_date_to_string(Analogues.cube_date(daily_cube[a1,...]))
            date_list.append(date)
            date_list2 = Analogues.date_list_checks(date_list, days_apart=5)
        return date_list2

    # anaomaly period output for cubes

    # cube with daily values, cube of just the event, variable used, date of the event, region, number of analogues
    @staticmethod
    def anomaly_period_outputs_v2(daily_cube:iris.cube.Cube, event_cube:iris.cube.Cube, date:list, N:int) -> list:
        '''
        To be added

        Parameters:
            daily_cube (iris.cube.Cube): Cube with daily values
            event_cube (iris.cube.Cube): Cube of just a single day
            date (list): Date of the event [year, month abbrev, day]
            N (int): Number of analogues
        
        Returns
            list: To be added
        '''

        # cube_daily = cube_daily[:, 0, :, :]  # shape = (time, lat, lon)
        # multiple days
        daily_cube = daily_cube - daily_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # event for anavar to plot (fig a)

        # cube_event = cube_event[:, 0, :, :]  # shape = (time, lat, lon)
        # single day
        event_cube = event_cube - event_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # event for anavar to plot (fig a)

        P1_dates = Analogues.analogue_dates_v3(daily_cube, event_cube, N*5)[:N]

        if str(date[0])+str("{:02d}".format(list(calendar.month_abbr).index(date[1])))+str(date[2]) in P1_dates: # Remove the date being searched for
            P1_dates.remove(str(date[0])+str("{:02d}".format(list(calendar.month_abbr).index(date[1])))+str(date[2]))

        return P1_dates

    # J: add return type
    @staticmethod
    def analogues_composite_anomaly_v2(cube:iris.cube.Cube, dates:list):
        '''
        To be added

        Parameters:
            cube (iris.cube.Cube): Single cube with daily values
            dates (list): List of dates

        Returns:
            To be added
        '''

        P1_spatialmean = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)   # Calculate spatial mean for each day
        P1_field = cube - P1_spatialmean # Remove spatial mean from each day
        P1_comp = Analogues.composite_dates_anomaly(P1_field, dates) # composite analogues
        return P1_comp

    # J: add return type
    @staticmethod
    def analogues_composite_v2(cube:iris.cube.Cube, dates:list):
        '''
        To be added

        Parameters:
            cube (iris.cube.Cube): Single cube with daily values
            dates (list): List of dates

        Returns:
            To be added
        '''

        P1_comp = Analogues.composite_dates(cube, dates) # composite analogues
        return P1_comp

    # J: add return type
    # correlation value
    @staticmethod
    def var_correlation(var_cube:iris.cube.Cube, correlation_cube:iris.cube.Cube):
        '''
        Calculates the correlation between var_cube and correlation_cube

        Parameters:
            var_cube (iris.cube.Cube): Cube with daily values of variable t2m or tp
            correlation_cube (iris.cube.Cube): Cube with daily values of variable z500 or slp/msl

        Returns:
            To be added
        '''

        z_data = correlation_cube 
        z_data = z_data - z_data.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # event for anavar to plot (fig a)
        a, b, c = np.shape(z_data.data)
        corr_field = np.empty((b,c))
        p_field = np.empty((b,c))
        for i in np.arange(b):
            for j in np.arange(c):
                x, y = stats.pearsonr(var_cube.data, z_data.data[:,i,j])
                corr_field[i,j] = x
                p_field[i,j] = y

        # p_field does not seem to be used
        return z_data, corr_field, p_field

    @staticmethod
    def impact_index_v2(cube:iris.cube.Cube) -> iris.cube.Cube:
        '''
        Calculates the impact index over the cube

        Parameters:
            cube (iris.cube.Cube): Spatial extent of index

        Returns:
            To be added
        '''
        grid_areas = iris.analysis.cartography.area_weights(cube)
        cube.data = ma.masked_invalid(cube.data)
        return cube.collapsed(('longitude','latitude'),iris.analysis.MEAN,weights=grid_areas).data


    # Plotting
    ######################################################################################

    # adds background to plots
    @staticmethod
    def background(ax):
        '''
        Adds background to given plot (ax)
        '''

        ax.coastlines(linewidth=0.4)
        #ax.add_feature(cf.BORDERS, lw = 1, alpha = 0.7, ls = "--", zorder = 99)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.2, color='k',alpha=0.5,linestyle='--')
        gl.right_labels =gl.left_labels = gl.top_labels = gl.bottom_labels= False
        gl.xlabel_style = {'size': 5, 'color': 'gray'}
        gl.ylabel_style = {'size': 5, 'color': 'gray'}

    # Plot map of correlation
    @staticmethod
    def plot_correlation_map(z_data, z500_correlation, slp_correlation, region, z500_domain,
                             slp_domain, draw_labels:bool=True, fig_size:tuple[float, float]=(10,10)):

        '''
        Plots the correlation figures for z500 and slp
        '''

        fig, ax = plt.subplots(2, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=fig_size)
        lats = z_data.coord('latitude').points
        lons = z_data.coord('longitude').points
        con_lev = np.linspace(-1, 1, 20)

        c1 = ax[0].contourf(lons, lats, z500_correlation, levels=con_lev, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax[0].add_feature(cfeature.COASTLINE)
        ax[0].coastlines(linewidth=0.4)
        ax[0].set_title('Corr Z500')
        ax[0].gridlines(draw_labels=draw_labels)
        Analogues.plot_box(ax[0], region)
        Analogues.plot_box(ax[0], z500_domain)

        c1 = ax[1].contourf(lons, lats, slp_correlation, levels=con_lev, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax[1].add_feature(cfeature.COASTLINE)
        ax[1].coastlines(linewidth=0.4)
        ax[1].set_title('Corr MSL')
        ax[1].gridlines(draw_labels=draw_labels)
        Analogues.plot_box(ax[1], region)
        Analogues.plot_box(ax[1], slp_domain)

        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.3, 0.01, 0.4])
        cbar = fig.colorbar(c1, cax=cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'])
        plt.tight_layout()

        return fig, ax

    # Violin Plot (to visually check the result)
    @staticmethod
    def violin_plot(Haz:str, II_event, II_z500, II_slp, fig_size:tuple[float, float]=(2.5, 2.5)):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        plots = axs.violinplot([II_z500, II_slp], showmeans=True, showextrema=False, widths = .8)
        plots["bodies"][1].set_facecolor('green')
        axs.axhline(II_event, color='r', label = 'Event')
        axs.set_xticks([1,2], labels=['Z500', 'SLP'])
        axs.tick_params(axis='x', length=0)
        if Haz == 't2m': axs.set_ylabel('Temperature (K)')
        if Haz == 'tp': axs.set_ylabel('Daily Rainfall (mm)')
        t, p = stats.ttest_ind(II_z500, II_slp, equal_var=False, alternative='two-sided')
        if p < 0.05:
            axs.set_title(('%.2f'%t), pad=-20, loc='left', fontweight="bold")
        else:
            axs.set_title(('%.2f'%t), pad=-20, loc='left')

        return fig, axs

    # Shown for larger range of analogue proportions
    @staticmethod
    def plot_analogue_proportions(II_event, II_z500, II_slp, N,
                                  fig_size:tuple[float,float]=(2.5, 2.5), xlim:float=10):

        meanT = []
        for i in np.arange(len(II_z500)):
            meanT.append(np.mean(II_z500[:i]))

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        axs.plot(meanT, 'b', label = 'Z500')

        meanT = []
        for i in np.arange(len(II_slp)):
            meanT.append(np.mean(II_slp[:i]))

        axs.plot(meanT, 'g', label = 'SLP')
        axs.set_xlim([xlim, N])
        axs.axhline(II_event, color='r', label = 'Event')
        axs.legend() 
        axs.set_ylabel('Hazard')
        axs.set_xlabel('# of analogues')

        return fig, axs

    # Plot: Analogue variable
    @staticmethod
    def plot_analogue_variable(ana_var:str, event_cube, selected_daily_cube,
                               dates_past, dates_prst, event_date):

        # EVENT FIELDS
        event_cube = event_cube - event_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

        # ANALOGUE COMPOSITES
        PAST_comp = Analogues.analogues_composite_anomaly_v2(selected_daily_cube, dates_past)
        PRST_comp = Analogues.analogues_composite_anomaly_v2(selected_daily_cube, dates_prst)

        if ana_var == 'z500':
            PAST_comp = PAST_comp/10
            PRST_comp = PRST_comp/10
            event_cube = event_cube/10

        fig = plt.figure(figsize=(12,3),layout='constrained',dpi=200)

        lats=PRST_comp.coord('latitude').points
        lons=PRST_comp.coord('longitude').points

        x= np.round(np.arange(0, np.max([np.abs(PAST_comp.data), np.abs(PRST_comp.data), np.abs(event_cube.data)]), 2))
        con_lev=np.append(-x[::-1][:-1],x)

        ax= plt.subplot(1,3,1,projection=ccrs.PlateCarree())
        c1 = ax.contourf(lons, lats, event_cube.data, levels=con_lev, cmap="RdBu_r", transform=ccrs.PlateCarree(), extend='both')
        cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
        cbar.ax.tick_params()
        ax.set_title('a) Event, '+str(event_date[2])+event_date[1]+str(event_date[0]), loc='left')
        Analogues.background(ax)

        ax= plt.subplot(1,3,2,projection=ccrs.PlateCarree())
        c1 = ax.contourf(lons, lats, PAST_comp.data, levels=con_lev, cmap="RdBu_r", transform=ccrs.PlateCarree(), extend='both')
        cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
        cbar.ax.tick_params()
        ax.set_title('b) Past Analogues', loc='left')
        Analogues.background(ax)

        ax= plt.subplot(1,3,3,projection=ccrs.PlateCarree())
        c1 = ax.contourf(lons, lats, PRST_comp.data, levels=con_lev, cmap="RdBu_r", transform=ccrs.PlateCarree(), extend='both')
        cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
        cbar.ax.tick_params()
        ax.set_title('c) Present Analogues', loc='left')
        Analogues.background(ax)

        fig.suptitle('Analogue Variable: '+ ana_var)

        return fig, ax

    # J: again redundant data imports and region extractions?
    @staticmethod
    def plot_z500_slp_t2m_tp(ana_var, var_list:list[str], cube_map:dict[str, any], 
                             R2:list[float], region:list[float],
                             sig_field, event_date, dates_past, dates_prst,
                             fig_size:tuple[float, float]=(12,12), dpi:int=200):

        # Plot: Z500, SLP, t2m, tp (ToDo: +winds when ERA5 not extended)
        # Z500 plotted as anomalies (to remove the influence of the longterm trend)
        fig = plt.figure(figsize=fig_size, layout='constrained', dpi=dpi)

        for i, var in enumerate(var_list):
            if var == 'z500' or var == 'msl': CMAP = ["RdBu_r", "RdBu_r"]
            if var == 'tp': CMAP = ["YlGnBu", "BrBG"]
            if var == 't2m': CMAP = ["YlOrRd", "RdYlBu_r"]

            # EVENT FIELDS
            # here we can just extract the date from the cubes we imported earlier
            # event_cube = my.extract_region(my.reanalysis_data_single_date_v2(var, event_date), R2)
            # replace with line below
            # can we just use the cube_map_event_reg?
            event_cube =  Analogues.extract_date_v2(Analogues.extract_region(cube_map[var], R2), event_date)

            if var == ana_var:
                event_cube = event_cube - event_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

                    # ANALOGUE COMPOSITES
            if var == ana_var:
                PRST_comp = Analogues.analogues_composite_anomaly_v2(Analogues.extract_region(cube_map[var], R2), dates_prst)
                PAST_comp = Analogues.analogues_composite_anomaly_v2(Analogues.extract_region(cube_map[var], R2), dates_past)
            else:
                PRST_comp = Analogues.analogues_composite_v2(Analogues.extract_region(cube_map[var], R2), dates_prst)
                PAST_comp = Analogues.analogues_composite_v2(Analogues.extract_region(cube_map[var], R2), dates_past)


            # Unit conversions
            # ============== Check this because the units should already be correct =============
            if var == 'z500':
                PAST_comp = PAST_comp * 0.0980665 # adjust for gravity?
                PRST_comp = PRST_comp * 0.0980665
                event_cube = event_cube * 0.0980665
            if var == 'msl':
                PAST_comp = PAST_comp * .01   # adjust for?
                PRST_comp = PRST_comp * .01
                event_cube = event_cube * .01    
            if var == 't2m':
                PAST_comp = PAST_comp - 273.15    # adjust kelvin to celsius
                PRST_comp = PRST_comp - 273.15
                event_cube = event_cube - 273.15   
            #======================================================================================

            lats=PRST_comp.coord('latitude').points
            lons=PRST_comp.coord('longitude').points 
            if var == 'z500' or var == 'msl':  
                con_lev = np.round(np.arange(np.min([PAST_comp.data, PRST_comp.data, event_cube.data]), np.max([PAST_comp.data, PRST_comp.data, event_cube.data]), 2))
                #con_lev = np.round(np.arange(-abs(max(([np.min([PAST_comp.data, PRST_comp.data, E.data]), np.max([PAST_comp.data, PRST_comp.data, E.data])]), key=abs)), abs(max(([np.min([PAST_comp.data, PRST_comp.data, E.data]), np.max([PAST_comp.data, PRST_comp.data, E.data])]), key=abs)), 2))
            if var == 't2m':
                con_lev = np.round(np.arange(0, np.max([PAST_comp.data, PRST_comp.data, event_cube.data]), 2))
            if var == 'tp':
                con_lev = np.arange(0, np.max([PAST_comp.data,PRST_comp.data, event_cube.data])/2, .2)
            # Plotting event
            ax= plt.subplot(len(var_list),4,(i*4)+1,projection=ccrs.PlateCarree())
            c1 = ax.contourf(lons, lats, event_cube.data, levels=con_lev, cmap=CMAP[0], transform=ccrs.PlateCarree(), extend='both')
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            ax.set_ylabel(var)
            Analogues.background(ax)
            # Plotting Past Composite
            ax= plt.subplot(len(var_list),4,(i*4)+2,projection=ccrs.PlateCarree())
            c1 = ax.contourf(lons, lats, PAST_comp.data, levels=con_lev, cmap=CMAP[0], transform=ccrs.PlateCarree(), extend='both')
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            Analogues.background(ax)
            # Plotting Present Composite
            ax= plt.subplot(len(var_list),4,(i*4)+3,projection=ccrs.PlateCarree())
            c1 = ax.contourf(lons, lats, PRST_comp.data, levels=con_lev, cmap=CMAP[0], transform=ccrs.PlateCarree(), extend='both')
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            Analogues.background(ax)
            # Plotting Change
            ax= plt.subplot(len(var_list),4,(i*4)+4,projection=ccrs.PlateCarree())
            Dmax = np.round(np.nanmax(np.abs([np.nanmin((PRST_comp-PAST_comp).data), np.nanmax((PRST_comp-PAST_comp).data)])))
            diff_lev = np.linspace(-Dmax, Dmax, 41)
            c1 = ax.contourf(lons, lats, (PRST_comp-PAST_comp).data, levels=diff_lev, cmap=CMAP[1], transform=ccrs.PlateCarree(), extend='both')
            c2 = ax.contourf(lons, lats, sig_field[i].data, levels=[-2, 0, 2], hatches=['////', None], colors='none', transform=ccrs.PlateCarree())
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            Analogues.background(ax)
            #fig.suptitle('Analogue Variable: '+ana_var)

        ax= plt.subplot(4,4,1,projection=ccrs.PlateCarree())    
        ax.set_title('a) '+str(event_date[2])+event_date[1]+str(event_date[0])+'  '+var_list[0], loc='left')
        Analogues.plot_box(ax, region)
        ax= plt.subplot(4,4,2,projection=ccrs.PlateCarree())  
        ax.set_title('b) Past Analogues', loc='left')
        Analogues.plot_box(ax, region)
        ax= plt.subplot(4,4,3,projection=ccrs.PlateCarree())  
        ax.set_title('c) Present Analogues', loc='left')
        Analogues.plot_box(ax, region)
        ax= plt.subplot(4,4,4,projection=ccrs.PlateCarree())  
        ax.set_title('d) Change (Present - Past)', loc='left')
        Analogues.plot_box(ax, region)


        plt.subplot(4,4,5,projection=ccrs.PlateCarree()) .set_title('e) '+var_list[1], loc='left')
        plt.subplot(4,4,6,projection=ccrs.PlateCarree()) .set_title('f) ', loc='left')
        plt.subplot(4,4,7,projection=ccrs.PlateCarree()) .set_title('g) ', loc='left')
        plt.subplot(4,4,8,projection=ccrs.PlateCarree()) .set_title('h) ', loc='left')

        plt.subplot(4,4,9,projection=ccrs.PlateCarree()) .set_title('i) '+var_list[2], loc='left')
        plt.subplot(4,4,10,projection=ccrs.PlateCarree()) .set_title('j) ', loc='left')
        plt.subplot(4,4,11,projection=ccrs.PlateCarree()) .set_title('k) ', loc='left')
        plt.subplot(4,4,12,projection=ccrs.PlateCarree()) .set_title('l) ', loc='left')

        plt.subplot(4,4,13,projection=ccrs.PlateCarree()) .set_title('m) '+var_list[3], loc='left')
        plt.subplot(4,4,14,projection=ccrs.PlateCarree()) .set_title('n) ', loc='left')
        plt.subplot(4,4,15,projection=ccrs.PlateCarree()) .set_title('o) ', loc='left')
        plt.subplot(4,4,16,projection=ccrs.PlateCarree()) .set_title('p) ', loc='left')

        return fig, ax

    @staticmethod
    def plot_frequency_timeseries(yr_vals:list, roll_vals:list, Y1:int, Y2:int,
                                  fig_size:tuple[float,float]=(8,4)):
        # Plot timeseries with linear trends and 10-yr rolling means
        fig, ax = plt.subplots(1, 1, figsize = fig_size)

        # linear trends
        trend = np.polyfit(np.arange(Y1, Y2), yr_vals[1] ,1)
        trendpoly = np.poly1d(trend) 
        slope10, intercept, r_value, pval_10, std_err = stats.linregress(np.arange(Y1, Y2), yr_vals[1])
        ax.plot(np.arange(Y1, Y2), trendpoly(np.arange(Y1, Y2)), color='orange', linewidth=.5)
        text10 = 'Upper 10%. trend: '+"%.2f" % round(slope10, 2)+', pval: '+"%.2f" % round(pval_10, 2)

        trend = np.polyfit(np.arange(Y1, Y2), yr_vals[0],1)
        trendpoly = np.poly1d(trend) 
        slope5, intercept, r_value, pval_5, std_err = stats.linregress(np.arange(Y1, Y2), yr_vals[0])
        ax.plot(np.arange(Y1, Y2), trendpoly(np.arange(Y1, Y2)), color='r', linewidth=.5)
        text5 = 'Upper 5%. trend: '+"%.2f" % round(slope5, 2)+', pval: '+"%.2f" % round(pval_5, 2)

        trend = np.polyfit(np.arange(Y1, Y2), yr_vals[2],1)
        trendpoly = np.poly1d(trend) 
        slope20, intercept, r_value, pval_20, std_err = stats.linregress(np.arange(Y1, Y2), yr_vals[2])
        ax.plot(np.arange(Y1, Y2), trendpoly(np.arange(Y1, Y2)), color='b', linewidth=.5)
        text20 = 'Upper 20%. trend: '+"%.2f" % round(slope20, 2)+', pval: '+"%.2f" % round(pval_20, 2)

        ax.plot(np.arange(Y1+5, Y2-5), roll_vals[0], 'r:')
        ax.plot(np.arange(Y1+5, Y2-5), roll_vals[1], color='orange', linestyle=':')
        ax.plot(np.arange(Y1+5, Y2-5), roll_vals[2], 'b:')

        ax.plot(np.arange(Y1, Y2), yr_vals[0], 'r', label=text5)
        ax.plot(np.arange(Y1, Y2), yr_vals[1], color='orange', label=text10)
        ax.plot(np.arange(Y1, Y2), yr_vals[2], 'b', label=text20)

        ax.set_xlabel('Year')
        ax.set_ylabel("Similar days per year")

        ax.set_xlim([Y1, Y2])
        ax.legend(loc=2)

        return fig, ax, [slope5, slope10, slope20], [pval_5, pval_10, pval_20]

    # J: look at data imports and region extractions
    # J: Look at what variables are repeated
    # J: Why are the options set here again?


        # ## OPTIONS ##
        # # How many analogues?
        # n = 19 # no more than 29
        # # With circulation?
        # circ_plot = 0 # 1: plots circulation contours, 0: does not

    @staticmethod
    def plot_postage_stamps(ana_var:str, haz_var:str, cube_map, event_date,
                            region, cmap, dates_past, circ_past, haz_past,
                            circ_plot, n, fig_size:tuple[float, float]=(12,12)):

        # Past dates plot
        lats=circ_past[0].coord('latitude').points
        lons=circ_past[0].coord('longitude').points
        fig, axs = plt.subplots(nrows=(int(np.ceil((n+1)/5))),
                                ncols=5,
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                figsize=fig_size)

        circ = Analogues.extract_region(Analogues.extract_date_v2(cube_map[ana_var], event_date), region)
        haz = Analogues.extract_region(Analogues.extract_date_v2(cube_map[haz_var], event_date), region)

        if haz_var == 'tp':
            c = axs[0,0].contourf(lons, lats, haz.data,
                                  levels=np.linspace(1, 80, 9),
                                  cmap = cmap,
                                  transform=ccrs.PlateCarree(),
                                  extend='max')

            fig.subplots_adjust(right=0.8, hspace=-.2)
            cbar_ax = fig.add_axes([0.81, 0.4, 0.01, 0.2])
            fig.colorbar(c, cax=cbar_ax, ticks=np.arange(0, 100, 10))
            cbar_ax.set_ylabel('Total Precipitation (mm)', labelpad=10, rotation=270, fontsize=10)
            cbar_ax.set_yticklabels(['0', '', '20','','40','','60','','80',''])
        elif haz_var == 't2m':
            c = axs[0,0].contourf(lons, lats, haz.data-273.15,
                                  levels=np.linspace(np.min(haz.data-273.15),np.max(haz.data-273.15), 9),
                                  cmap = plt.cm.get_cmap('RdBu_r'),
                                  transform=ccrs.PlateCarree(),
                                  extend='max')

        if circ_plot == 1:
            c2 = axs[0,0].contour(lons, lats, circ.data/100,
                                  colors='k', transform=ccrs.PlateCarree(),
                                  extend='both')
            axs[0,0].clabel(c2, inline=1, fontsize=12)

        axs[0,0].add_feature(cfeature.BORDERS, color='grey')
        axs[0,0].add_feature(cfeature.COASTLINE, color='grey')
        axs[0,0].set_title('Event: '+str(event_date[2])+event_date[1]+str(event_date[0]), loc='left',
                           fontsize=8)

        for i, ax in enumerate(np.ravel(axs)[1:n+1]):
            if haz_var == 'tp':
                c = ax.contourf(lons, lats, haz_past[i].data,
                                levels=np.linspace(1, 80, 9),
                                cmap = cmap,
                                transform=ccrs.PlateCarree(),
                                extend='max')
            elif haz_var == 't2m':
                c = ax.contourf(lons, lats, haz_past[i].data-273.15,
                                levels=np.linspace(np.min(haz_past[i].data-273.15), np.max(haz_past[i].data-273.15), 9),
                                cmap = plt.cm.get_cmap('RdBu_r'),
                                transform=ccrs.PlateCarree(),
                                extend='max')

            if circ_plot == 1:
                c2 = ax.contour(lons, lats, circ_past[i].data/100,
                                colors='k', transform=ccrs.PlateCarree(),
                                extend='both')
                ax.clabel(c2, inline=1, fontsize=12)

            ax.add_feature(cfeature.BORDERS, color='grey')
            ax.add_feature(cfeature.COASTLINE, color='grey')
            ax.set_title('Analogue: '+str(dates_past[i][-2:])+calendar.month_abbr[int(dates_past[i][4:-2])]+str(dates_past[i][:4]), loc='left',
                         fontsize=8)

        return fig, axs


    # Util
    ######################################################################################

    # guess bounds
    @staticmethod
    def guess_bounds(cube, lat:str='latitude', lon:str='longitude'):
        if not cube.coord(lat).has_bounds():
            cube.coord(lat).guess_bounds()
        if not cube.coord(lon).has_bounds():
            cube.coord(lon).guess_bounds()

        return cube

    @staticmethod
    def remove_bounds(cube, lat:str='latitude', lon:str='longitude'):
        if cube.coord(lat).has_bounds():
            cube.coord(lat).bounds = None
        if cube.coord(lon).has_bounds():
            cube.coord(lon).bounds = None

        return cube

    # get the ana_van and haz_var values for a range of single dates
    @staticmethod
    def ana_and_haz_date_values(ana_var:str, haz_var:str, cube_map,
                                region:list[float], dates:list):

        # Prst date fields
        circ_vals = iris.cube.CubeList([])
        haz_vals = iris.cube.CubeList([])

        n = len(dates)
        for each in range(n):
            year = int(dates[each][:4])
            month = calendar.month_abbr[int(dates[each][4:-2])]
            day = int(dates[each][-2:])

            # print(ana_var, [year, month, day])
            circ_vals.append(Analogues.extract_region(Analogues.extract_date_v2(cube_map[ana_var], [year, month, day]), region))
            # print(haz_var, [year, month, day])
            haz_vals.append(Analogues.extract_region(Analogues.extract_date_v2(cube_map[haz_var], [year, month, day]), region))

        return circ_vals, haz_vals, n

    # values for variable choice
    @staticmethod
    def variable_choice_values(cube, event_date, dates_z500, dates_slp):

        II_event = Analogues.impact_index_v2(Analogues.extract_date_v2(cube, event_date))

        II_z500 = []
        daily_analogues = Analogues.analogues_list(cube, dates_z500)
        for each in daily_analogues:
            II_z500.append(Analogues.impact_index_v2(each))

        II_slp = []
        daily_analogues = Analogues.analogues_list(cube, dates_slp)
        for each in daily_analogues:
            II_slp.append(Analogues.impact_index_v2(each))

        return II_event, II_z500, II_slp


    # excess
    ######################################################################################

    # J: this function is never used
    @staticmethod
    def plot_specified_date(axs, fig, ax, date, title, ana_var, haz_var, R1): 
        circ = Analogues.extract_region(Analogues.reanalysis_data_single_date(ana_var, date), R1)
        #E = E - E.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        haz = Analogues.extract_region(Analogues.reanalysis_data_single_date(haz_var, date), R1)
        lats=haz.coord('latitude').points
        lons=haz.coord('longitude').points
        if haz_var == 'tp':
            c = ax.contourf(lons, lats, haz.data, levels=np.linspace(1, 80, 9), cmap = plt.cm.get_cmap('Blues'), transform=ccrs.PlateCarree(), extend='max')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
            fig.colorbar(c, cax=cbar_ax, ticks=np.arange(0, 100, 10))
            cbar_ax.set_ylabel('Total Precipitation (mm)', labelpad=10, rotation=270, fontsize=12)
            cbar_ax.set_yticklabels(['0', '', '20','','40','','60','','80',''])
        elif haz_var == 't2m':
            c = ax.contourf(lons, lats, haz.data-273.15, levels=np.linspace(np.min(haz.data-273.15), np.max(haz.data-273.15), 9), cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='max')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
            fig.colorbar(c, cax=cbar_ax, ticks=np.arange(np.min(haz.data-273.15), np.max(haz.data-273.15), 5))
            cbar_ax.set_ylabel('t2m', labelpad=10, rotation=270, fontsize=12)
            #cbar_ax.set_yticklabels(['0', '', '20','','40','','60','','80',''])
        lats=circ.coord('latitude').points
        lons=circ.coord('longitude').points
        c2 = axs.contour(lons, lats, circ.data/100, colors='k', transform=ccrs.PlateCarree(), extend='both')
        axs.clabel(c2, inline=1, fontsize=12)
        axs.add_feature(cfeature.BORDERS)
        axs.add_feature(cfeature.COASTLINE)
        axs.set_title(title, loc='left')