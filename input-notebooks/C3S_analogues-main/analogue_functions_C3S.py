# Functions for analogues

import subprocess
import iris
import iris.coord_categorisation as icc # type: ignore
from iris.coord_categorisation import add_season_membership
import numpy as np
import cartopy
import cartopy.crs as ccrs
import glob
import matplotlib.cm as mpl_cm
import os, sys
import scipy.stats as sps
from scipy.stats import genextreme as gev
import random
import scipy.io
import xarray as xr
import netCDF4 as nc
import iris.coords # type: ignore
import iris.util # type: ignore
from iris.util import equalise_attributes # type: ignore
from iris.util import unify_time_units # type: ignore
from scipy.stats.stats import pearsonr
import scipy.stats as stats
import calendar
import random
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris.analysis


import warnings


# Are we using daily or monthly data?
ERA5FILESUFFIX : str = "_daily"

def reanalysis_file_location() -> str:
    '''
    Return the location of the ERA5 data
    '''
    #CEX path 
    #return os.path.join(os.environ["CLIMEXP_DATA"],"ERA5")
    #local wrkstation path 
    return '/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5-CX-READY/'
    

def find_reanalysis_filename(var : str, daily : bool = True) -> str:
    '''
    Return the field filename for a given variable
    '''
    suffix : str = ERA5FILESUFFIX
    path : str = os.path.join(reanalysis_file_location(),"era5_{0}{1}.nc".format(var,suffix))
    return path


def event_data_era(event_data, date: list, ana_var: str) -> list:
    '''
    Get ERA data for a defined list of variables on a given event date
    '''
    event_list = []
    # TODO: Convert variable list into a non-local (possibly with a class?)
    if event_data == 'extended':
        for variable in [ana_var, 'tp', 't2m', 't2m']:
            event_list.append(reanalysis_data_single_date(variable, date))
    else:
        for variable in [ana_var, 'tp', 't2m', 'sfcWind']:
            event_list.append(reanalysis_data_single_date(variable, date))
    return event_list

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
        NEXT_FIELD = pull_out_day_era(P1_field, year, month, day)
        if NEXT_FIELD == None:
            print('Field failure for: ',+each)
            n = n-1
        else:
            if FIELD == 0:
                FIELD = NEXT_FIELD
            else:
                FIELD = FIELD + NEXT_FIELD
    return FIELD/n

def ED_similarity(event, P_cube, region, method):
    '''
    Returns similarity values based on euclidean distance
    '''
    E = extract_region(event, region)
    P = extract_region(P_cube, region)
    D = []
    for yx_slice in P.slices(['grid_latitude', 'grid_longitude']):
        if method == 'ED':
            D.append(euclidean_distance(yx_slice, E))
        elif method == 'CC':
            D.append(correlation_coeffs(yx_slice, E))
    ED_max = np.max(np.max(D))
    S = [(1-x / ED_max) for x in D]
    return S

def regrid(original, new):
    ''' Regrids onto a new grid '''
    mod_cs = original.coord_system(iris.coord_systems.CoordSystem)
    new.coord(axis='x').coord_system = mod_cs
    new.coord(axis='y').coord_system = mod_cs
    new_cube = original.regrid(new, iris.analysis.Linear())
    return new_cube

def extract_region(cube_list, R1):
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
            print(each)
            subset = cube_list[each].extract(const_lat)
            reg_cubes.append(subset.intersection(longitude=(R1[3], R1[2])))
    return reg_cubes

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


def reanalysis_data(var, Y1=1950, Y2=2023, months='[Jan]'):
    '''
    Loads in reanalysis daily data
    VAR can be psi250, msl, or tp (to add more)
    '''
    cubes = iris.load(find_reanalysis_filename(var), var)
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


def anomaly_period_outputs(Y1, Y2, ana_var, N, date, months, R1):
    '''
    Function to identify the N closest analogues of (N: number)
    date between (date format: [YYYY, 'Mon', DD], e.g. [2021, 'Jul', 14])
    Y1 and Y2, (year between 1950 and 2023)
    for 'event' 
    '''
    P1_msl = reanalysis_data(ana_var, Y1, Y2, months) # Get ERA5 data, Y1 to Y2, for var and season chosen. Global.
    P1_field = extract_region(P1_msl, R1) # Extract the analogues domain (R1) from global field
    ### difference for anomaly version
    P1_spatialmean = P1_field.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)   # Calculate spatial mean for each day
    P1_field = P1_field - P1_spatialmean # Remove spatial mean from each day
    event = reanalysis_data_single_date(ana_var, date)
    E = extract_region(event, R1) # Extract domain for event field
    E = E - E.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # remove spatial mean for event field
    ###
    P1_dates = analogue_dates_v2(E, P1_field, R1, N*5)[:N] # calculate the closest analogues
    if str(date[0])+str("{:02d}".format(list(calendar.month_abbr).index(date[1])))+str(date[2]) in P1_dates: # Remove the date being searched for
        P1_dates.remove(str(date[0])+str("{:02d}".format(list(calendar.month_abbr).index(date[1])))+str(date[2]))
    return P1_dates




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



def eucdist_of_datelist(event, reanalysis_cubelist, date_list, region):
    ED_list = []
    E = extract_region(event, region)
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
        field = extract_region(pull_out_day_era(reanalysis_cubelist, yr, mon, day), region)
        field = field*weights
        b, c = np.shape(field)
        XA = E.data.reshape(b*c,1)
        XB = field.data.reshape(b*c, 1)
        D = np.sqrt(np.sum(np.square(XA - XB)))
        ED_list.append(D)
    return ED_list

def analogue_dates_v2(event, reanalysis_cube, region, N):
    '''
    '''
    def cube_date_to_string(cube_date : tuple) -> tuple:
        year,month,day,time = cube_date
        return str(year)+str(month).zfill(2)+str(day).zfill(2), time
    E = extract_region(event, region)
    reanalysis_cube = extract_region(reanalysis_cube, region)
    D = euclidean_distance(reanalysis_cube, E)
    date_list = []
    time_list = []
    for i in np.arange(N):
        #print(i)
        I = np.sort(D)[i]
        for n, each in enumerate(D):
            if I == each:
                a1 = n
        date, time = cube_date_to_string(cube_date(reanalysis_cube[a1,...]))
        date_list.append(date)
        date_list2 = date_list_checks(date_list, days_apart=5)
    return date_list2


def pull_out_day_era(psi, sel_year, sel_month, sel_day):
    if type(psi)==iris.cube.Cube:
        psi_day = extract_date(psi, sel_year, sel_month, sel_day)
    else:
        for each in psi:
            if len(each.coords('year')) > 0:
                pass
            else:
                iris.coord_categorisation.add_year(each, 'time')
            if each.coord('year').points[0]==sel_year:
                psi_day = extract_date(each, sel_year, sel_month, sel_day)
            else:
                pass
    try:
        return psi_day
    except NameError:
        print('ERROR: Date not in data')
        return

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
        NEXT_FIELD = pull_out_day_era(psi, year, month, day)
        if NEXT_FIELD == None:
            print('Field failure for: ',+each)
            n = n-1
        else:
            if FIELD == 0:
                FIELD = NEXT_FIELD
            else:
                FIELD = FIELD + NEXT_FIELD
    return FIELD/n


def reanalysis_data_single_date(var : str, date : list):
    '''
    Loads in reanalysis daily data
    VAR can be: msl, or tp (to add more)
    '''
    filename = find_reanalysis_filename(var)
    print("Read file: {} for date {}".format(filename,date))
    cube = iris.load(filename, var)[0]
    cube = extract_date(cube,date[0],date[1],date[2])
    return cube


def quality_analogs(field, date_list, N, analogue_variable, region):
    '''
    For the 30 closest analogs of the event day, calculates analogue quality
    '''
    Q = []
    filename = find_reanalysis_filename(analogue_variable)
    cube = iris.load(filename, analogue_variable)[0]
    for i, each in enumerate(date_list):
        year = int(each[:4])
        month = calendar.month_abbr[int(each[4:-2])]
        day = int(each[-2:])
        cube = extract_date(cube,year,month,day)
        cube = extract_region(cube,region)
        D = euclidean_distance(field, cube) # calc euclidean distances
        Q.append(np.sum(np.sort(D)[:N]))
    return Q



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
        field_list1.append(pull_out_day_era(field1, year, month, day))
    n = len(dates2)
    field_list2 = iris.cube.CubeList([])
    for each in range(n):
        year = int(dates2[each][:4])
        month = calendar.month_abbr[int(dates2[each][4:-2])]
        day = int(dates2[each][-2:])
        field_list2.append(pull_out_day_era(field2, year, month, day))    
    sig_field = field_list1[0].data
    a, b = np.shape(field_list1[0].data)
    for i in range(a):
        print(i)
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
        field_list.append(pull_out_day_era(field, year, month, day))
    sig_field = field_list[0].data
    a, b = np.shape(field_list[0].data)
    for i in range(a):
        print(i)
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


def impact_index(cube, II_domain):
    '''
    Calculates the impact index over the cube
    II_domain: spatial extent of index
    '''
    cube = extract_region(cube, II_domain)
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    cube.data = ma.masked_invalid(cube.data)
    return cube.collapsed(('longitude','latitude'),iris.analysis.MEAN,weights=grid_areas).data

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
        NEXT_FIELD = pull_out_day_era(cube, year, month, day)
        if NEXT_FIELD == None:
            print('Field failure for: ',+each)
            n = n-1
        x.append(NEXT_FIELD)
    return x


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


def plot_box(axs, bdry):
    axs.plot([bdry[3], bdry[2]], [bdry[1], bdry[1]],'k')
    axs.plot([bdry[3], bdry[2]], [bdry[0], bdry[0]],'k')
    axs.plot([bdry[3], bdry[3]], [bdry[1], bdry[0]],'k')
    axs.plot([bdry[2], bdry[2]], [bdry[1], bdry[0]],'k')
    return

def set_coord_system(cube, chosen_system = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS):
    '''
    This is used to prevent warnings that no coordinate system defined
    Defaults to DEFAULT_SPHERICAL_EARTH_RADIUS
    '''
    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(chosen_system)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(chosen_system)
    return cube