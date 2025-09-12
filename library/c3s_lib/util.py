import requests
import numpy as np
from shapely.geometry import Polygon
import webbrowser
from urllib.parse import urlencode
from typing import Dict, Any
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
import base64
from io import BytesIO
from c3s_lib import plot


# Select a region using the C3S-451 Region Picker service
def select_region(regionType:str, params:Dict[str, Any]=None):
    
    allowed_region_types = ['wraf', 'hydrobasin']
    
    if regionType not in allowed_region_types:
        raise ValueError(f"Invalid regionType '{regionType}'. Allowed values are: {allowed_region_types}")
    
    print('The region picker will shortly open in your web browser. Please select a region, close the browser tab and return to the notebook when done.')
    
    url = f"http://c3s-451/region-picker/start-m2m/{regionType}"

    #if params != None:
    #    url += urlencode(params)
    
    response = requests.post(url=url, json=params)
    
    poll_url = False
    
    if response.status_code == 200:
        data = response.json()
        print(f"Region Picker started successfully for {regionType}:")
        #print(f"Open the following page in your browser to select a region: ")
        #print(f"\t\t{data['url']}")
        webbrowser.open(data['url'])
        poll_url = data['poll_url']
    else:
        print(f"Failed to start Region Picker for {regionType}. Status code: {response.status_code}")
        print(f"Response: {response.text}")
    
    result = None
    
    if poll_url:
        print(f"Polling for region selection...")
        
        done = False
        
        while not done:
            response = requests.get(poll_url)
            if response.status_code == 200:
                data = response.json()
                if data['done']:
                    done = True
                    result = data['result']
            else:
                print(f"Failed to poll Region Picker for {regionType}. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                break
        
        print("Region selection process done.")
    
    print("Received polygon data:")
    print(result)

    return result


# wrap lat lon coordinates to ensure they are within the range [-180, 180] for longitude
def wrap_lon(ds):
    
    if "longitude" in ds.coords:
        lon = "longitude"
        lat = "latitude"
    elif "lon" in ds.coords:
        lon = "lon"
        lat = "lat"
    else: 
        # can only wrap longitude
        return ds
    
    if ds[lon].max() > 180:
        ds[lon] = (ds[lon].dims, (((ds[lon].values + 180) % 360) - 180), ds[lon].attrs)
        
    if lon in ds.dims:
        ds = ds.reindex({ lon : np.sort(ds[lon]) })
        ds = ds.reindex({ lat : np.sort(ds[lat]) })
    return ds

def data_2_poly(data):
    all_coords = []  
    polygons = []    

    for feature in data["features"]:
        coords = feature['geometry']['coordinates'][0]
        all_coords.extend(coords)  
        polygons.append(Polygon(coords))
    
    return polygons, all_coords

def get_base_fig(date, gdf, value_col:str, datetime_col:str='valid_time', dpi:int=100, cmap=None, projection=ccrs.PlateCarree(), show_fig:bool=False, marker:str='s'):


    selected_gdf_anomoly = gdf[(gdf[datetime_col] >= date) & (gdf[datetime_col] <= date)]

    vmin = gdf[value_col].min()
    vmax = gdf[value_col].max()

    cmap, norm = plot.get_colormap(cmap if cmap else value_col, vmin, vmax)


    fig, ax = plt.subplots(
        ncols = 1, nrows = 1, figsize = (5,5), dpi = dpi, 
        subplot_kw = {"projection" : projection}
    )

    # ax.plot(selected_gdf_anomoly['longitude'], selected_gdf_anomoly['latitude'], "o", markersize=1)  # markersize = diameter in points

    temp_kwargs = {"cmap" : cmap, "norm": norm}

    selected_gdf_anomoly.plot(ax = ax, **temp_kwargs,
        column = value_col,
        vmin = vmin,
        vmax = vmax,
        marker = marker
    )

    ax.set_axis_off()
    plt.tight_layout()

    # Save to memory buffer instead of file
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, transparent=True, bbox_inches="tight", pad_inches=0)
    if not show_fig:
        plt.close(fig)  # Close the figure to avoid displaying it in non-interactive environments
    buf.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return img_base64