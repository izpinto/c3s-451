import requests
import numpy as np
from shapely.geometry import Polygon
import webbrowser


# Sekect a region using the C3S-451 Region Picker service
def select_region(regionType):
    
    allowed_region_types = ['wraf', 'hydrobasin']
    
    if regionType not in allowed_region_types:
        raise ValueError(f"Invalid regionType '{regionType}'. Allowed values are: {allowed_region_types}")
    
    url = f"http://c3s-451/region-picker/start-m2m/{regionType}"
    
    response = requests.get(url)
    
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