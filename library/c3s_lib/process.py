from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd

# Rewrite so only the mean is calculated and not yet the anomoly
def calculate_anomaly(final_gdf, gdfs, value_col="t2m", datetime_col='valid_time'):
    # Ensure datetime columns
    final_gdf = final_gdf.copy()
    gdfs = gdfs.copy()
    final_gdf[datetime_col] = pd.to_datetime(final_gdf[datetime_col])
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])
    
    # Prepare list for adjusted results
    adjusted_list = []
    
    for target_date in gdfs[datetime_col].unique():
        # --- Step 1: get mean around date for all years ---
        results = []
        for year in final_gdf[datetime_col].dt.year.unique():
            center_date = datetime(year, target_date.month, target_date.day)
            start_window = center_date - timedelta(days=15)
            end_window = center_date + timedelta(days=15)
            
            subset = final_gdf[
                (final_gdf[datetime_col] >= start_window) &
                (final_gdf[datetime_col] <= end_window)
            ]
            results.append(subset)
        
        combined = pd.concat(results, ignore_index=True)
        mean_gdf = combined.groupby("geometry")[value_col].mean().reset_index()
        
        # --- Step 2: subtract from gdfs for this target date ---
        current_day_gdf = gdfs[gdfs[datetime_col] == target_date]
        
        # Merge on geometry so subtraction matches locations
        merged = current_day_gdf.merge(mean_gdf, on="geometry", suffixes=("", "_mean"))

        if value_col == "t2m":
            merged[value_col] = merged[value_col] - merged[f"{value_col}_mean"]
        elif value_col == "tp":
            merged[value_col] = merged[value_col] / merged[f"{value_col}_mean"]
        else:
            raise ValueError("value_col must be 't2m' or 'tp'")
    
        merged.drop(columns=[f"{value_col}_mean"], inplace=True)
        
        adjusted_list.append(merged)
    
    # Combine adjusted daily frames back together
    final_adjusted_gdf = gpd.GeoDataFrame(
        pd.concat(adjusted_list, ignore_index=True),
        crs=gdfs.crs
    )
    
    return final_adjusted_gdf

def test_calculate_anomaly(gdf, gdfs, padding:int=15, value_col:str="t2m", datetime_col:str='valid_time'):
    # Ensure datetime columns
    gdf = gdf.copy()
    gdfs = gdfs.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])
    
    # Prepare list for adjusted results
    adjusted_list = []
    
    for target_date in gdfs[datetime_col].unique():
        # --- Step 1: get mean around date for all years ---
        results = []
        for year in gdf[datetime_col].dt.year.unique():
            center_date = datetime(year, target_date.month, target_date.day)
            start_window = center_date - timedelta(days=padding)
            end_window = center_date + timedelta(days=padding)
            
            subset = gdf[
                (gdf[datetime_col] >= start_window) &
                (gdf[datetime_col] <= end_window)
            ]
            results.append(subset)
        
        combined = pd.concat(results, ignore_index=True)
        mean_gdf = combined.groupby("geometry")[value_col].mean().reset_index()
        
        # --- Step 2: subtract from gdfs for this target date ---
        current_day_gdf = gdfs[gdfs[datetime_col] == target_date]
        
        # Merge on geometry so subtraction matches locations
        merged = current_day_gdf.merge(mean_gdf, on="geometry", suffixes=("", "_mean"))

        if value_col == "t2m":
            merged[value_col] = merged[value_col] - merged[f"{value_col}_mean"]
        elif value_col == "tp":
            merged[value_col] = merged[value_col] / merged[f"{value_col}_mean"]
        else:
            raise ValueError("value_col must be 't2m' or 'tp'")
    
        merged.drop(columns=[f"{value_col}_mean"], inplace=True)
        
        adjusted_list.append(merged)
    
    # Combine adjusted daily frames back together
    final_adjusted_gdf = gpd.GeoDataFrame(
        pd.concat(adjusted_list, ignore_index=True),
        crs=gdfs.crs
    )
    
    return final_adjusted_gdf


