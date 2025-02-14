import os
import netCDF4 as nc
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd

from shapely.vectorized import contains


def plot_europe_map(df_places, output_filename, show_plot=False):
    """
    Plot a map of European NUTS3 regions with boundaries and dissolved country borders.
    
    Parameters:
        df_places (GeoDataFrame): GeoDataFrame with NUTS3 geometries.
        output_filename (str): File path where the map image will be saved.
        show_plot (bool): Whether to display the plot interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot boundaries for each NUTS3 region
    df_places.boundary.plot(ax=ax, color='black', linewidth=0.4)
    
    # Create a 'country' column using the first two letters of NUTS_ID for grouping
    df_places['country'] = df_places['NUTS_ID'].apply(lambda x: x[:2])
    # Dissolve boundaries by country
    df_dissolved = df_places.dissolve(by='country')
    # Plot country boundaries in a different style
    df_dissolved.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.tight_layout()
    # Save the figure to disk
    fig.savefig(output_filename, dpi=300)
    if '.png' in output_filename:
        fig.savefig(output_filename.replace('.png', '.pdf'))
    print(f"Europe map saved to {output_filename}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def get_coord2nuts(polygon, latitude_grid, longitude_grid):
    """
    For a given polygon (assumed to be in EPSG:4326) and grid arrays (latitude and longitude in EPSG:4326),
    return the list of grid indices [i, j] that fall inside the polygon.
    Here i indexes latitude and j indexes longitude.
    
    Steps:
      1. Restrict grid indices based on the polygon's bounding box.
      2. Create a meshgrid for the restricted region.
      3. Use vectorized contains to test which points fall inside the polygon.
      4. If no grid points are found, use the grid point closest to the polygon's centroid.
    """
    # Get polygon bounds (minx, miny, maxx, maxy) in EPSG:4326
    minx, miny, maxx, maxy = polygon.bounds
    
    # Restrict grid indices based on these bounds.
    lat_idx = np.where((latitude_grid >= miny) & (latitude_grid <= maxy))[0]
    lon_idx = np.where((longitude_grid >= minx) & (longitude_grid <= maxx))[0]
    
    coords = []
    if lat_idx.size > 0 and lon_idx.size > 0:
        # Create a meshgrid for the restricted region.
        sub_lat = latitude_grid[lat_idx]
        sub_lon = longitude_grid[lon_idx]
        mesh_lon, mesh_lat = np.meshgrid(sub_lon, sub_lat)  # shapes: (len(lat_idx), len(lon_idx))
    
        # Use vectorized 'contains' to test which meshgrid points lie within the polygon.
        mask = contains(polygon, mesh_lon, mesh_lat)
        rel_i, rel_j = np.where(mask)
        # Map relative indices back to the global grid indices.
        global_i = lat_idx[rel_i]
        global_j = lon_idx[rel_j]
        coords = list(zip(global_i.tolist(), global_j.tolist()))
    
    # Fallback: if no grid points were found, choose the grid point closest to the polygon's centroid.
    if not coords:
        centroid = polygon.centroid  # centroid in EPSG:4326
        lat_centroid = centroid.y
        lon_centroid = centroid.x
        i_closest = int(np.abs(latitude_grid - lat_centroid).argmin())
        j_closest = int(np.abs(longitude_grid - lon_centroid).argmin())
        coords.append((i_closest, j_closest))
        print(f"No grid points found; using closest grid point at index ({i_closest}, {j_closest}).")
    else:
        print(f"Polygon {polygon} contains {len(coords)} grid points.")
    
    return coords

def get_mean_location(variable_array, location):
    """
    Given a variable array (assumed shape: time x lon x lat) and a list of grid indices,
    compute the mean time series over those grid points.
    """
    if not location:
        # Return an array of NaNs if no grid points were found
        return np.full(variable_array.shape[0], np.nan)
    # Gather time series from each grid point in the location
    values = np.array([variable_array[:, i, j] for i, j in location])
    # Compute the mean across grid points (axis=0: time series mean)
    return values.mean(axis=0)

def get_mean_location(variable_array, coords):
    """
    Given a variable array (assumed shape: time x lat x lon) and a list of grid indices (coords),
    compute the mean time series over those grid points using vectorized indexing.
    
    Parameters:
        variable_array (np.ndarray): Array of shape (T, lat, lon).
        coords (list): List of (i, j) pairs.
        
    Returns:
        np.ndarray: Mean time series over the selected grid cells (length T).
    """
    if not coords:
        # Return an array of NaNs if no grid points were found.
        return np.full(variable_array.shape[0], np.nan)
    
    # Convert coords list to a NumPy array of shape (N, 2).
    coords_arr = np.array(coords)
    i_idx = coords_arr[:, 0]
    j_idx = coords_arr[:, 1]
    
    # Use advanced indexing: variable_array[:, i_idx, j_idx] has shape (T, N)
    # Then compute the mean along the grid cell dimension (axis=1).
    return variable_array[:, i_idx, j_idx].mean(axis=1)

def get_variable(ds, variable_name, df_places):
    """
    Extract a time series for a specified variable from the netCDF dataset for each region in df_places.
    Each region uses its grid indices (stored in 'coord') to compute the mean, using vectorized operations.
    
    Parameters:
        ds: netCDF4.Dataset object.
        variable_name (str): Name of the variable to extract (assumed dimensions: time x lat x lon).
        df_places: GeoDataFrame with a 'coord' column containing lists of grid indices.
        
    Returns:
        list: List of time series (np.ndarray) for each region.
    """
    # Extract the variable array (assuming dimensions: time x lat x lon)
    variable_array = np.asarray(ds.variables[variable_name][:])
    
    # Compute the mean time series for each region using the vectorized get_mean_location.
    variable_table = [get_mean_location(variable_array, coords) for coords in df_places['coord']]
    
    return variable_table


def process_nc_file(input_path, file_name, location_file, output_path, variable):
    """
    Process a netCDF file using a CSV file (location_file) that contains grid indices
    (in a column named 'coord') and produce a CSV with the variable time series for each region.
    """
    ds = nc.Dataset(os.path.join(input_path, file_name))
    df_places = pd.read_csv(os.path.join(input_path, location_file))
    # Convert string representation of lists back to actual lists
    df_places['coord'] = df_places['coord'].apply(eval)
    variable_table = get_variable(ds, variable, df_places)
    pd.DataFrame(variable_table).to_csv(os.path.join(output_path, variable + '.csv'), index=False)
    ds.close()





def main():
    # Set file paths
    nc_path = '/work/louibo/climate/data'
    geojson_file = '/work/louibo/climate/Climate/physical_risks/nuts_rg_60m_2013_lvl_3.geojson'
    output_fig_dir = 'figures'
    
    # Create output directory if it does not exist
    os.makedirs(output_fig_dir, exist_ok=True)
    
    # Load geospatial data (NUTS3 boundaries)
    df_places = gpd.read_file(geojson_file)
    
    # Filter out non-European regions based on centroid longitude (adjust as needed)
    df_places = df_places[df_places.apply(lambda row: -20 < row['geometry'].centroid.x < 40, axis=1)]
    
    # Reproject to EPSG:3035 for a nicer Europe map
    df_places = df_places.to_crs(epsg=3035)
    
    # Plot the map of Europe using the dedicated function
    europe_map_path = os.path.join(output_fig_dir, 'europe_map.png')
    plot_europe_map(df_places, europe_map_path, show_plot=True)
    
    # Reproject polygons back to EPSG:4326 since our grid is in lon/lat
    df_places_4326 = df_places.to_crs(epsg=4326)
    
    # Use one netCDF file to extract the grid (assuming all files share the same grid)
    nc_files = glob.glob(os.path.join(nc_path, "*.nc"))
    if not nc_files:
        print("No netCDF files found in", nc_path)
        return
    ds_temp = nc.Dataset(nc_files[0])
    latitude_grid = ds_temp.variables['latitude'][:]
    longitude_grid = ds_temp.variables['longitude'][:]
    ds_temp.close()
    
    # For each region, compute grid indices using the grid (lat, lon in EPSG:4326)
    df_places_4326['coord'] = df_places_4326['geometry'].apply(
        lambda geom: get_coord2nuts(geom, latitude_grid, longitude_grid)
    )
    
    # Check how many regions still have an empty 'coord'
    empty_counts = df_places_4326['coord'].apply(lambda x: len(x) == 0).value_counts()
    print("Empty coord counts:\n", empty_counts)
    
    # Save the GeoDataFrame with grid indices to CSV
    df_places_4326.to_csv('/data/nuts3_grid_indices.csv', index=False)
    
    # Loop over all netCDF files in nc_path
    for nc_file in nc_files:
        print("Processing file:", nc_file)
        ds = nc.Dataset(nc_file)
        # Use the last variable in the netCDF file (adjust if needed)
        variable_name = list(ds.variables.keys())[-1]
        print("Extracting variable:", variable_name)
        variable_table = get_variable(ds, variable_name, df_places_4326)
        # Save the regional time series to CSV; filename based on the netCDF file name and variable
        base_name = os.path.basename(nc_file)
        output_csv = os.path.join(nc_path, f"{os.path.splitext(base_name)[0]}_{variable_name}.csv")
        pd.DataFrame(variable_table).to_csv(output_csv, index=False)
        print(f"Variable time series saved to {output_csv}")
        ds.close()


if __name__ == '__main__':
    main()
