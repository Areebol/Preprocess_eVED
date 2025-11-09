import os
import pandas as pd

def get_lat_lon_range(dataset_dir, lat_col='Matchted Latitude[deg]', lon_col='Matched Longitude[deg]'):
    """
    Extract the global min and max latitude/longitude values 
    across all CSV files in a given dataset directory.

    Args:
        dataset_dir (str): Path to the dataset folder containing CSV files.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.

    Returns:
        tuple: (lat_min, lat_max, lon_min, lon_max)
    """
    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')

    # Walk through all files in the dataset directory
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path)

                    # Skip files that do not contain the required columns
                    if lat_col not in df.columns or lon_col not in df.columns:
                        print(f"Skipping {file}: missing columns '{lat_col}' or '{lon_col}'")
                        continue

                    # Update global min/max values
                    lat_min = min(lat_min, df[lat_col].min())
                    lat_max = max(lat_max, df[lat_col].max())
                    lon_min = min(lon_min, df[lon_col].min())
                    lon_max = max(lon_max, df[lon_col].max())

                except Exception as e:
                    print(f"Error reading {file}: {e}")

    return lat_min, lat_max, lon_min, lon_max


if __name__ == "__main__":
    if False:
        dataset_dir = "../data/eVED/"  # <-- Replace with your dataset path
        lat_min, lat_max, lon_min, lon_max = get_lat_lon_range(dataset_dir)

        print("==== Dataset Latitude/Longitude Range ====")
        print(f"Latitude range: {lat_min:.6f} ~ {lat_max:.6f}")
        print(f"Longitude range: {lon_min:.6f} ~ {lon_max:.6f}")

        # ==== Dataset Latitude/Longitude Range ====
        # Latitude range: 42.220268 ~ 42.325853
        # Longitude range: -83.804839 ~ -83.673437
    else:
        import math

        # Input range
        lat_min, lat_max = 42.220268, 42.325853
        lon_min, lon_max = -83.804839, -83.673437

        # Earth radius (km)
        R = 6371.0

        # Convert degree differences to radians
        dlat = math.radians(lat_max - lat_min)
        dlon = math.radians(lon_max - lon_min)
        mean_lat = math.radians((lat_max + lat_min) / 2)

        # Distance calculations
        height_km = R * dlat                     # North-South distance
        width_km = R * dlon * math.cos(mean_lat) # East-West distance
        area_km2 = height_km * width_km          # Approximate area

        print(f"Height (North–South): {height_km:.3f} km")
        print(f"Width (East–West):   {width_km:.3f} km")
        print(f"Approx. Area:        {area_km2:.3f} km^2")
        # Height (North–South): 11.741 km
        # Width (East–West):   10.812 km
        # Approx. Area:        126.933 km^2

