import os
import osmnx as ox
import geopandas as gpd

def extract_speed_related_osm(lat_min, lat_max, lon_min, lon_max, save_path="./osm_speed_features"):
    """
    Extract and save OSM features related to vehicle speed, such as:
    - road network (with attributes like maxspeed, surface, lanes, oneway)
    - traffic lights
    - traffic calming devices
    - pedestrian crossings

    Args:
        lat_min, lat_max, lon_min, lon_max: Bounding box coordinates
        save_path: Folder to save GeoJSON outputs
    """
    north, south, east, west = lat_max, lat_min, lon_max, lon_min
    os.makedirs(save_path, exist_ok=True)

    features = {}

    # 1. Road network (with speed-related attributes)
    tags_roads = {
        'highway': True,
        'maxspeed': True,
        'lanes': True,
        'surface': True,
        'oneway': True,
        'junction': True,
    }
    try:
        roads = ox.features_from_bbox(north, south, east, west, tags=tags_roads)
    except Exception as e:
        print(f"Error fetching roads: {e}")
        roads = gpd.GeoDataFrame()
    features['roads'] = roads

    # 2. Traffic lights
    try:
        tl = ox.features_from_bbox(north, south, east, west, tags={'highway': 'traffic_signals'})
    except Exception as e:
        print(f"Error fetching traffic lights: {e}")
        tl = gpd.GeoDataFrame()
    features['traffic_lights'] = tl

    # 3. Traffic calming
    try:
        tc = ox.features_from_bbox(north, south, east, west, tags={'traffic_calming': True})
    except Exception as e:
        print(f"Error fetching traffic calming: {e}")
        tc = gpd.GeoDataFrame()
    features['traffic_calming'] = tc

    # 4. Crossings
    try:
        cross = ox.features_from_bbox(north, south, east, west, tags={'crossing': True})
    except Exception as e:
        print(f"Error fetching crossings: {e}")
        cross = gpd.GeoDataFrame()
    features['crossings'] = cross

    # Save each dataset
    for key, gdf in features.items():
        out_path = os.path.join(save_path, f"{key}.geojson")
        if not gdf.empty:
            gdf.to_file(out_path, driver="GeoJSON")
            print(f"Saved {out_path} ({len(gdf)} features)")
        else:
            # Create an empty placeholder file (so you always get all 4)
            gpd.GeoDataFrame(geometry=[]).to_file(out_path, driver="GeoJSON")
            print(f"No {key} found — created empty {out_path}")

    print("\n=== Extraction Done ===")
    for key, gdf in features.items():
        print(f"{key}: {len(gdf)} features")

    return features


if __name__ == "__main__":
    # Example bounding box (replace with your own)
    lat_min, lat_max = 42.220268, 42.325853
    lon_min, lon_max = -83.804839, -83.673437
    if False:
        extract_speed_related_osm(lat_min, lat_max, lon_min, lon_max)
        # === Extraction Done ===
        # roads: 44691 features
        # traffic_lights: 268 features
        # traffic_calming: 156 features
        # crossings: 6904 features
    else:
        import geopandas as gpd

        # gdf = gpd.read_file('./osm_speed_features/traffic_calming.geojson')
        # traffic_calming: table, hump, bump, no
        
        # gdf = gpd.read_file('./osm_speed_features/crossings.geojson')
        # highway: crossing footway
        
        # gdf = gpd.read_file('./osm_speed_features/traffic_lights.geojson')
        # highway: traffic_signals
        
        gdf = gpd.read_file("./osm_speed_features/roads.geojson")
        # highway: traffic_signlas, "gemotry": POINT
        # highway: motorway, trunk, primary, secondary, tertiary, residential, service, footway, path, living_street, pedestrian, etc. "gemotry": LINESTRING
        

        print(gdf.info())

        print(gdf.head(3))

        print("Columns:", list(gdf.columns))

        # ['element_type', 'osmid', 'traffic_calming', 'direction', 'surface', 'check_date', 'crossing', 'crossing:island', 'crossing:markings', 'crossing:signals', 'highway', 'kerb', 'tactile_paving', 'colour', 'flashing_lights', 'nodes', 'footway', 'lit', 'smoothness', 'area', 'wikimedia_commons', 'landuse', 'bicycle', 'cycleway', 'foot', 'geometry']
