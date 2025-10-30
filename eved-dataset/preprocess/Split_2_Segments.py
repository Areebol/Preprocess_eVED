import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# === Haversine 计算两点间距离（单位：米） ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a)) * 1000

# === Segment 计算函数 ===
def process_file(file_path, segment_len=50.0):
    df = pd.read_csv(file_path)

    cols = {
        "latitude": "Matchted Latitude[deg]",
        "longitude": "Matched Longitude[deg]",
        "gradient": "Gradient",
        "energy": "Energy_Consumption",
        "speed": "Vehicle Speed[km/h]",
        "speed_limit": "Speed Limit with Direction[km/h]",
        "focus": "Focus Points",
        "bus_stop": "Bus Stops",
        "intersection": "Intersection",
        "class_speed_limit": "Class of Speed Limit",
    }

    for new, old in cols.items():
        if old in df.columns:
            df[new] = df[old]
        elif old == "Focus Points":
            df[new] = df["Focus Points;"]

    df['speed_limit'] = df['speed_limit'].astype(float) 
    mean_speed_limit = df['speed_limit'].mean(skipna=True)  
    df['speed_limit'] = df['speed_limit'].fillna(mean_speed_limit)

    df['gradient'] = df['gradient'].fillna(0).astype(float)
    df["focus"] = df["focus"].fillna("none").astype(str)
    df["bus_stop"] = df["bus_stop"].fillna(0).astype(int)
    df["intersection"] = df["intersection"].fillna(0).astype(int)
    df["class_speed_limit"] = df["class_speed_limit"].fillna(-2).astype(int)

    for k in ["latitude", "longitude", "gradient", "energy", "speed", "speed_limit"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    cum_dist = np.zeros(len(lat))
    for i in range(1, len(lat)):
        cum_dist[i] = cum_dist[i-1] + haversine(lat[i-1], lon[i-1], lat[i], lon[i])

    cut_points = np.arange(0, cum_dist[-1], segment_len)
    cut_points = np.append(cut_points, cum_dist[-1])

    interp = {k: np.interp(cut_points, cum_dist, df[k]) for k in ["latitude","longitude","gradient","energy","speed","speed_limit"]}

    def segment_mean(arr, lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        return np.mean(arr[mask]) if mask.any() else np.interp((lo + hi)/2, cum_dist, arr)

    def segment_mode_focus(lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            return df.loc[mask, "focus"].value_counts().idxmax()
        idx = np.argmin(np.abs(cum_dist - (lo + hi)/2))
        return df.loc[idx, "focus"]

    def segment_bus_stop(lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            return int(df.loc[mask, "bus_stop"].max())
        idx = np.argmin(np.abs(cum_dist - (lo + hi)/2))
        return int(df.loc[idx, "bus_stop"])

    def segment_intersection(lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            return int(df.loc[mask, "intersection"].max())
        idx = np.argmin(np.abs(cum_dist - (lo + hi)/2))
        return int(df.loc[idx, "intersection"])

    def segment_mode(arr_name, lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            values = df.loc[mask, arr_name].dropna()
            if not values.empty:
                return values.value_counts().idxmax()
        idx = np.argmin(np.abs(cum_dist - (lo + hi)/2))
        return df.loc[idx, arr_name]

    segments = []
    for i in range(len(cut_points)-1):
        lo, hi = cut_points[i], cut_points[i+1]
        start, end = (interp["latitude"][i], interp["longitude"][i]), (interp["latitude"][i+1], interp["longitude"][i+1])
        segments.append({
            "start_lat": start[0],
            "start_lon": start[1],
            "end_lat": end[0],
            "end_lon": end[1],
            "length_m": haversine(*start, *end),
            "mean_gradient": segment_mean(df["gradient"].to_numpy(), lo, hi),
            "mean_energy_consumption": segment_mean(df["energy"].to_numpy(), lo, hi),
            "mean_vehicle_speed": segment_mean(df["speed"].to_numpy(), lo, hi),
            "speed_limit": segment_mean(df["speed_limit"].to_numpy(), lo, hi),
            "focus_point": segment_mode_focus(lo, hi),
            "bus_stop": segment_bus_stop(lo, hi),
            "intersection": segment_intersection(lo, hi),
            "class_speed_limit": segment_mode("class_speed_limit", lo, hi),
        })

    return pd.DataFrame(segments)

EVs_ids = [10, 455, 541]
PHEVs_ids = [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457, 492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]

categories = {
    "EV": EVs_ids,
    "PHEV": PHEVs_ids
}

for category, ids in categories.items():
    for vehid in ids:
        input_dir = f"./data/filtered_vehID_eVED/{category}/{vehid}/"
        output_dir = f"./data/segmented_50m_eVED/{category}/{vehid}/"
        os.makedirs(output_dir, exist_ok=True) 
        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]

        all_segments = []
        with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
            for f in all_files:
                seg_df = process_file(f)
                if seg_df[['mean_energy_consumption', 'mean_vehicle_speed']].isna().any().any():
                    print(f"Skipping file {os.path.basename(f)} due to NaN in energy or speed")
                    pbar.update(1)
                elif seg_df.isna().any().any():
                    raise ValueError(f"NaN detected in segment DataFrame for file {os.path.basename(f)}")
                else:
                    out_file = os.path.join(output_dir, os.path.basename(f))
                    seg_df.to_csv(out_file, index=False)
                pbar.set_postfix(file=os.path.basename(f), segments=len(seg_df))
                pbar.update(1)