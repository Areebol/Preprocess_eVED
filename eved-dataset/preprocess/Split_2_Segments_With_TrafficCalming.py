import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from shapely.geometry import LineString
import geopandas as gpd

# === Haversine 计算两点间距离（单位：米） ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a)) * 1000


# === 处理单个文件 ===
def process_file(file_path, traffic_gdf, traffic_calming_gdf, segment_len=50.0, buffer_m=25):
    df = pd.read_csv(file_path)

    # === 列映射 ===
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

    # === 基本清洗 ===
    df["speed_limit"] = df["speed_limit"].astype(float)
    df["speed_limit"] = df["speed_limit"].fillna(df["speed_limit"].mean(skipna=True))
    df["gradient"] = df["gradient"].fillna(0).astype(float)
    df["focus"] = df["focus"].fillna("none").astype(str)
    df["bus_stop"] = df["bus_stop"].fillna(0).astype(int)
    df["intersection"] = df["intersection"].fillna(0).astype(int)
    df["class_speed_limit"] = df["class_speed_limit"].fillna(-2).astype(int)

    for k in ["latitude", "longitude", "gradient", "energy", "speed", "speed_limit"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    # === 距离累计 ===
    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    cum_dist = np.zeros(len(lat))
    for i in range(1, len(lat)):
        cum_dist[i] = cum_dist[i - 1] + haversine(lat[i - 1], lon[i - 1], lat[i], lon[i])

    cut_points = np.arange(0, cum_dist[-1], segment_len)
    cut_points = np.append(cut_points, cum_dist[-1])
    interp = {k: np.interp(cut_points, cum_dist, df[k]) for k in ["latitude", "longitude", "gradient", "energy", "speed", "speed_limit"]}

    # === 辅助函数 ===
    def segment_mean(arr, lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        return np.mean(arr[mask]) if mask.any() else np.interp((lo + hi) / 2, cum_dist, arr)

    def segment_mode_focus(lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            return df.loc[mask, "focus"].value_counts().idxmax()
        idx = np.argmin(np.abs(cum_dist - (lo + hi) / 2))
        return df.loc[idx, "focus"]

    def segment_bus_stop(lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            return int(df.loc[mask, "bus_stop"].max())
        idx = np.argmin(np.abs(cum_dist - (lo + hi) / 2))
        return int(df.loc[idx, "bus_stop"])

    def segment_intersection(lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            return int(df.loc[mask, "intersection"].max())
        idx = np.argmin(np.abs(cum_dist - (lo + hi) / 2))
        return int(df.loc[idx, "intersection"])

    def segment_mode(arr_name, lo, hi):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        if mask.any():
            values = df.loc[mask, arr_name].dropna()
            if not values.empty:
                return values.value_counts().idxmax()
        idx = np.argmin(np.abs(cum_dist - (lo + hi) / 2))
        return df.loc[idx, arr_name]

    # === 投影转换 ===
    traffic_gdf_proj = traffic_gdf.to_crs(epsg=3857)
    traffic_calming_gdf_proj = traffic_calming_gdf.to_crs(epsg=3857)

    # === traffic lights ===
    def segment_traffic_lights(lo, hi, buffer_m=25):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        seg_lat = lat[mask]
        seg_lon = lon[mask]

        valid_mask = (~np.isnan(seg_lat)) & (~np.isnan(seg_lon))
        seg_lat = seg_lat[valid_mask]
        seg_lon = seg_lon[valid_mask]

        if len(seg_lat) == 0:
            return 0
        elif len(seg_lat) == 1:
            seg_lat = np.array([seg_lat[0], seg_lat[0]])
            seg_lon = np.array([seg_lon[0], seg_lon[0]])

        segment_line = LineString(zip(seg_lon, seg_lat))
        if segment_line.length == 0:
            return 0

        segment_line_proj = gpd.GeoSeries([segment_line], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

        count = (traffic_gdf_proj.distance(segment_line_proj) <= buffer_m).sum()
        return count

    # === traffic calming ===
    CALMING_CATEGORY = {
        "none": 0,
        "bump": 1,
        "hump": 2,
        "table": 3,
        "cushion": 4,
        "chicane": 5,
        "rumble_strip": 6,
        "dip": 7,
        "choker": 8,
        "other": 9,
    }

    # === traffic calming ===
    def segment_traffic_calming(lo, hi, buffer_m=50):
        mask = (cum_dist >= lo) & (cum_dist <= hi)
        seg_lat = lat[mask]
        seg_lon = lon[mask]

        # 移除 NaN 点
        valid_mask = (~np.isnan(seg_lat)) & (~np.isnan(seg_lon))
        seg_lat = seg_lat[valid_mask]
        seg_lon = seg_lon[valid_mask]

        # 空段直接返回 "none"
        if len(seg_lat) == 0:
            return "none"
        elif len(seg_lat) == 1:
            # 单点段复制成双点，防止 LineString 报错
            seg_lat = np.array([seg_lat[0], seg_lat[0]])
            seg_lon = np.array([seg_lon[0], seg_lon[0]])
    
        # 构造线段并投影到米制
        segment_line = LineString(zip(seg_lon, seg_lat))
        if segment_line.length == 0:
            return "none"
        segment_line_proj = gpd.GeoSeries([segment_line], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

        # 找出距离 <= buffer_m 的 calming 要素
        nearby = traffic_calming_gdf_proj[traffic_calming_gdf_proj.distance(segment_line_proj) <= buffer_m]

        # 如果没有任何 nearby，要返回 "none"
        if nearby.empty:
            return "none"
        print("nearby")
        # 尝试提取属性列（常见字段名 'traffic_calming'，若不存在尝试其他列）
        if "traffic_calming" in nearby.columns:
            vals = nearby["traffic_calming"].dropna().astype(str).str.lower()
        else:
            # 如果没有该字段，尝试从 properties 的其他可能字段获取（例如 'name'）
            # 否则统一标为 "other"
            possible_cols = [c for c in nearby.columns if c not in ["geometry"]]
            if possible_cols:
                vals = nearby[possible_cols[0]].dropna().astype(str).str.lower()
            else:
                return "other"

        if vals.empty:
            return "none"

        # 用众数（mode）作为 segment 的 raw 值
        mode_val = vals.mode()
        return mode_val.iloc[0] if not mode_val.empty else vals.iloc[0]

    def calming_to_int(name):
        name = str(name).lower()
        return CALMING_CATEGORY[name] if name in CALMING_CATEGORY else CALMING_CATEGORY["other"]

    # === 构建 segments ===
    segments = []
    for i in range(len(cut_points) - 1):
        lo, hi = cut_points[i], cut_points[i + 1]
        start, end = (interp["latitude"][i], interp["longitude"][i]), (interp["latitude"][i + 1], interp["longitude"][i + 1])
        raw_calming = segment_traffic_calming(lo, hi, buffer_m)
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
            "traffic_lights": segment_traffic_lights(lo, hi, buffer_m),
            "raw_traffic_calming": raw_calming,
            "traffic_calming": calming_to_int(raw_calming)
        })

    return pd.DataFrame(segments)


# === 主程序：批量处理 EV / PHEV ===
if __name__ == "__main__":
    traffic_lights_gdf = gpd.read_file("./osm_speed_features/traffic_lights.geojson")
    traffic_calming_gdf = gpd.read_file("./osm_speed_features/traffic_calming.geojson")

    # 过滤空/无效几何
    traffic_lights_gdf = traffic_lights_gdf[traffic_lights_gdf.geometry.notnull()]
    traffic_lights_gdf = traffic_lights_gdf[traffic_lights_gdf.geometry.type == "Point"]

    traffic_calming_gdf = traffic_calming_gdf[traffic_calming_gdf.geometry.notnull()]
    traffic_calming_gdf = traffic_calming_gdf[traffic_calming_gdf.geometry.type.isin(["Point", "LineString"])]

    print(traffic_calming_gdf["traffic_calming"].value_counts())
    # === ID 列表 ===
    EVs_ids = [10, 455, 541]
    PHEVs_ids = [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457, 492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]

    categories = {
        "EV": EVs_ids,
        "PHEV": PHEVs_ids
    }

    # === 批量执行 ===
    for category, ids in categories.items():
        for vehid in ids:
            input_dir = f"../data/filtered_vehID_eVED/{category}/{vehid}/"
            output_dir = f"../data/segmented_50m_eVED_trafficcalming/{category}/{vehid}/"
            os.makedirs(output_dir, exist_ok=True)
            all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]

            with tqdm(total=len(all_files), desc=f"Processing {category} {vehid}", unit="file") as pbar:
                for f in all_files:
                    seg_df = process_file(f, traffic_lights_gdf, traffic_calming_gdf, buffer_m=25)
                    if seg_df[["mean_energy_consumption", "mean_vehicle_speed"]].isna().any().any():
                        print(f"Skipping file {os.path.basename(f)} due to NaN in energy or speed")
                    else:
                        out_file = os.path.join(output_dir, os.path.basename(f))
                        seg_df.to_csv(out_file, index=False)
                    pbar.update(1)
