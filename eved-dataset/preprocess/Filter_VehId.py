import os
import pandas as pd

# === Configuration ===
input_folder = "./data/eVED"  # Folder containing input CSV files

EVs_ids = [10, 455, 541]
PHEVs_ids = [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457, 492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]

categories = {
    "EV": EVs_ids,
    "PHEV": PHEVs_ids
}

def filter_and_split_by_trip(veh_id_to_keep, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    trip_data = {}

    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(".csv"):
            continue

        file_path = os.path.join(input_folder, file_name)
        print(f"Processing {file_name} for VehId={veh_id_to_keep} ...")

        df = pd.read_csv(file_path)

        if "VehId" not in df.columns or "Trip" not in df.columns:
            print(f"Skipped {file_name}: missing VehId or Trip column.")
            continue

        filtered_df = df[df["VehId"] == veh_id_to_keep]

        for trip_id, group in filtered_df.groupby("Trip"):
            trip_data.setdefault(trip_id, []).append(group)

    for trip_id, dfs in trip_data.items():
        combined = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(output_folder, f"{int(trip_id)}.csv")
        combined.to_csv(output_path, index=False)
        print(f"Saved {output_path} with {len(combined)} rows.")

    if not trip_data:
        print(f"No rows found for VehId={veh_id_to_keep}.")

    print(f"Done for VehId={veh_id_to_keep}.\n")


# === Main loop for both EVs and PHEVs ===
for category, veh_ids in categories.items():
    for veh_id in veh_ids:
        output_folder = f"./data/filtered_vehID_eVED/{category}/{veh_id}"
        filter_and_split_by_trip(veh_id, output_folder)
