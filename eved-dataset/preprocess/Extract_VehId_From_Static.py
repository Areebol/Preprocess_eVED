import pandas as pd

# === Configuration ===
input_file = "./data/VED_Static_Data_PHEV&EV.xlsx"  # Path to your Excel file

# Read Excel file
df = pd.read_excel(input_file)

# Check that required columns exist
required_cols = {"EngineType", "VehId"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

# Filter by EngineType
ev_list = df.loc[df["EngineType"] == "EV", "VehId"].unique().tolist()
phev_list = df.loc[df["EngineType"] == "PHEV", "VehId"].unique().tolist()

print("EV VehIds:", ev_list) # EV VehIds: [10, 455, 541]
print("PHEV VehIds:", phev_list) # PHEV VehIds: [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457, 492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]



# # === Configuration === #TODO 添加另一个xlsx文件
# input_file = "./data/VED_Static_Data_PHEV&EV.xlsx"  # Path to your Excel file

# # Read Excel file
# df = pd.read_excel(input_file)


# # Check that required columns exist
# required_cols = {"EngineType", "VehId"}
# if not required_cols.issubset(df.columns):
#     raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

# # Filter by EngineType
# ev_list = df.loc[df["EngineType"] == "EV", "VehId"].unique().tolist()
# phev_list = df.loc[df["EngineType"] == "PHEV", "VehId"].unique().tolist()

# print("EV VehIds:", ev_list)
# print("PHEV VehIds:", phev_list)