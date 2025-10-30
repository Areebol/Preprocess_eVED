import pandas as pd

# df = pd.read_csv("data/filtered_vehID_eVED/EV/10/1561.csv")
df = pd.read_csv("data/eVED/eVED_171101_week.csv")
print(df.keys())
keys = ['DayNum', 'VehId', 'Trip', 'Timestamp(ms)', 'Latitude[deg]',
       'Longitude[deg]', 'Vehicle Speed[km/h]', 'MAF[g/sec]',
       'Engine RPM[RPM]', 'Absolute Load[%]', 'OAT[DegC]', 'Fuel Rate[L/hr]',
       'Air Conditioning Power[kW]', 'Air Conditioning Power[Watts]',
       'Heater Power[Watts]', 'HV Battery Current[A]', 'HV Battery SOC[%]',
       'HV Battery Voltage[V]', 'Short Term Fuel Trim Bank 1[%]',
       'Short Term Fuel Trim Bank 2[%]', 'Long Term Fuel Trim Bank 1[%]',
       'Long Term Fuel Trim Bank 2[%]', 'Elevation Raw[m]',
       'Elevation Smoothed[m]', 'Gradient', 'Energy_Consumption',
       'Matchted Latitude[deg]', 'Matched Longitude[deg]', 'Match Type',
       'Class of Speed Limit', 'Speed Limit[km/h]',
       'Speed Limit with Direction[km/h]', 'Intersection', 'Bus Stops',
       'Focus Points']

# key = "Vehicle Speed[km/h]"
# key = "Focus Points" \in \mathbb{R}^10
# key = "Bus Stops"
# key = "Intersection"
# key = "Speed Limit with Direction[km/h]"
# key = "Speed Limit[km/h]"
key = "Class of Speed Limit"
# print("unique:", df[key].nunique())
print("nan:", df[key].isna().sum())
print("10 uniques:", df[key].unique())
print("\nfrequency:\n", df[key].value_counts().head(10))
