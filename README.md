This repository is used to extract EV and PHEV vehicle trip data from the eVED dataset, constructing a cycle dataset segmented in 50-meter units.

Step 1: Download the eVED dataset and VED static data  
1. [eVED](https://github.com/zhangsl2013/eVED)  
2. [VED](https://github.com/gsoh/VED)  

Step 2: Extract vehicle IDs for EVs and PHEVs  
`preprocess/Extract_VehId_From_Static.py`

Step 3: Extract all trip data for a specific vehicle ID, with each trip saved as a CSV file  
`preprocess/Filter_VehId.py`

Step 4: Construct distance features to prepare for segmenting into 50-meter units  
`preprocess/Split_2_Segments.py`

Step 5: Extract OSM features
`preprocess/Extract_LL_From_eVED.py` extract Lalitude/Longitude Range in eVED
# ==== Dataset Latitude/Longitude Range ====
# Latitude range: 42.220268 ~ 42.325853
# Longitude range: -83.804839 ~ -83.673437

`preprocess/Extract_From_OSM.py` use OSM apis to extract road conditions and traffic lights etc.

Step 6: Construct distance features with OSM features
`preprocess/Split_2_Segments_With_TrafficLights.py`
`preprocess/Split_2_Segments_With_TRafficCalming.py`

Versions:  
- `segmented_50m_eVED_trafficlight` – Adds traffic light identifiers to the eVED base.  
- `segmented_50m_eVED_trafficcalming` – Adds traffic calming category identifiers to the eVED base.  