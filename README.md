该仓库用于从eVED数据集中提取EV和PHEV车辆行程数据，构建50米为单位的cycle数据集

step 1 下载eVED数据集以及VED的静态数据
1. [eVED](https://github.com/zhangsl2013/eVED)
2. [VED](https://github.com/gsoh/VED)

step 2 提取EV和PHEV的车辆id
`preprocess/Extract_VehId_From_Static.py`

step 3 提取某一车辆id的所有行程数据，每个行程数据用一个csv文件保存
`preprocess/Filter_VehId.py`

step 4 构建距离特征，为后续划分50米 segment做铺垫
`preprocess/Split_2_Segments.py`

数据分析
step 1 分析不同VehId的行程数量，行程长度等

