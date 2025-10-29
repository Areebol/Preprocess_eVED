该仓库用于从eVED数据集中提取EV和PHEV车辆行程数据，构建50米为单位的cycle数据集

step 1 下载eVED数据集以及VED的静态数据

step 2 提取EV和PHEV的车辆id

step 3 提取某一车辆id的所有行程数据，每个行程数据用一个csv文件保存

step 4 构建距离特征，为后续划分50米 segment做铺垫
