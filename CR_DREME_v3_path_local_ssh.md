【svr_4d_dreme——CR_DREME_v3】

## codex运行环境为本地
### 本地设置
本地环境：knesvr_torch，含有torch,只有cpu,只能进行smoke和无需gpu的代码检查工作
本地代码路径：/home/universe/SVR/code/CR_DREME_v3/
本地原始dicom数据路径：/home/universe/SVR/data/DYL0709/DYL_20260709_dicom/

### 服务器设置
服务器环境：cr_dreme
服务器代码路径：/data/dengyz/code/CR_DREME_v3/
服务器原始dicom数据路径：/data/dengyz/dataset/DYL0709/DYL_20260709_dicom/

### github仓库
git remote set-url origin git@github.com:universe26dyz/CR_DREME_v3.git

### 注意
每次在服务器上跑通流程生成结果时，需要同时保存这次实验使用的代码的commit sha，以便后续复现。
