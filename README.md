# Machine Learning Course Design

本仓库为机器学习课程设计的四个任务代码与报告整理版。

- 姓名：张子腾
- 学号：20245630
- 班级：人智 3 班
- 指导教师：吴宏林
- 报告：[`report/机器学习课程设计报告-张子腾-20245630.docx`](report/机器学习课程设计报告-张子腾-20245630.docx)

## 目录结构

```text
machine-learning/
├─ README.md
├─ requirements.txt
├─ task1_plant_classification/
│  ├─ README.md
│  └─ task1_plant_classification.py
├─ task2_emotion_vit/
│  ├─ README.md
│  └─ task2_emotion_vit.py
├─ task3_fovea_localization/
│  ├─ README.md
│  ├─ task3_fovea_localization.py
│  └─ task3_fovea_localization.ipynb
├─ task4_vessel_segmentation/
│  ├─ README.md
│  ├─ task4_vessel_segmentation.py
│  └─ task4_vessel_segmentation.ipynb
├─ report/
│  └─ 机器学习课程设计报告-张子腾-20245630.docx
└─ data/
   └─ README.md
```

## 四个任务

| 任务 | 内容 | 主要方法 | 评价指标 |
|---|---|---|---|
| 任务一 | 植物幼苗图像分类 | ResNet18 迁移学习，5 类 | micro-F1 |
| 任务二 | 图像情感分类 | ViT-B/16，CutMix，5 折集成，6 类 | micro-F1 |
| 任务三 | 黄斑中心凹定位 | ResNet34-FPN 热图 + 可见性分类 | RMSE |
| 任务四 | 眼底血管分割 | ResNet34 U-Net，BCE + Dice，RLE | Dice |

## 运行环境

- Python 3.10+
- PyTorch / torchvision
- numpy、pandas、opencv-python、scikit-learn、Pillow、tqdm

```bash
pip install -r requirements.txt
```

所有训练脚本按 Kaggle Notebook 的路径编写，数据需要挂载到：

```text
/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2
/kaggle/input/competitions/2026task2/fer_data/fer_data
/kaggle/input/2026task3/detection
/kaggle/input/competitions/2026task-4/segmentation
```

如果本机路径不同，请修改各脚本顶部的 `TRAIN_DIR` / `TEST_DIR` / `CFG["data_root"]`。

## 说明

- 代码按课程提交时的 Notebook 内容整理，保留原始训练与推理流程。
- 报告中的结果分析以可复核的提交文件、版本一致率和执行记录为依据；Kaggle 保存版本未保留逐 epoch 的标准输出，因此报告没有填写无法复核的 F1/RMSE/Dice 分数。
- 数据文件体积较大，未上传到仓库；请按 `data/README.md` 从 Kaggle 比赛页面挂载。
- 报告文件位于 `report/`，可直接下载后提交。
