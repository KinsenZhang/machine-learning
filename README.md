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

