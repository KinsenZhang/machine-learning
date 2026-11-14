# 数据说明

本仓库不包含比赛图像数据，原因是数据体积较大且受比赛 Rules 约束。运行时请从 Kaggle 比赛页面挂载数据。

| 任务 | Kaggle 路径 |
|---|---|
| 任务一 | `/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2` |
| 任务二 | `/kaggle/input/competitions/2026task2/fer_data/fer_data` |
| 任务三 | `/kaggle/input/2026task3/detection` |
| 任务四 | `/kaggle/input/competitions/2026task-4/segmentation` |

如果实际挂载路径不同，修改对应脚本顶部的路径变量即可。

任务三需要：

```text
train/
  *.jpg
test/
  *.jpg
fovea_localization_train_GT.csv
sample_submission.csv
```

任务四需要：

```text
train/image/*.jpg
train/label/*.jpg
test/image/*.jpg
sample_submission.csv
```
