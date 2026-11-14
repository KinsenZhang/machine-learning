# 任务一：植物幼苗图像分类

## 方法

- 骨干：ImageNet 预训练 ResNet18
- 输入：224×224×3
- 分类头：Dropout(0.3) + Linear(512, 5)
- 损失：CrossEntropyLoss
- 优化器：AdamW，lr=1e-4，weight_decay=1e-4
- 训练：batch=32，最多 20 epoch，验证 micro-F1 早停 patience=4
- 提交：`ID,Category`

## 运行

在 Kaggle Notebook 中挂载 2026task1 数据，然后运行：

```bash
python task1_plant_classification.py
```

默认路径：

```text
/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train
/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test
```

输出：

```text
/kaggle/working/submission.csv
```

## 已知实现问题

`val_dataset.dataset.transform = val_test_transforms` 会修改共享的 ImageFolder 实例，
训练子集也会使用验证变换，导致数据增强未按预期生效。报告 1.4 节对此有说明；
下一版应分别复制数据集对象或为 Subset 包装独立 transform。
