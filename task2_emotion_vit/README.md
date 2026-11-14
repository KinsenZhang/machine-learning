# 任务二：图像情感分类（ViT-B/16）

## 方法

- 数据：FER，6 类；标签重映射 `{0:0,1:1,2:2,3:5,4:3,5:4}`
- 骨干：ImageNet 预训练 ViT-B/16，patch 16，12 层，hidden 768
- 分类头：Dropout(0.4) + Linear(768, 6)
- 训练：5 折 StratifiedKFold，batch=64，最多 12 epoch，早停 patience=3
- 损失：CrossEntropy(label_smoothing=0.12)
- 优化器：AdamW，lr=3e-5，weight_decay=1e-4，CosineAnnealingLR
- 增强：翻转、旋转、颜色抖动、CutMix(alpha=1.0)
- 推理：5 折 softmax 概率平均

## 运行

挂载 2026task2 数据后运行：

```bash
python task2_emotion_vit.py
```

默认路径：

```text
/kaggle/input/competitions/2026task2/fer_data/fer_data/train
/kaggle/input/competitions/2026task2/fer_data/fer_data/test
```

输出：

```text
/kaggle/working/submission_vit_5fold.csv
```

## 已知实现问题

`val_sub.dataset.transform = val_test_transform` 会同时改变训练子集的 transform；
训练阶段的翻转、旋转、颜色抖动实际上没有生效，只有 CutMix 仍然作用。
报告 2.4 节对此有说明。下一版应为每折分别构建独立的数据集/transform。
