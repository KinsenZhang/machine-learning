# 任务四：眼底血管分割

## 方法

- 模型：ImageNet 预训练 ResNet34 编码器 + U-Net 解码器
- 输入：512×512×3
- 损失：BCEWithLogits(pos_weight=2) + soft Dice
- 训练：5 折，batch=6，最多 40 epoch，验证 Dice 早停 patience=10
- 优化器：AdamW，主干 lr=3e-4×0.25，解码器 lr=3e-4
- 推理：4 路翻转 TTA，5 折概率平均，OOF 阈值搜索，连通域去小噪点
- 提交：512×512 mask 的 RLE，`Id,Predicted`

## 运行

挂载 2026task-4 数据后运行：

```bash
python task4_vessel_segmentation.py
```

或使用 Notebook：

```text
task4_vessel_segmentation.ipynb
```

默认路径：

```text
/kaggle/input/competitions/2026task-4/segmentation
```

输出：

```text
/kaggle/working/submission.csv
```

代码同时把二值 mask 保存到 `/kaggle/working/image/`，供官方
`segmentation_to_csv.py` 使用；但脚本已经直接生成 RLE，不需要再运行官方脚本。

## 标签方向

任务文字写“血管=0，其他=255”，而官方脚本用 `image > 0` 当前景。代码会把血管保存为
255、背景为 0 后再编码 RLE，避免把背景编码进去。报告 4.1 节有说明。
