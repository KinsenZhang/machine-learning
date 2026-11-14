# 任务三：黄斑中心凹定位

## 方法

- 骨干：ImageNet 预训练 ResNet34 + FPN
- 输出：1/4 分辨率中心凹热图（192×192）+ 图像级可见性 logit
- 训练：5 折，letterbox 768×768，batch=8，最多 40 epoch
- 损失：focal heatmap loss + 0.2 × BCE
- 优化器：AdamW，主干 lr=2.5e-4×0.25，头 lr=2.5e-4
- 推理：水平翻转 TTA，5 折热图概率平均，soft-argmax 解码
- 提交：每张图 X/Y 两行，`ImageID,value`

## 运行

挂载 2026task3 数据后运行：

```bash
python task3_fovea_localization.py
```

或使用 Notebook：

```text
task3_fovea_localization.ipynb
```

默认路径：

```text
/kaggle/input/2026task3/detection
```

如果实际路径是 `/kaggle/input/competitions/2026task3/detection`，修改 `CFG["data_root"]`。

输出：

```text
/kaggle/working/submission.csv
```
