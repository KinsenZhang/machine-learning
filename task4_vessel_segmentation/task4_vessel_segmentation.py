# %% [markdown]
# # 2026 Task 4 — 眼底血管分割（Vessel Segmentation）
#
# 输入：`train/image/1.jpg`、`train/label/1.jpg`、`test/image/1.jpg`。
# 输出：RLE 格式 `Id,Predicted`。
#
# 方案：ResNet34 预训练编码器 + U-Net 解码器；BCE + Dice；5 折；水平/垂直翻转 TTA；
# OOF 选 Dice 最优阈值；连通域去小噪点；保存 512×512 二值 mask，并直接生成 RLE。
#
# **标签像素约定**：任务文字写“血管=0，其他=255”，但官方 `segmentation_to_csv.py`
# 用的是 `image > 0` 作为前景。为了让 RLE 编码的是血管而不是背景，本代码会把
# **血管保存为 255、背景为 0** 后再编码。代码会自动检测训练 label 的哪一边是血管。

# %%
import os
import re
import math
import time
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    from torchvision.models import resnet34, ResNet34_Weights
except ImportError:
    from torchvision.models import resnet34
    ResNet34_Weights = None
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
cv2.setNumThreads(2)
IMREAD_FLAGS = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)

CFG = {
    "data_root": os.environ.get(
        "VESSEL_DATA_ROOT",
        "/kaggle/input/competitions/2026task-4/segmentation",
    ),
    "img_size": 512,
    "folds": 5,
    "epochs": 40,
    "batch_size": 6,
    "infer_batch": 4,
    "lr": 3e-4,
    "backbone_lr_mult": 0.25,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "patience": 10,
    "seed": 42,
    "num_workers": 4,
    "min_area": 8,
    "threshold": 0.5,
}
SEED = CFG["seed"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("output")
WORK_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR = WORK_DIR / "image"
MASK_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = WORK_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(SEED)
print("torch:", torch.__version__, "| device:", DEVICE, "| GPUs:", torch.cuda.device_count())


def find_root():
    candidates = [
        Path(CFG["data_root"]),
        Path("/kaggle/input/competitions/2026task-4/segmentation"),
        Path("/kaggle/input/2026task-4/segmentation"),
        Path("/kaggle/input/2026task-4"),
    ]
    for p in candidates:
        if (p / "train/image").is_dir() and (p / "test/image").is_dir():
            return p

    base = Path("/kaggle/input")
    print("CFG['data_root'] =", CFG["data_root"])
    print("/kaggle/input 顶层：")
    if base.exists():
        for p in sorted(base.iterdir())[:40]:
            print("  ", p)
    raise FileNotFoundError(
        "找不到数据。请先在右侧 Add Input → Competition Data 挂载 task-4；"
        "或从比赛页 Code → New Notebook 新建（自动挂数据）。"
    )


DATA_ROOT = find_root()
TRAIN_IMG = DATA_ROOT / "train" / "image"
TRAIN_LBL = DATA_ROOT / "train" / "label"
TEST_IMG = DATA_ROOT / "test" / "image"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
if not SAMPLE_SUB.exists():
    hits = list(DATA_ROOT.rglob("sample_submission.csv"))
    if hits:
        SAMPLE_SUB = hits[0]

print("DATA_ROOT =", DATA_ROOT)
for p in [TRAIN_IMG, TRAIN_LBL, TEST_IMG, SAMPLE_SUB]:
    print("  OK " if p.exists() else "  MISS ", p)

# %% [markdown]
# ## Cell 2：数据表、标签方向、RLE 工具

# %%
def canon_id(v):
    s = Path(str(v).strip()).stem
    return str(int(s)) if s.isdigit() else s.lower()


def index_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return {canon_id(p.stem): p for p in sorted(folder.iterdir())
            if p.suffix.lower() in exts}


train_imgs = index_images(TRAIN_IMG)
train_lbls = index_images(TRAIN_LBL)
test_imgs = index_images(TEST_IMG)

train_ids = sorted(set(train_imgs) & set(train_lbls), key=lambda x: int(x) if x.isdigit() else x)
test_ids = sorted(test_imgs, key=lambda x: int(x) if x.isdigit() else x)
print("train pairs:", len(train_ids), "| test images:", len(test_ids))

if len(train_ids) == 0:
    raise RuntimeError("train/image 和 train/label 没有匹配的文件。")

train_df = pd.DataFrame({
    "id": train_ids,
    "image": [str(train_imgs[k]) for k in train_ids],
    "label": [str(train_lbls[k]) for k in train_ids],
})
test_df = pd.DataFrame({
    "id": test_ids,
    "image": [str(test_imgs[k]) for k in test_ids],
})

# 自动判断 label 中哪一边是血管：
# 血管面积通常远小于 50%；如果 0 像素占多数，则血管是白色 255；否则血管是黑色 0。
zero_fracs = []
for p in train_df["label"].head(12):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is not None:
        zero_fracs.append(float((m < 128).mean()))
FG_IS_WHITE = bool(np.median(zero_fracs) > 0.5) if zero_fracs else True
print("label zero-fraction median:", round(float(np.median(zero_fracs)), 4) if zero_fracs else None)
print("FG_IS_WHITE =", FG_IS_WHITE, "（True: 血管=255；False: 血管=0）")
if zero_fracs:
    print("如果这个方向反了，手动改 FG_IS_WHITE = not FG_IS_WHITE 后重跑。")


def to_binary(mask):
    """把 label 转成 0/1，血管=1。"""
    if mask is None:
        return np.zeros((CFG["img_size"], CFG["img_size"]), dtype=np.float32)
    if mask.max() <= 1:
        return (mask > 0.5).astype(np.float32) if FG_IS_WHITE else (mask < 0.5).astype(np.float32)
    return (mask > 127).astype(np.float32) if FG_IS_WHITE else (mask < 128).astype(np.float32)


def rle_encode(mask):
    """mask: HxW 0/1。返回官方格式的 'start length ...'（1-based）。"""
    pixels = np.asarray(mask).reshape(-1).astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_score(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))


print("sample submission:", SAMPLE_SUB)
print(pd.read_csv(SAMPLE_SUB).head().to_string(index=False))

# %% [markdown]
# ## Cell 3：Dataset 和增强

# %%
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def color_augment(img):
    img = img.astype(np.float32)
    if random.random() < 0.8:
        img = img * random.uniform(0.85, 1.15) + random.uniform(-12, 12)
    if random.random() < 0.35:
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, random.uniform(0.8, 1.25))
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.4, 1.2))
    return np.clip(img, 0, 255).astype(np.uint8)


def augment(img, mask):
    S = CFG["img_size"]
    if random.random() < 0.5:
        img = img[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    if random.random() < 0.5:
        img = img[::-1, :].copy()
        mask = mask[::-1, :].copy()

    angle = random.uniform(-15, 15)
    scale = random.uniform(0.88, 1.12)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, scale)
    M[0, 2] += random.uniform(-0.05, 0.05) * S
    M[1, 2] += random.uniform(-0.05, 0.05) * S
    img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (S, S), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = color_augment(img)
    return img, mask.astype(np.float32)


class VesselDataset(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.S = CFG["img_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.image, IMREAD_FLAGS)
        if img is None:
            raise FileNotFoundError(r.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.S, self.S), interpolation=cv2.INTER_AREA)

        if "label" in self.df.columns and pd.notna(r.label) and str(r.label) != "":
            m = cv2.imread(str(r.label), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.S, self.S), interpolation=cv2.INTER_NEAREST)
            mask = to_binary(m)
        else:
            mask = np.zeros((self.S, self.S), dtype=np.float32)

        if self.train:
            img, mask = augment(img, mask)

        arr = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return {
            "img": torch.from_numpy(arr.transpose(2, 0, 1)),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "idx": torch.tensor(i, dtype=torch.long),
        }


def make_loader(df, train, batch_size, shuffle):
    return DataLoader(
        VesselDataset(df, train), batch_size=batch_size, shuffle=shuffle,
        num_workers=CFG["num_workers"], pin_memory=True,
        persistent_workers=CFG["num_workers"] > 0, drop_last=False,
    )

# %% [markdown]
# ## Cell 4：U-Net 模型和 Dice loss

# %%
def conv_bn(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(conv_bn(in_ch, out_ch), conv_bn(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_bn(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.up(x))


class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained and ResNet34_Weights is not None:
            try:
                weights = ResNet34_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            bb = resnet34(weights=weights)
        except TypeError:
            bb = resnet34(pretrained=weights is not None)
        except Exception as e:
            print("预训练权重加载失败，随机初始化：", e)
            bb = resnet34(weights=None)

        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # stride 2, 64
        self.pool = bb.maxpool
        self.enc1 = bb.layer1   # stride 4, 64
        self.enc2 = bb.layer2   # stride 8, 128
        self.enc3 = bb.layer3   # stride 16, 256
        self.enc4 = bb.layer4   # stride 32, 512

        self.center = DoubleConv(512, 512)
        self.up4 = Up(512, 256)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = Up(256, 128)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = Up(128, 64)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = Up(64, 32)
        self.dec1 = DoubleConv(96, 32)
        self.final = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.final.bias, -2.0)

    def forward(self, x):
        x0 = self.stem(x)       # 64, stride 2
        x1 = self.pool(x0)      # 64, stride 4
        e1 = self.enc1(x1)      # 64, stride 4
        e2 = self.enc2(e1)      # 128, stride 8
        e3 = self.enc3(e2)      # 256, stride 16
        e4 = self.enc4(e3)      # 512, stride 32
        c = self.center(e4)
        d4 = self.dec4(torch.cat([self.up4(c), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], 1))
        out = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        return self.final(out)


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1
    return 1 - (num / den).mean()

# %% [markdown]
# ## Cell 5：预测、Dice、训练一个 fold

# %%
@torch.no_grad()
def predict(model, loader, tta=True):
    model.eval()
    probs = []
    for batch in loader:
        imgs = batch["img"].to(DEVICE, non_blocking=True)
        if tta:
            x = torch.cat([imgs, torch.flip(imgs, dims=[3]),
                           torch.flip(imgs, dims=[2]),
                           torch.flip(imgs, dims=[2, 3])], 0)
            out = torch.sigmoid(model(x))
            B = imgs.size(0)
            p = (out[:B] + torch.flip(out[B:2*B], dims=[3]) +
                 torch.flip(out[2*B:3*B], dims=[2]) +
                 torch.flip(out[3*B:], dims=[2, 3])) / 4.0
        else:
            p = torch.sigmoid(model(imgs))
        probs.append(p[:, 0].detach().float().cpu().numpy())
    return np.concatenate(probs, 0)


def evaluate(model, loader, gt_masks, thr=0.5):
    probs = predict(model, loader, tta=False)
    scores = []
    for p, g in zip(probs, gt_masks):
        scores.append(dice_score(p > thr, g))
    return float(np.mean(scores)), probs


def train_fold(fold, tr_df, va_df):
    print(f"\n===== FOLD {fold} train={len(tr_df)} val={len(va_df)} =====")
    tr_loader = make_loader(tr_df, True, CFG["batch_size"], True)
    va_loader = make_loader(va_df, False, CFG["infer_batch"], False)
    S = CFG["img_size"]
    va_gt = []
    for p in va_df["label"]:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
        va_gt.append(to_binary(m))

    model = UNet(pretrained=True).to(DEVICE)
    bb = [p for n, p in model.named_parameters() if n.startswith(("stem", "pool", "enc"))]
    head = [p for n, p in model.named_parameters() if not n.startswith(("stem", "pool", "enc"))]
    opt = AdamW([
        {"params": bb, "lr": CFG["lr"] * CFG["backbone_lr_mult"]},
        {"params": head, "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])

    def lr_lambda(epoch):
        w = CFG["warmup_epochs"]
        if epoch < w:
            return (epoch + 1) / w
        progress = (epoch - w) / max(1, CFG["epochs"] - w)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best = -1
    best_state = None
    bad = 0

    for epoch in range(CFG["epochs"]):
        model.train()
        losses = []
        t0 = time.time()
        for batch in tr_loader:
            imgs = batch["img"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=torch.tensor(2.0, device=DEVICE)
                ) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        sched.step()

        dv, _ = evaluate(model, va_loader, va_gt, thr=0.5)
        print(f"epoch {epoch+1:02d}/{CFG['epochs']} loss={np.mean(losses):.4f} "
              f"val_dice={dv:.4f} ({time.time()-t0:.1f}s)")

        if dv > best + 1e-5:
            best = dv
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_DIR / f"fold{fold}.pt")
            bad = 0
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print("early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"FOLD {fold} best dice={best:.4f}")
    return model

# %% [markdown]
# ## Cell 6：5 折训练

# %%
n_splits = min(CFG["folds"], len(train_df))
if n_splits < 2:
    raise RuntimeError("训练图太少，至少需要 2 张。")
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
splits = list(kf.split(train_df))

models = []
for fold, (tr_idx, va_idx) in enumerate(splits):
    model = train_fold(
        fold,
        train_df.iloc[tr_idx].copy(),
        train_df.iloc[va_idx].copy(),
    )
    models.append(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# ## Cell 7：OOF 预测并选 Dice 最优阈值

# %%
oof_probs = []
oof_gt = []
oof_ids = []

for fold, (tr_idx, va_idx) in enumerate(splits):
    va_df = train_df.iloc[va_idx].copy().reset_index(drop=True)
    loader = make_loader(va_df, False, CFG["infer_batch"], False)
    probs = predict(models[fold], loader, tta=True)
    for p, path, img_id in zip(probs, va_df["label"], va_df["id"]):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (CFG["img_size"], CFG["img_size"]), interpolation=cv2.INTER_NEAREST)
        oof_probs.append(p.astype(np.float16))
        oof_gt.append(to_binary(m))
        oof_ids.append(img_id)

oof_probs = np.stack(oof_probs, 0)
oof_gt = np.stack(oof_gt, 0)
print("OOF shape:", oof_probs.shape)

best_thr = 0.5
best_dice = -1
for thr in np.arange(0.25, 0.76, 0.05):
    scores = [dice_score(p > thr, g) for p, g in zip(oof_probs, oof_gt)]
    d = float(np.mean(scores))
    print(f"thr={thr:.2f} dice={d:.4f}")
    if d > best_dice:
        best_dice = d
        best_thr = float(thr)

print("BEST threshold:", best_thr, "OOF Dice:", best_dice)

# %% [markdown]
# ## Cell 8：测试推理、保存 mask、生成 submission.csv

# %%
for old in MASK_DIR.glob("*.png"):
    old.unlink()

test_loader = make_loader(test_df, False, CFG["infer_batch"], False)
prob_sum = None
for fold, model in enumerate(models):
    p = predict(model, test_loader, tta=True)
    prob_sum = p if prob_sum is None else prob_sum + p
    print("test fold", fold, "done")
prob_mean = prob_sum / len(models)

# 连通域去小噪点
def postprocess(mask, min_area=20):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


pred_masks = {}
for i, r in test_df.iterrows():
    mask = (prob_mean[i] > best_thr).astype(np.uint8)
    mask = postprocess(mask, CFG["min_area"])
    pred_masks[r["id"]] = mask
    img_id = str(r["id"])
    cv2.imwrite(str(MASK_DIR / f"{img_id}.png"), (mask * 255).astype(np.uint8))

# 直接生成 RLE，不依赖 segmentation_to_csv.py 的 ./image 相对路径
sample = pd.read_csv(SAMPLE_SUB)
values = []
for raw_id in sample.iloc[:, 0].astype(str):
    key = canon_id(raw_id)
    mask = pred_masks.get(key)
    values.append("" if mask is None else rle_encode(mask))

sample.iloc[:, 1] = values
out_path = WORK_DIR / "submission.csv"
sample.to_csv(out_path, index=False)
print("saved:", out_path)
print(sample.head(10).to_string(index=False))
print("mask dir:", MASK_DIR, "| masks:", len(list(MASK_DIR.glob('*.png'))))

# 可选：官方脚本等价检查
# %cd /kaggle/working
# !python /kaggle/input/.../segmentation_to_csv.py
