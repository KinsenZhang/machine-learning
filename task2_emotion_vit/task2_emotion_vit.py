"""Task 2: facial emotion classification with ViT-B/16 and 5-fold ensemble.

The script follows the Kaggle notebook used for the course project:
- FER data, 6 classes, label remapping {0:0,1:1,2:2,3:5,4:3,5:4}
- ImageNet-pretrained ViT-B/16, dropout 0.4 + linear head
- CutMix, label smoothing, AdamW, cosine schedule, 5-fold StratifiedKFold
- 5-fold softmax probability averaging for the submission
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "/kaggle/input/competitions/2026task2/fer_data/fer_data"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
WORK_DIR = "/kaggle/working"

# Original FER labels -> submission labels.
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4}

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def rand_bbox(size, lam):
    """Random CutMix rectangle for a [B, C, H, W] tensor."""
    width = size[2]
    height = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix and return mixed inputs, two labels and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image = Image.open(os.path.join(self.test_dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, name


def build_model():
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 6),
    )
    return model.to(device)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("设备:", device)

    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    all_targets = full_dataset.targets
    print("总训练样本:", len(full_dataset))

    test_dataset = TestDataset(TEST_DIR, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    print("测试样本:", len(test_dataset))

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_best_f1 = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(all_targets)), all_targets)):
        print(f"\n{'=' * 50}")
        print(f"开始第 {fold + 1}/{n_splits} 折")
        print("=" * 50)

        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)

        # NOTE: same issue as task 1: the shared ImageFolder transform is
        # replaced, so the training subset also loses augmentation.  The
        # report analyses this behavior explicitly.
        val_sub.dataset.transform = val_test_transform

        train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=2)

        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 6),
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.12)
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=12, eta_min=1e-6
        )

        best_f1 = 0.0
        patience_counter = 0
        max_epochs = 12

        for epoch in range(max_epochs):
            model.train()
            train_preds, train_labels_list = [], []
            running_loss = 0.0

            for inputs, labels in tqdm(
                    train_loader, desc=f"Fold{fold + 1} Epoch{epoch + 1} Train"):
                inputs = inputs.to(device)
                labels = torch.tensor(
                    [LABEL_REMAP[x.item()] for x in labels], dtype=torch.long
                ).to(device)

                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = (lam * criterion(outputs, targets_a)
                        + (1 - lam) * criterion(outputs, targets_b))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())

            train_f1 = f1_score(train_labels_list, train_preds, average="micro")
            scheduler.step()

            model.eval()
            val_preds, val_labels_list = [], []
            with torch.no_grad():
                for inputs, labels in tqdm(
                        val_loader, desc=f"Fold{fold + 1} Epoch{epoch + 1} Val"):
                    inputs = inputs.to(device)
                    labels = torch.tensor(
                        [LABEL_REMAP[x.item()] for x in labels], dtype=torch.long
                    ).to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            val_f1 = f1_score(val_labels_list, val_preds, average="micro")
            print(f"Epoch {epoch + 1} | Loss:{running_loss / len(train_loader):.4f} | "
                  f"Train F1:{train_f1:.4f} | Val F1:{val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(WORK_DIR, f"fold{fold + 1}_best.pth"),
                )
                print(f"保存本折最优模型，Val F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"本折早停，最优F1: {best_f1:.4f}")
                    break

        fold_best_f1.append(best_f1)

    print(f"5折验证F1: {fold_best_f1}")
    print(f"平均验证F1: {np.mean(fold_best_f1):.4f}")

    all_probs = []
    filenames_list = []
    for fold_idx in range(1, 6):
        model = build_model()
        model.load_state_dict(
            torch.load(os.path.join(WORK_DIR, f"fold{fold_idx}_best.pth"))
        )
        model.eval()

        fold_probs = []
        fold_filenames = []
        with torch.no_grad():
            for images, fnames in tqdm(
                    test_loader, desc=f"预测 Fold {fold_idx}"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())
                if fold_idx == 1:
                    fold_filenames.extend(fnames)

        all_probs.append(np.concatenate(fold_probs, axis=0))
        if fold_idx == 1:
            filenames_list = fold_filenames

    avg_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)

    submit_df = pd.DataFrame({
        "ID": filenames_list,
        "Emotion": final_preds,
    })
    output_path = os.path.join(WORK_DIR, "submission_vit_5fold.csv")
    submit_df.to_csv(output_path, index=False)

    print("5折融合预测完成:", output_path)
    print("shape:", submit_df.shape)
    print("columns:", list(submit_df.columns))
    print("prediction counts:")
    print(submit_df["Emotion"].value_counts().sort_index())


if __name__ == "__main__":
    main()
