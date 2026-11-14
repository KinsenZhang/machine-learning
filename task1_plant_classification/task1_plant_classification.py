"""Task 1: plant seedling classification with a pretrained ResNet18.

This file is a cleaned script version of the Kaggle notebook used for the
course project.  It keeps the original 8:2 random split, augmentation and
early-stopping logic.  The comment near ``val_dataset.dataset.transform``
records a known issue that is also discussed in the report.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

TRAIN_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train"
TEST_DIR = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test"
WORK_DIR = "/kaggle/working"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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


class EarlyStopping:
    """Save the best validation-F1 checkpoint and stop after ``patience``."""

    def __init__(self, patience=4, path=os.path.join(WORK_DIR, "best_model.pth")):
        self.patience = patience
        self.path = path
        self.best_val_f1 = 0.0
        self.counter = 0

    def __call__(self, val_f1, model):
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"保存最优模型，验证集F1: {val_f1:.4f}")
        else:
            self.counter += 1
            print(f"验证F1无提升，早停计数 {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # NOTE: in the submitted version this assignment changes the shared
    # ImageFolder transform, so the training subset also loses augmentation.
    # The report discusses this issue explicitly.
    val_dataset.dataset.transform = val_test_transforms

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    test_dataset = TestDataset(TEST_DIR, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("训练集图片数量:", len(train_dataset))
    print("验证集图片数量:", len(val_dataset))
    print("测试集图片数量:", len(test_dataset))
    print("类别:", full_train_dataset.classes)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 5),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    early_stop = EarlyStopping(patience=4)
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average="micro")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="micro")
        print(f"第 {epoch + 1} 轮 | Train Loss:{epoch_loss:.4f} | "
              f"Train F1:{epoch_f1:.4f} | Val F1:{val_f1:.4f}")

        if early_stop(val_f1, model):
            print("触发早停，训练提前终止")
            break

    model.load_state_dict(torch.load(os.path.join(WORK_DIR, "best_model.pth")))
    model.eval()

    prediction_list, filename_list = [], []
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Predicting Test Set"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            prediction_list.extend(preds.cpu().numpy())
            filename_list.extend(filenames)

    idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
    pred_class_names = [idx_to_class[p] for p in prediction_list]

    submit_df = pd.DataFrame({
        "ID": filename_list,
        "Category": pred_class_names,
    })
    output_path = os.path.join(WORK_DIR, "submission.csv")
    submit_df.to_csv(output_path, index=False)

    print("预测完成:", output_path)
    print("shape:", submit_df.shape)
    print("columns:", list(submit_df.columns))
    print("category counts:")
    print(submit_df["Category"].value_counts())


if __name__ == "__main__":
    main()
