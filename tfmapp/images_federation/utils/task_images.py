import os
import json
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

# --------------------------
# CNN Model (ResNet18)
# --------------------------
def get_model(num_classes: int = 3):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # freeze base layers
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )
    return model

# --------------------------
# Data Transforms
# --------------------------
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.ColorJitter(brightness=(0.1, 0.7)),
    T.RandomAffine(degrees=0, translate=(0.5, 0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

valid_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# --------------------------
# Load image data
# --------------------------
def load_image_data(dataset_path: str, batch_size: int):
    """
    Load image data from a single folder with class subfolders.
    Split into train/val/test (72/08/20) and return corresponding DataLoaders.

    Args:
        dataset_path (str): Path to a client folder with class subfolders.
        batch_size (int): Batch size.

    Returns:
        train_loader, val_loader, test_loader, num_classes, class_names
    """
    full_dataset = ImageFolder(root=dataset_path, transform=None)
    num_samples = len(full_dataset)

    train_size = int(0.72 * num_samples)
    val_size   = int(0.08 * num_samples)
    test_size  = num_samples - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms to each split
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = valid_transform

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    num_classes = len(class_names)

    return train_loader, val_loader, test_loader, num_classes, class_names

# --------------------------
# Training
# --------------------------
def train(model, train_loader, val_loader, epochs, device, optimizer):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_recall": [], "val_recall": [],
        "train_precision": [], "val_precision": [],
        "train_f1": [], "val_f1": [],
        "train_conf_matrix": None,
        "val_conf_matrix": None,
    }

    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs}")
        print("-" * 60)
        model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(running_loss / len(train_loader))
        history["train_accuracy"].append(acc)
        history["train_recall"].append(recall)
        history["train_f1"].append(f1)
        history["train_precision"].append(precision)
        history["train_conf_matrix"] = conf_matrix

        print(f"✅ Train | Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"Confusion Matrix (Train):\n{conf_matrix}")

        # ---------- VALIDATION ----------
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        epoch_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_accuracy"].append(acc)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)
        history["val_precision"].append(precision)
        history["val_conf_matrix"] = conf_matrix

        print(f"🧪 Val   | Loss: {epoch_val_loss:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"Confusion Matrix (Val):\n{conf_matrix}")
        print("=" * 60)

    return history

# --------------------------
# Evaluation
# --------------------------
def test(model, dataloader, device):
    model.eval()
    model.to(device)

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, recall, f1, precision, conf_matrix

# --------------------------
# Model weight utilities
# --------------------------
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
