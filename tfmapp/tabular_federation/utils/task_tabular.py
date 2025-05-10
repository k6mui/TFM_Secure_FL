import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer

from flwr.common.typing import UserConfig

torch.manual_seed(42)
class MLP(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_output)
        )
    def forward(self, x): return self.model(x)

def train(model, train_loader: DataLoader, val_loader: DataLoader, epochs: int, batch_size: int, device, optimizer, privacy_engine, target_delta, use_dp):
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Para guardar métricas
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "train_prec": [],
        "val_prec": [],
        "train_conf_matrix": None,
        "val_conf_matrix": None,
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds_train = []
        all_labels_train = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs).squeeze()
            preds = probs > 0.5

            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(y_batch.cpu().numpy())

        # Calcular métricas de entrenamiento
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels_train, all_preds_train)
        epoch_recall = recall_score(all_labels_train, all_preds_train, average='macro', zero_division=0)
        epoch_f1 = f1_score(all_labels_train, all_preds_train, average='macro', zero_division=0)
        epoch_prec = precision_score(all_labels_train, all_preds_train, average='macro', zero_division=0)
        conf_matrix_train = confusion_matrix(all_labels_train, all_preds_train)

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["train_recall"].append(epoch_recall)
        history["train_f1"].append(epoch_f1)
        history["train_prec"].append(epoch_prec)

        # Evaluación en validación
        model.eval()
        val_loss = 0.0
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val.unsqueeze(1))  
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(y_val.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = accuracy_score(all_labels_val, all_preds_val)
        val_epoch_recall = recall_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        val_epoch_f1 = f1_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        val_epoch_prec = precision_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        conf_matrix_val = confusion_matrix(all_labels_val, all_preds_val)

        history["val_loss"].append(val_epoch_loss)
        history["val_accuracy"].append(val_epoch_acc)
        history["val_recall"].append(val_epoch_recall)
        history["val_f1"].append(val_epoch_f1)
        history["val_prec"].append(val_epoch_prec)

        # print(f"Epoch [{epoch+1}/{epochs}]")
        # print(f"Train -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Recall: {epoch_recall:.4f} | F1: {epoch_f1:.4f} | Prec: {epoch_prec}")
        # print(f"Val   -> Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f} | Recall: {val_epoch_recall:.4f} | F1: {val_epoch_f1:.4f} | Prec: {val_epoch_prec}")
        # print("-" * 60)

    # Guardar últimas matrices de confusión (podrías guardar por cada epoch si quieres)
    history["train_conf_matrix"] = conf_matrix_train
    history["val_conf_matrix"] = conf_matrix_val

    epsilon = privacy_engine.get_epsilon(delta=target_delta) if use_dp else None

    return history, epsilon

def test(model, x_test, y_test, device):
    """Model's Test evaluation."""

    model.eval()
    model.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        test_logits = model(x_test)
        test_loss = criterion(test_logits, y_test.unsqueeze(1)).item()
        test_probs = torch.sigmoid(test_logits)
        test_preds = test_probs > 0.5

    # ✅ Conversión a NumPy en CPU antes de pasar a sklearn
    y_true_np = y_test.detach().cpu().numpy()
    y_pred_np = test_preds.detach().cpu().numpy()

    accuracy = accuracy_score(y_true_np, y_pred_np)
    recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true_np, y_pred_np)

    return test_loss, accuracy, recall, f1, precision, conf_matrix


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def load_data(csv_path: str, batch_size: int, device: str):
    """Carga CSV, divide en train/val/test, y devuelve DataLoaders, input_size y num_classes."""
    # Device detection
    DEVICE = device

    # Load dataset
    df = pd.read_csv(csv_path)
    labels = df["smoking"]
    samples = df.drop(columns=["smoking", "ID"])

    # Scaling selected columns
    cols_to_scale = [c for c in samples.columns if (samples[c].nunique() > 50) or c in ['height(cm)', 'weight(kg)']]
    samples = TableVectorizer(specific_transformers=[(StandardScaler(), cols_to_scale)]).fit_transform(samples)

    # Data split
    x_train, x_eval, y_train, y_eval = train_test_split(samples, labels, test_size=0.2, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_eval, y_eval, test_size=0.2, stratify=y_eval)

    # Convert to tensors
    x_test = torch.tensor(x_test.values, dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(y_test.values, dtype=torch.float32, device=DEVICE)

    train_data = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32, device=DEVICE),
                           torch.tensor(y_train.values, dtype=torch.float32, device=DEVICE))
    val_data = TensorDataset(torch.tensor(x_val.values, dtype=torch.float32, device=DEVICE),
                         torch.tensor(y_val.values, dtype=torch.float32, device=DEVICE))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Extraer input_size y num_classes
    input_size = train_data[0][0].shape[0]
    num_classes = 1

    return train_loader, val_loader, x_test, y_test, input_size, num_classes


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""

    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")

    # Save path includes custom subdirectory path
    save_path = Path.cwd() / "tfmapp/tabular_federation/outputs" / run_dir
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir