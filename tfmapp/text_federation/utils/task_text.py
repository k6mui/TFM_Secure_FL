from typing import Any
import torch
from collections import OrderedDict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torch
import pandas as pd
import numpy as np  

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding

def clean(text):
    return str(text).strip().replace("\n", " ").replace("\t", " ").replace("  ", " ").lower()

def load_data(partition_id: int, num_partitions: int, model_name: str) -> tuple[DataLoader, DataLoader, int, torch.Tensor]:
    """Load and preprocess federated MedMCQA data"""
    
    # Federated dataset loading
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="openlifescienceai/medmcqa",
        partitioners={"train": partitioner}
    )
    partition = fds.load_partition(partition_id)

    # Convert to DataFrame for preprocessing
    df = pd.DataFrame(partition)

    # Input: question + options
    df["input_text"] = df.apply(
        lambda x: f"question: {clean(x['question'])} A: {clean(x['opa'])} B: {clean(x['opb'])} C: {clean(x['opc'])} D: {clean(x['opd'])}",
        axis=1,
    )

    # Encode subject_name as label
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["subject_name"])
    num_labels = len(le.classes_)

    # Undersample max 3000 per class
    df = df.groupby("label").apply(lambda x: x.sample(n=min(3000, len(x)), random_state=42)).reset_index(drop=True)

    # Stratified split
    train_df, val_df = train_test_split(df, stratify=df["label"], test_size=0.1, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["label"]),
        y=df["label"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # PyTorch datasets
    train_dataset = TextDataset(train_df["input_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = TextDataset(val_df["input_text"].tolist(), val_df["label"].tolist(), tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    return train_loader, val_loader, num_labels, class_weights_tensor


def train(model, train_loader, val_loader, epochs, device, class_weights):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs * len(train_loader),
    )
    model.to(device)

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": [],
        "train_precision": [], "val_precision": [],
        "train_conf_matrix": None, "val_conf_matrix": None,
    }

    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        train_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(acc)
        history["train_precision"].append(prec)
        history["train_recall"].append(rec)
        history["train_f1"].append(f1)
        history["train_conf_matrix"] = conf_matrix

        # ---------- VALIDACIÓN ----------
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs["loss"]
                val_loss += loss.item()

                preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_conf = confusion_matrix(all_labels, all_preds)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["val_conf_matrix"] = val_conf

    return history

def test(model, test_loader, device):
    model.eval()
    model.to(device)

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs["logits"], batch["labels"])
            total_loss += loss.item()

            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, prec, rec, f1, conf_matrix


def get_model(num_labels: int, class_weights=None):
    base = AutoModel.from_pretrained("nlpie/bio-tinybert")
    class Model(nn.Module):
        def __init__(self): 
            super().__init__()
            self.base = base
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Linear(base.config.hidden_size, num_labels)
            self.weights = class_weights

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
            pooled = self.base(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0]
            logits = self.head(self.dropout(pooled))
            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.weights)
                loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}
    return Model()

# --------------------------
# Model weight utilities
# --------------------------
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
