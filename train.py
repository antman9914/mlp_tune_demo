"""
两层 MLP 训练脚本
用法：python train.py [--config config.yaml]

训练结果写入 logs/exp_TIMESTAMP.csv，格式：
epoch,train_loss,val_loss,train_acc,val_acc
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import TwoLayerMLP


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_data(cfg: dict):
    data_cfg = cfg["data"]
    X, y = make_classification(
        n_samples=data_cfg["n_samples"],
        n_features=data_cfg["n_features"],
        n_informative=max(2, data_cfg["n_features"] // 2),
        n_classes=data_cfg["n_classes"],
        random_state=data_cfg["random_seed"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=data_cfg["val_ratio"],
        random_state=data_cfg["random_seed"],
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(to_tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    return train_ds, val_ds, data_cfg["n_features"], data_cfg["n_classes"]


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


def train(cfg_path: str = "config.yaml"):
    cfg = load_config(cfg_path)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    torch.manual_seed(cfg["data"]["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, input_dim, n_classes = make_data(cfg)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = TwoLayerMLP(
        input_dim=input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=n_classes,
        dropout_rate=model_cfg["dropout_rate"],
        activation=model_cfg["activation"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # lr scheduler
    scheduler = None
    if train_cfg.get("lr_scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif train_cfg.get("lr_scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    # 日志文件
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/exp_{timestamp}.csv"
    summary_path = f"logs/exp_{timestamp}_summary.json"

    print(f"训练开始 | lr={train_cfg['learning_rate']} | "
          f"wd={train_cfg['weight_decay']} | "
          f"dropout={model_cfg['dropout_rate']}")
    print(f"日志写入: {log_path}")

    rows = []
    best_val_loss = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        if scheduler:
            scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        }
        rows.append(row)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # 写 CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)

    # 写 summary JSON
    summary = {
        "config": cfg,
        "log_file": log_path,
        "train_loss_final": rows[-1]["train_loss"],
        "val_loss_final": rows[-1]["val_loss"],
        "train_acc_final": rows[-1]["train_acc"],
        "val_acc_final": rows[-1]["val_acc"],
        "best_val_loss": round(best_val_loss, 6),
        "timestamp": timestamp,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n训练完成 | val_acc={rows[-1]['val_acc']:.3f} | val_loss={rows[-1]['val_loss']:.4f}")
    print(f"摘要写入: {summary_path}")
    print("\n现在可以运行 /mlp-tune 进行超参数分析。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)
