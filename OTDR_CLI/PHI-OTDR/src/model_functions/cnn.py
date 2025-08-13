from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(31, 3), stride=(4, 1), padding=(15, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            nn.Conv2d(16, 32, kernel_size=(15, 3), stride=(2, 1), padding=(7, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        feats = self.net(x)
        logits = self.classifier(feats)
        return feats, logits

def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
    return logits

@dataclass
class TrainConfig:
    save_path: Path
    device: torch.device
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5

def train_cnn(model: CNN, train_loader, val_loader, cfg: TrainConfig) -> CNN:
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        # train
        model.train()
        tr_correct, tr_total, tr_loss, tr_batches = 0, 0, 0.0, 0
        for batch in train_loader:
            if batch is None:
                continue  # all items in this batch were filtered
            x = batch["data"].unsqueeze(1).to(cfg.device, dtype=torch.float32)
            y = batch["label"].to(cfg.device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.numel()
            tr_batches += 1

        # val
        model.eval()
        va_correct, va_total, va_loss, va_batches = 0, 0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x = batch["data"].unsqueeze(1).to(cfg.device, dtype=torch.float32)
                y = batch["label"].to(cfg.device, dtype=torch.long)
                _, logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item()
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += y.numel()
                va_batches += 1

        tr_acc = tr_correct / max(tr_total, 1)
        va_acc = va_correct / max(va_total, 1)
        print(f"[CNN] Epoch {epoch:03d}  train_acc={tr_acc:.3f}  train_loss={tr_loss/max(tr_batches,1):.4f}  "
              f"val_acc={va_acc:.3f}  val_loss={va_loss/max(va_batches,1):.4f}")

        if va_acc > best_acc and va_batches > 0:
            best_acc = va_acc
            torch.save(model.state_dict(), cfg.save_path)

    print(f"[CNN] Best val acc={best_acc:.4f} (saved to {cfg.save_path})")
    return model
