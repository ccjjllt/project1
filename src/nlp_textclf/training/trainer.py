from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import compute_metrics, compute_report
from ..utils.io import ensure_dir, save_json

@dataclass
class EpochResult:
    loss: float
    acc: float
    f1_macro: float

def _unwrap_logits(outputs):
    # outputs can be logits or (logits, attn_weights)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    total = 0

    for input_ids, labels in tqdm(dataloader, desc="train", leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = _unwrap_logits(model(input_ids))
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)

        preds = logits.argmax(dim=-1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    m = compute_metrics(y_true, y_pred)
    return EpochResult(loss=total_loss / max(total, 1), acc=m["acc"], f1_macro=m["f1_macro"])

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Tuple[EpochResult, str]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    total = 0

    for input_ids, labels in tqdm(dataloader, desc="eval", leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = _unwrap_logits(model(input_ids))
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)

        preds = logits.argmax(dim=-1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    m = compute_metrics(y_true, y_pred)
    report = compute_report(y_true, y_pred)
    return EpochResult(loss=total_loss / max(total, 1), acc=m["acc"], f1_macro=m["f1_macro"]), report

def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: str | Path,
    save_best_metric: str = "f1_macro",
) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_path = out_dir / "best.pt"

    history = []
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va, va_report = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": ep,
            "train": tr.__dict__,
            "val": va.__dict__,
        })

        score = getattr(va, save_best_metric)
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), best_path)

        # save "last"
        torch.save(model.state_dict(), out_dir / "last.pt")

        # write epoch metrics (append)
        save_json({"history": history}, out_dir / "history.json")
        (out_dir / "val_report.txt").write_text(va_report, encoding="utf-8")

        print(
            f"Epoch {ep:02d} | "
            f"train loss {tr.loss:.4f} acc {tr.acc:.4f} f1 {tr.f1_macro:.4f} | "
            f"val loss {va.loss:.4f} acc {va.acc:.4f} f1 {va.f1_macro:.4f} | "
            f"best {best_val:.4f}"
        )

    # final test with best
    model.load_state_dict(torch.load(best_path, map_location=device))
    te, te_report = evaluate(model, test_loader, criterion, device)
    (out_dir / "test_report.txt").write_text(te_report, encoding="utf-8")

    summary = {
        "best_val_" + save_best_metric: float(best_val),
        "test": te.__dict__,
        "best_path": str(best_path),
    }
    save_json(summary, out_dir / "summary.json")
    return summary
