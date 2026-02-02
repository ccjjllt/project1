from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"acc": float(acc), "f1_macro": float(f1)}

def compute_report(y_true: List[int], y_pred: List[int]) -> str:
    return classification_report(y_true, y_pred, digits=4)
