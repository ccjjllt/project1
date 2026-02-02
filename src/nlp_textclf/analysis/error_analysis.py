from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

from ..data.tokenizers import get_tokenizer, Lang
from ..data.vocab import Vocab, encode

@torch.no_grad()
def predict_table(
    model,
    df: pd.DataFrame,
    vocab: Vocab,
    lang: Lang,
    max_len: int,
    text_col: str = "text",
    label_col: str = "label",
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    model.eval()
    model.to(device)

    tok = get_tokenizer(lang)

    texts = df[text_col].tolist()
    labels = df[label_col].astype(int).tolist()

    rows = []
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_texts = texts[start:end]
        batch_labels = labels[start:end]

        ids = [encode(t, vocab=vocab, tokenizer=tok, max_len=max_len) for t in batch_texts]
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=-1).tolist()
        pred_conf = probs.max(axis=-1).tolist()

        # binary task helper: prob_pos
        prob_pos = probs[:, 1].tolist() if probs.shape[1] >= 2 else [float(p[0]) for p in probs]

        for t, y, yp, pp, pc in zip(batch_texts, batch_labels, preds, prob_pos, pred_conf):
            rows.append({
                "text": t,
                "y_true": int(y),
                "y_pred": int(yp),
                "prob_pos": float(pp),
                "pred_conf": float(pc),
                "is_error": int(y) != int(yp),
            })

    return pd.DataFrame(rows)

def overall_metrics(df_pred: pd.DataFrame) -> Dict[str, Any]:
    y_true = df_pred["y_true"].to_numpy()
    y_pred = df_pred["y_pred"].to_numpy()
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    rep_text = classification_report(y_true, y_pred, digits=4)
    rep_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "classification_report": rep_text,
        "report_table": rep_dict,
        "confusion_matrix": cm.tolist(),
    }

def plot_confusion_matrix(cm: np.ndarray, out_path: str | Path, labels: List[str] | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def add_diagnostics(
    df_pred: pd.DataFrame,
    vocab: Vocab,
    lang: Lang,
    max_len: int,
) -> pd.DataFrame:
    import re
    tok = get_tokenizer(lang)

    lens = []
    truncated = []
    unk_ratio = []

    for t in df_pred["text"].tolist():
        toks = tok(t)
        lens.append(len(toks))
        truncated.append(len(toks) > max_len)
        ids = [vocab.stoi.get(tok_, vocab.unk_idx) for tok_ in toks[:max_len]]
        if len(ids) == 0:
            unk_ratio.append(1.0)
        else:
            unk_ratio.append(sum(1 for i in ids if i == vocab.unk_idx) / len(ids))

    out = df_pred.copy()
    out["tok_len"] = lens
    out["is_truncated"] = truncated
    out["unk_ratio"] = unk_ratio
    return out

def sample_buckets(
    df_pred_diag: pd.DataFrame,
    borderline_prob_low: float = 0.35,
    borderline_prob_high: float = 0.65,
    high_conf_topk: int = 50,
    sample_cap_per_bucket: int = 50,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    errors = df_pred_diag[df_pred_diag["is_error"]].copy()
    fp = errors[(errors["y_true"] == 0) & (errors["y_pred"] == 1)].copy()
    fn = errors[(errors["y_true"] == 1) & (errors["y_pred"] == 0)].copy()

    high_conf_fp = fp.sort_values("pred_conf", ascending=False).head(high_conf_topk)
    high_conf_fn = fn.sort_values("pred_conf", ascending=False).head(high_conf_topk)

    border_fn = fn[(fn["prob_pos"] > borderline_prob_low) & (fn["prob_pos"] < 0.5)]
    border_fp = fp[(fp["prob_pos"] > 0.5) & (fp["prob_pos"] < borderline_prob_high)]

    trunc_samples = errors[errors["is_truncated"]]
    high_unk_samples = errors[errors["unk_ratio"] > 0.1]

    def _cap(df: pd.DataFrame):
        if len(df) <= sample_cap_per_bucket:
            return df
        return df.sample(n=sample_cap_per_bucket, random_state=seed)

    return {
        "high_conf_fn": high_conf_fn,
        "high_conf_fp": high_conf_fp,
        "border_fn": _cap(border_fn),
        "border_fp": _cap(border_fp),
        "trunc": _cap(trunc_samples),
        "high_unk": _cap(high_unk_samples),
    }

def export_labeling_sheet(
    buckets: Dict[str, pd.DataFrame],
    out_csv: str | Path,
    label_export_size: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.concat(list(buckets.values()), ignore_index=True).drop_duplicates(subset=["text"])
    if len(df) > label_export_size:
        df = df.sample(n=label_export_size, random_state=seed)
    df = df.copy()
    df["error_type"] = ""
    df["comment"] = ""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df
