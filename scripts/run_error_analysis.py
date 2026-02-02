from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from nlp_textclf.utils.config import load_yaml, save_yaml
from nlp_textclf.utils.seed import set_seed
from nlp_textclf.utils.device import resolve_device
from nlp_textclf.utils.io import ensure_dir, save_json
from nlp_textclf.data.loaders import load_imdb, load_csv_splits
from nlp_textclf.data.vocab import load_vocab
from nlp_textclf.models.lstm_attn import LSTMAttentionClassifier
from nlp_textclf.analysis.error_analysis import (
    predict_table, overall_metrics, plot_confusion_matrix, add_diagnostics,
    sample_buckets, export_labeling_sheet
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["run"]["seed"])
    device = resolve_device(cfg["run"].get("device", "auto"))

    out_dir = ensure_dir(cfg["run"]["output_dir"])
    save_yaml(cfg, out_dir / "config.yaml")

    # dataset
    ds = cfg["dataset"]
    if ds["type"] == "imdb":
        bundle = load_imdb(train_size=ds["train_size"], val_size=ds["val_size"], seed=cfg["run"]["seed"])
        test_df = bundle.test_df
        text_col, label_col = "text", "label"
    elif ds["type"] == "csv_splits":
        bundle = load_csv_splits(ds["split_dir"], text_col=ds["text_col"], label_col=ds["label_col"])
        test_df = bundle.test_df
        text_col, label_col = ds["text_col"], ds["label_col"]
    else:
        raise ValueError(f"Unknown dataset type: {ds['type']}")

    vocab = load_vocab(cfg["checkpoint"]["vocab_path"])

    # We assume attention model checkpoint; if you want pooling model, add another loader here.
    # For interview projects, being explicit is a plus.
    mcfg = cfg["model"]
    model = LSTMAttentionClassifier(
        vocab_size=vocab.size,
        embed_dim=int(mcfg["embed_dim"]),
        hidden_dim=int(mcfg["hidden_dim"]),
        num_classes=2,
        num_layers=int(mcfg.get("num_layers", 1)),
        bidirectional=bool(mcfg.get("bidirectional", True)),
        dropout=float(mcfg.get("dropout", 0.5)),
        pad_idx=vocab.pad_idx,
        attn_dim=int(mcfg.get("attn_dim", 128)),
    )

    ckpt_path = cfg["checkpoint"]["path"]
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 1) predict + export raw table
    pred_df = predict_table(
        model=model,
        df=test_df,
        vocab=vocab,
        lang=ds.get("lang", "en"),
        max_len=cfg["text"]["max_len"],
        text_col=text_col,
        label_col=label_col,
        batch_size=cfg["analysis"]["batch_size"],
        device=device,
    )
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 2) metrics + confusion
    m = overall_metrics(pred_df)
    save_json({k: v for k, v in m.items() if k != "report_table"}, out_dir / "metrics.json")
    (out_dir / "classification_report.txt").write_text(
        m["classification_report"], encoding="utf-8"
    )

    cm = np.array(m["confusion_matrix"])
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", labels=["0", "1"])

    # 3) buckets + labeling sheet
    pred_diag = add_diagnostics(
        pred_df,
        vocab=vocab,
        lang=ds.get("lang", "en"),
        max_len=cfg["text"]["max_len"],
    )
    pred_diag.to_csv(
        out_dir / "test_predictions_with_diag.csv",
        index=False,
        encoding="utf-8-sig",
    )

    buckets = sample_buckets(
        pred_diag,
        borderline_prob_low=cfg["analysis"]["borderline_prob_low"],
        borderline_prob_high=cfg["analysis"]["borderline_prob_high"],
        high_conf_topk=cfg["analysis"]["high_conf_topk"],
        sample_cap_per_bucket=cfg["analysis"]["sample_cap_per_bucket"],
        seed=cfg["run"]["seed"],
    )

    for name, df in buckets.items():
        df.to_csv(out_dir / f"bucket_{name}.csv", index=False, encoding="utf-8-sig")

    export_labeling_sheet(
        buckets=buckets,
        out_csv=out_dir / "samples_to_label.csv",
        label_export_size=cfg["analysis"]["label_export_size"],
        seed=cfg["run"]["seed"],
    )

    print("Done.")
    print(f"Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
