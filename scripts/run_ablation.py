from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from nlp_textclf.utils.config import load_yaml, save_yaml
from nlp_textclf.utils.seed import set_seed
from nlp_textclf.utils.device import resolve_device
from nlp_textclf.utils.io import ensure_dir, save_json
from nlp_textclf.data.loaders import load_imdb, load_csv_splits, build_dataloaders
from nlp_textclf.data.vocab import save_vocab
from nlp_textclf.models.bilstm_pool import BiLSTMPoolingClassifier
from nlp_textclf.models.lstm_attn import LSTMAttentionClassifier
from nlp_textclf.training.trainer import fit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["run"]["seed"])
    device = resolve_device(cfg["run"].get("device", "auto"))

    out_root = ensure_dir(cfg["run"]["output_dir"])
    save_yaml(cfg, out_root / "config.yaml")

    ds = cfg["dataset"]
    if ds["type"] == "imdb":
        bundle = load_imdb(train_size=ds["train_size"], val_size=ds["val_size"], seed=cfg["run"]["seed"])
        text_col, label_col = "text", "label"
    elif ds["type"] == "csv_splits":
        bundle = load_csv_splits(ds["split_dir"], text_col=ds["text_col"], label_col=ds["label_col"])
        text_col, label_col = ds["text_col"], ds["label_col"]
    else:
        raise ValueError(f"Unknown dataset type: {ds['type']}")

    vocab, train_loader, val_loader, test_loader = build_dataloaders(
        bundle=bundle,
        lang=ds.get("lang", "en"),
        max_len=cfg["text"]["max_len"],
        min_freq=cfg["text"]["min_freq"],
        batch_size=cfg["train"]["batch_size"],
        text_col=text_col,
        label_col=label_col,
    )
    save_vocab(vocab, out_root / "vocab.json")

    results = []
    for exp in cfg["experiments"]:
        name = exp["name"]
        exp_dir = ensure_dir(out_root / name)
        # record config for that run
        save_yaml(exp, exp_dir / "exp.yaml")

        mcfg = cfg["train"]
        base_model_cfg = cfg["text"] | cfg.get("model", {})
        # choose model
        if exp["model_type"] == "bilstm_pool":
            model = BiLSTMPoolingClassifier(
                vocab_size=vocab.size,
                embed_dim=cfg["model"]["embed_dim"],
                hidden_dim=cfg["model"]["hidden_dim"],
                num_classes=2,
                num_layers=cfg["model"].get("num_layers", 1),
                bidirectional=bool(cfg["model"].get("bidirectional", True)),
                dropout=float(cfg["model"].get("dropout", 0.5)),
                pad_idx=vocab.pad_idx,
                pooling=exp.get("pooling", "mean"),
            )
        elif exp["model_type"] == "bilstm_attn":
            model = LSTMAttentionClassifier(
                vocab_size=vocab.size,
                embed_dim=cfg["model"]["embed_dim"],
                hidden_dim=cfg["model"]["hidden_dim"],
                num_classes=2,
                num_layers=cfg["model"].get("num_layers", 1),
                bidirectional=bool(cfg["model"].get("bidirectional", True)),
                dropout=float(cfg["model"].get("dropout", 0.5)),
                pad_idx=vocab.pad_idx,
                attn_dim=int(exp.get("attn_dim", cfg["model"].get("attn_dim", 128))),
            )
        else:
            raise ValueError(f"Unknown model_type: {exp['model_type']}")

        summary = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=cfg["train"]["epochs"],
            lr=cfg["train"]["lr"],
            out_dir=exp_dir,
            save_best_metric="f1_macro",
        )
        res = {
            "name": name,
            "model_type": exp["model_type"],
            "pooling": exp.get("pooling"),
            "attn_dim": exp.get("attn_dim"),
            "best_val_f1": summary["best_val_f1_macro"],
            "test_acc": summary["test"]["acc"],
            "test_f1": summary["test"]["f1_macro"],
            "best_path": summary["best_path"],
        }
        results.append(res)

    df = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    df.to_csv(out_root / "ablation_results.csv", index=False, encoding="utf-8-sig")
    save_json({"results": results}, out_root / "ablation_results.json")

    print("\n=== Ablation Results (sorted by test_f1) ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
