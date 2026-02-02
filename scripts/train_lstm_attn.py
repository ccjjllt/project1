from __future__ import annotations

import argparse
from pathlib import Path

from nlp_textclf.utils.config import load_yaml, save_yaml
from nlp_textclf.utils.seed import set_seed
from nlp_textclf.utils.device import resolve_device
from nlp_textclf.utils.io import ensure_dir
from nlp_textclf.data.loaders import load_csv_splits, load_imdb, build_dataloaders
from nlp_textclf.data.vocab import save_vocab
from nlp_textclf.models.lstm_attn import LSTMAttentionClassifier
from nlp_textclf.training.trainer import fit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["run"]["seed"])
    device = resolve_device(cfg["run"].get("device", "auto"))

    out_dir = ensure_dir(cfg["run"]["output_dir"])
    save_yaml(cfg, out_dir / "config.yaml")

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

    save_vocab(vocab, out_dir / "vocab.json")

    mcfg = cfg["model"]
    model = LSTMAttentionClassifier(
        vocab_size=vocab.size,
        embed_dim=mcfg["embed_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_classes=2,
        num_layers=mcfg.get("num_layers", 1),
        bidirectional=bool(mcfg.get("bidirectional", True)),
        dropout=float(mcfg.get("dropout", 0.5)),
        pad_idx=vocab.pad_idx,
        attn_dim=int(mcfg.get("attn_dim", 128)),
    )

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        out_dir=out_dir,
        save_best_metric="f1_macro",
    )

    print("==== Done ====")
    print(summary)

if __name__ == "__main__":
    main()
