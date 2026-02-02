from __future__ import annotations

import argparse
from pathlib import Path

from nlp_textclf.utils.config import load_yaml, save_yaml
from nlp_textclf.utils.seed import set_seed
from nlp_textclf.utils.io import ensure_dir, save_json
from nlp_textclf.data.loaders import load_csv_splits
from nlp_textclf.data.tfidf_baseline import train_tfidf_lr, evaluate_tfidf_lr, save_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["run"]["seed"])

    out_dir = ensure_dir(cfg["run"]["output_dir"])
    save_yaml(cfg, out_dir / "config.yaml")

    ds = cfg["dataset"]
    bundle = load_csv_splits(ds["split_dir"], text_col=ds["text_col"], label_col=ds["label_col"])

    tfidf_params = cfg["tfidf"]
    # sklearn expects tuples for ngram_range
    tfidf_params = dict(tfidf_params)
    tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])

    lr_params = {
        "C": cfg["model"].get("C", 1.0),
        "max_iter": cfg["model"].get("max_iter", 2000),
        "solver": "lbfgs",
        "n_jobs": None,
    }

    artifacts, val_metrics = train_tfidf_lr(
        train_df=bundle.train_df,
        val_df=bundle.val_df,
        text_col=ds["text_col"],
        label_col=ds["label_col"],
        lang=ds["lang"],
        tfidf_params=tfidf_params,
        lr_params=lr_params,
    )

    test_metrics = evaluate_tfidf_lr(
        artifacts,
        df=bundle.test_df,
        text_col=ds["text_col"],
        label_col=ds["label_col"],
        lang=ds["lang"],
    )

    save_artifacts(artifacts, out_dir)

    save_json({"val": val_metrics, "test": test_metrics}, out_dir / "metrics.json")
    (out_dir / "val_report.txt").write_text(val_metrics["val_report"], encoding="utf-8")
    (out_dir / "test_report.txt").write_text(test_metrics["report"], encoding="utf-8")

    print("==== TF-IDF + LR ====")
    print(f"Val  acc={val_metrics['val_acc']:.4f} f1_macro={val_metrics['val_f1_macro']:.4f}")
    print(f"Test acc={test_metrics['acc']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
