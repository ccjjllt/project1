from __future__ import annotations

import argparse
from pathlib import Path

from nlp_textclf.utils.config import load_yaml
from nlp_textclf.data.build_csv_dataset import build_from_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path (see configs/dataset_csv.yaml)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ds = cfg["dataset"]
    cl = cfg["cleaning"]
    sp = cfg["split"]

    input_csv = Path(ds["input_csv"])
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Put your raw dataset at data/raw/raw_data.csv (or update configs/dataset_csv.yaml).\n"
            "Expected columns: text,label (and optional source_file)."
        )

    build_from_csv(
        input_csv=input_csv,
        output_clean_csv=ds["output_clean_csv"],
        output_split_dir=ds["output_split_dir"],
        text_col=ds.get("text_col", "text"),
        label_col=ds.get("label_col", "label"),
        source_file_col=ds.get("source_file_col"),
        min_len=cl.get("min_len", 5),
        max_len=cl.get("max_len", 512),
        normalize_unicode=cl.get("normalize_unicode", True),
        collapse_whitespace=cl.get("collapse_whitespace", True),
        train_ratio=sp.get("train_ratio", 0.8),
        val_ratio=sp.get("val_ratio", 0.1),
        test_ratio=sp.get("test_ratio", 0.1),
        seed=sp.get("seed", 42),
    )

    print("Done.")
    print(f"Cleaned CSV -> {ds['output_clean_csv']}")
    print(f"Splits dir   -> {ds['output_split_dir']} (train/val/test.csv)")

if __name__ == "__main__":
    main()
