from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .cleaning import clean_dataset
from .splitting import split_dataframe, SplitRatios

def build_from_csv(
    input_csv: str | Path,
    output_clean_csv: str | Path,
    output_split_dir: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    source_file_col: Optional[str] = None,
    min_len: int = 5,
    max_len: int = 512,
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    df = clean_dataset(
        df,
        text_col=text_col,
        label_col=label_col,
        min_len=min_len,
        max_len=max_len,
        normalize_unicode=normalize_unicode,
        collapse_whitespace=collapse_whitespace,
    )

    output_clean_csv = Path(output_clean_csv)
    output_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_clean_csv, index=False, encoding="utf-8-sig")

    ratios = SplitRatios(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    train_df, val_df, test_df = split_dataframe(
        df,
        label_col=label_col,
        seed=seed,
        ratios=ratios,
        source_file_col=source_file_col,
    )

    out = Path(output_split_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out / "test.csv", index=False, encoding="utf-8-sig")
