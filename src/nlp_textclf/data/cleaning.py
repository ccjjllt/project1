from __future__ import annotations

import re
import unicodedata
import pandas as pd

def normalize_text(
    text: str,
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    text = "" if text is None else str(text)

    if normalize_unicode:
        # NFKC: full-width -> half-width, unify compatible chars
        text = unicodedata.normalize("NFKC", text)

    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    min_len: int = 5,
    max_len: int = 512,
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(
        lambda x: normalize_text(x, normalize_unicode=normalize_unicode, collapse_whitespace=collapse_whitespace)
    )

    df["__length"] = df[text_col].apply(len)
    df = df[(df["__length"] >= min_len) & (df["__length"] <= max_len)].copy()

    # ensure label is int
    df[label_col] = df[label_col].astype(int)

    df.drop(columns=["__length"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
