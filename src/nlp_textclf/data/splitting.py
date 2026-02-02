from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class SplitRatios:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def validate(self) -> None:
        s = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {s}")

def split_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    seed: int = 42,
    ratios: SplitRatios = SplitRatios(),
    source_file_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    If source_file_col is provided, we split by unique files first to prevent leakage.
    Otherwise split by rows (stratified by label_col).
    """
    ratios.validate()

    if source_file_col and source_file_col in df.columns:
        files = df[source_file_col].astype(str).unique().tolist()
        train_files, temp_files = train_test_split(
            files,
            test_size=(1 - ratios.train_ratio),
            random_state=seed
        )
        # temp split into val/test
        val_ratio_rel = ratios.val_ratio / (ratios.val_ratio + ratios.test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - val_ratio_rel),
            random_state=seed
        )
        train_df = df[df[source_file_col].isin(train_files)].copy()
        val_df = df[df[source_file_col].isin(val_files)].copy()
        test_df = df[df[source_file_col].isin(test_files)].copy()
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # row-level stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - ratios.train_ratio),
        stratify=df[label_col],
        random_state=seed
    )
    val_ratio_rel = ratios.val_ratio / (ratios.val_ratio + ratios.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_rel),
        stratify=temp_df[label_col],
        random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
