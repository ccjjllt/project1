from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizers import get_tokenizer, Lang
from .vocab import Vocab, build_vocab, encode

@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

def load_csv_splits(split_dir: str | Path, text_col: str="text", label_col: str="label") -> DatasetBundle:
    split_dir = Path(split_dir)
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    return DatasetBundle(train_df=train_df, val_df=val_df, test_df=test_df)

def load_imdb(train_size: int=20000, val_size: int=5000, seed: int=42) -> DatasetBundle:
    from datasets import load_dataset
    raw = load_dataset("imdb")
    train_valid = raw["train"].train_test_split(test_size=val_size, seed=seed)
    train = train_valid["train"].shuffle(seed=seed).select(range(train_size))
    val = train_valid["test"]
    test = raw["test"]
    # Convert to pandas for统一接口
    return DatasetBundle(
        train_df=train.to_pandas(),
        val_df=val.to_pandas(),
        test_df=test.to_pandas(),
    )

class TorchTextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        tokenizer: Callable[[str], List[str]],
        max_len: int,
        text_col: str = "text",
        label_col: str = "label",
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        text = r[self.text_col]
        label = int(r[self.label_col])
        ids = encode(text, vocab=self.vocab, tokenizer=self.tokenizer, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_dataloaders(
    bundle: DatasetBundle,
    lang: Lang,
    max_len: int,
    min_freq: int,
    batch_size: int,
    text_col: str = "text",
    label_col: str = "label",
    num_workers: int = 0,
):
    tokenizer = get_tokenizer(lang)
    vocab = build_vocab(bundle.train_df[text_col].tolist(), tokenizer=tokenizer, min_freq=min_freq)

    train_ds = TorchTextDataset(bundle.train_df, vocab, tokenizer, max_len, text_col=text_col, label_col=label_col)
    val_ds = TorchTextDataset(bundle.val_df, vocab, tokenizer, max_len, text_col=text_col, label_col=label_col)
    test_ds = TorchTextDataset(bundle.test_df, vocab, tokenizer, max_len, text_col=text_col, label_col=label_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return vocab, train_loader, val_loader, test_loader
