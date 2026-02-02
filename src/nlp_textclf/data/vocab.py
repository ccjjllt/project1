from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Callable, Iterable, Tuple

PAD = "<pad>"
UNK = "<unk>"

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad_idx: int = 0
    unk_idx: int = 1

    @property
    def size(self) -> int:
        return len(self.stoi)

def build_vocab(
    texts: Iterable[str],
    tokenizer: Callable[[str], List[str]],
    min_freq: int = 5,
) -> Vocab:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenizer(t))

    stoi: Dict[str, int] = {PAD: 0, UNK: 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            stoi[word] = len(stoi)
    itos = {i: w for w, i in stoi.items()}
    return Vocab(stoi=stoi, itos=itos, pad_idx=stoi[PAD], unk_idx=stoi[UNK])

def encode(
    text: str,
    vocab: Vocab,
    tokenizer: Callable[[str], List[str]],
    max_len: int = 256,
) -> List[int]:
    toks = tokenizer(text)
    ids = [vocab.stoi.get(tok, vocab.unk_idx) for tok in toks]

    if len(ids) < max_len:
        ids = ids + [vocab.pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


from pathlib import Path
import json

def save_vocab(vocab: Vocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "stoi": vocab.stoi,
        "pad_idx": vocab.pad_idx,
        "unk_idx": vocab.unk_idx,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_vocab(path: str | Path) -> Vocab:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    stoi = {str(k): int(v) for k, v in obj["stoi"].items()}
    itos = {i: w for w, i in stoi.items()}
    return Vocab(stoi=stoi, itos=itos, pad_idx=int(obj["pad_idx"]), unk_idx=int(obj["unk_idx"]))
