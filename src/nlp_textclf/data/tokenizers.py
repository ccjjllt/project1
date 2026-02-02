from __future__ import annotations

import re
from typing import List, Literal

Lang = Literal["en", "zh"]

def tokenize_en(text: str) -> List[str]:
    text = "" if text is None else str(text)
    text = text.lower()
    return re.findall(r"[a-z]+", text)

def tokenize_zh(text: str) -> List[str]:
    # jieba returns generator-like list of tokens
    import jieba
    text = "" if text is None else str(text)
    # IMPORTANT: for TF-IDF we often want to keep tokens separated by spaces.
    return [t for t in jieba.lcut(text) if t.strip()]

def get_tokenizer(lang: Lang):
    if lang == "zh":
        return tokenize_zh
    return tokenize_en
