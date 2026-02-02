from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Optional

import pandas as pd

def read_txt(path: str | Path) -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str | Path) -> str:
    from docx import Document  # optional dependency
    path = Path(path)
    doc = Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)

def build_dataframe_from_dirs(
    labeled_dirs: List[Tuple[str, int]],
    min_para_len: int = 5,
) -> pd.DataFrame:
    """
    Build a DataFrame from directories of .txt/.docx files.

    Each directory corresponds to one label.
    We split each file by newline into paragraphs, each paragraph becomes a sample.
    We also keep source_file to support leakage-free splitting.
    """
    samples: List[Dict[str, Any]] = []

    for dir_path, label in labeled_dirs:
        dir_path = Path(dir_path)
        for fp in dir_path.iterdir():
            if fp.is_dir():
                continue
            if fp.suffix.lower() == ".txt":
                text = read_txt(fp)
            elif fp.suffix.lower() == ".docx":
                text = read_docx(fp)
            else:
                continue

            for para in text.split("\n"):
                para = para.strip()
                if len(para) < min_para_len:
                    continue
                samples.append({"text": para, "label": int(label), "source_file": fp.name})

    df = pd.DataFrame(samples)
    return df
