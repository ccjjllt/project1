from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from .tokenizers import get_tokenizer, Lang

@dataclass
class TfidfArtifacts:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression

def _pretokenize(texts: List[str], lang: Lang) -> List[str]:
    tok = get_tokenizer(lang)
    # TF-IDF expects space-separated tokens; keep it explicit.
    return [" ".join(tok(t)) for t in texts]

def train_tfidf_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    lang: Lang,
    tfidf_params: Dict[str, Any],
    lr_params: Dict[str, Any],
) -> Tuple[TfidfArtifacts, Dict[str, Any]]:
    X_train = _pretokenize(train_df[text_col].tolist(), lang)
    y_train = train_df[label_col].astype(int).to_numpy()
    X_val = _pretokenize(val_df[text_col].tolist(), lang)
    y_val = val_df[label_col].astype(int).to_numpy()

    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    clf = LogisticRegression(**lr_params)
    clf.fit(X_train_tfidf, y_train)

    pred = clf.predict(X_val_tfidf)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average="macro")

    report = classification_report(y_val, pred, digits=4)
    metrics = {"val_acc": float(acc), "val_f1_macro": float(f1), "val_report": report}
    return TfidfArtifacts(vectorizer=vectorizer, clf=clf), metrics

def evaluate_tfidf_lr(
    artifacts: TfidfArtifacts,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    lang: Lang,
) -> Dict[str, Any]:
    X = _pretokenize(df[text_col].tolist(), lang)
    y = df[label_col].astype(int).to_numpy()

    X_tfidf = artifacts.vectorizer.transform(X)
    pred = artifacts.clf.predict(X_tfidf)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    report = classification_report(y, pred, digits=4)
    return {"acc": float(acc), "f1_macro": float(f1), "report": report}

def save_artifacts(artifacts: TfidfArtifacts, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(artifacts.clf, out_dir / "logreg.joblib")

def load_artifacts(out_dir: str | Path) -> TfidfArtifacts:
    out_dir = Path(out_dir)
    vectorizer = joblib.load(out_dir / "tfidf_vectorizer.joblib")
    clf = joblib.load(out_dir / "logreg.joblib")
    return TfidfArtifacts(vectorizer=vectorizer, clf=clf)
