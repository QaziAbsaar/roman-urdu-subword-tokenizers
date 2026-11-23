"""BPE vs WordPiece on Roman Urdu: full pipeline.

Usage: python run.py
Downloads the UCI Roman Urdu dataset, trains 8 tokenizers, and writes
metrics, spelling-variant scores, downstream results and plots to results/.
"""
import io
import itertools
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DATA_URL = "https://archive.ics.uci.edu/static/public/458/roman+urdu+data+set.zip"
DATA_ZIP = Path("data/roman_urdu.zip")
OUT = Path("results")
VOCAB_SIZES = [1000, 4000, 8000, 16000]
SEED = 42

# ---------------- data ----------------
def load_data():
    if not DATA_ZIP.exists():
        DATA_ZIP.parent.mkdir(exist_ok=True)
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
    raw = zipfile.ZipFile(DATA_ZIP).read("Roman Urdu DataSet.csv")
    df = pd.read_csv(io.BytesIO(raw), header=None, encoding="latin-1", usecols=[0, 1],
                     names=["text", "label"])
    df["label"] = df["label"].replace({"Neative": "Negative"})
    df["text"] = df["text"].fillna("").astype(str).map(clean)
    df = df[df["text"].str.len() > 0].drop_duplicates("text")
    return df[df["label"].isin(["Positive", "Negative", "Neutral"])].reset_index(drop=True)


def clean(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)  # emojis and other non-ASCII
    return re.sub(r"\s+", " ", text).strip()


# ---------------- tokenizers ----------------
def train_tokenizer(algo, vocab_size, texts):
    if algo == "BPE":
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    else:
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainer)
    return tok


def strip_prefix(token):
    return token[2:] if token.startswith("##") else token


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "tokenizers").mkdir(exist_ok=True)

    df = load_data()
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    print(f"Sentences: {len(df)} (train {len(train)}, test {len(test)})")
    print(df["label"].value_counts().to_string())

    for algo in ["BPE", "WordPiece"]:
        for size in VOCAB_SIZES:
            tok = train_tokenizer(algo, size, train["text"].tolist())
            tok.save(str(OUT / "tokenizers" / f"{algo.lower()}_{size}.json"))
            print(f"done {algo} {size}")


if __name__ == "__main__":
    main()
