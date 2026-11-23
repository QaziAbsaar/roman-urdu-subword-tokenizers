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


def main():
    df = load_data()
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    print(f"Sentences: {len(df)} (train {len(train)}, test {len(test)})")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
