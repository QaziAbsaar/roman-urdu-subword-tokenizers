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

# Common Roman Urdu words and their spelling variants (same meaning, different spelling).
VARIANTS = {
    "what": ["kya", "kia", "kyaa"],
    "no": ["nahi", "nahin", "nai", "nhi"],
    "is": ["hai", "hay", "hy"],
    "good": ["acha", "achha", "achaa"],
    "good (f)": ["achi", "achhi", "achchi"],
    "very": ["bohat", "bahut", "bohot", "boht"],
    "me": ["mujhe", "mujhay", "mjhe"],
    "why": ["kyun", "kyon", "kiun"],
    "friend": ["yaar", "yar", "yaarr"],
    "fine": ["theek", "thik", "theak"],
    "brother": ["bhai", "bhae", "bhaee"],
    "pakistan": ["pakistan", "pakstan", "pakistaan"],
    "life": ["zindagi", "zindgi", "zindagee"],
    "happy": ["khush", "khus", "kush"],
    "god": ["allah", "alla", "allaah"],
    "thanks": ["shukriya", "shukria", "shukriyah"],
}


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


# ---------------- metrics ----------------
def intrinsic(tok, texts):
    """UNK rate, fertility, % continued words and chars per token over texts."""
    n_tokens = n_unk = n_words = n_continued = n_chars = 0
    for enc in tok.encode_batch(texts):
        n_tokens += len(enc.tokens)
        n_unk += enc.tokens.count("[UNK]")
        per_word = Counter(w for w in enc.word_ids if w is not None)
        n_words += len(per_word)
        n_continued += sum(1 for c in per_word.values() if c >= 2)
        n_chars += sum(len(strip_prefix(t)) if t != "[UNK]" else 1 for t in enc.tokens)
    return {
        "unk_rate": 100 * n_unk / n_tokens,
        "fertility": n_tokens / n_words,
        "continued_words": 100 * n_continued / n_words,
        "chars_per_token": n_chars / n_tokens,
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def variant_consistency(tok, words):
    """Mean pairwise Jaccard similarity of the token sets of spelling variants."""
    pieces = [[strip_prefix(t) for t in tok.encode(w).tokens] for w in words]
    pairs = list(itertools.combinations(pieces, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def classify(train_docs, train_y, test_docs, test_y, analyzer=None):
    vec = TfidfVectorizer(analyzer=analyzer) if analyzer else TfidfVectorizer()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(train_docs), train_y)
    pred = clf.predict(vec.transform(test_docs))
    return {"accuracy": 100 * accuracy_score(test_y, pred),
            "macro_f1": 100 * f1_score(test_y, pred, average="macro")}


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "tokenizers").mkdir(exist_ok=True)

    df = load_data()
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    print(f"Sentences: {len(df)} (train {len(train)}, test {len(test)})")
    print(df["label"].value_counts().to_string())

    metrics, variants, downstream = [], [], []
    for algo in ["BPE", "WordPiece"]:
        for size in VOCAB_SIZES:
            tok = train_tokenizer(algo, size, train["text"].tolist())
            tok.save(str(OUT / "tokenizers" / f"{algo.lower()}_{size}.json"))
            row = {"algorithm": algo, "vocab_size": size}

            metrics.append({**row, **intrinsic(tok, test["text"].tolist())})

            scores = {name: variant_consistency(tok, words) for name, words in VARIANTS.items()}
            variants.append({**row, **scores, "mean": sum(scores.values()) / len(scores)})

            tr = [e.tokens for e in tok.encode_batch(train["text"].tolist())]
            te = [e.tokens for e in tok.encode_batch(test["text"].tolist())]
            downstream.append({**row, **classify(tr, train["label"], te, test["label"],
                                                 analyzer=lambda x: x)})
            print(f"done {algo} {size}")

    baseline = classify(train["text"], train["label"], test["text"], test["label"])
    downstream.append({"algorithm": "Word-level", "vocab_size": None, **baseline})

    metrics, variants, downstream = map(pd.DataFrame, (metrics, variants, downstream))
    metrics.round(3).to_csv(OUT / "metrics.csv", index=False)
    variants.round(3).to_csv(OUT / "variants.csv", index=False)
    downstream.round(2).to_csv(OUT / "downstream.csv", index=False)


    print("\nIntrinsic metrics:\n", metrics.round(3).to_string(index=False))
    print("\nVariant consistency (mean):\n",
          variants[["algorithm", "vocab_size", "mean"]].round(3).to_string(index=False))
    print("\nDownstream:\n", downstream.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
