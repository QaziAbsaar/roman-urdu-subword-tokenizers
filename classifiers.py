"""Compare sentiment classifiers on the same split as run.py, for 3 classes and for positive/negative only.

Usage: python classifiers.py   (writes results/classifiers.csv)
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from run import OUT, SEED, load_data

MODELS = {
    "Word unigrams + Logistic Regression (baseline)":
        lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
    "Word 1-2 grams + LinearSVC":
        lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LinearSVC(C=0.5)),
    "Character 2-5 grams + LinearSVC":
        lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                              LinearSVC(C=0.5)),
    "Word 1-2 + character 2-5 grams + LinearSVC": lambda: best_model(),
}


def best_model():
    """Best classifier found: word 1-2 grams and character 2-5 grams, linear SVM."""
    features = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))
    return make_pipeline(features, LinearSVC(C=0.3))


def tasks(df):
    """The two tasks: all 3 classes, and positive vs negative only."""
    return {"3-class": df, "2-class": df[df["label"] != "Neutral"]}


def split(df):
    return train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)


def main():
    rows = []
    for task, data in tasks(load_data()).items():
        train, test = split(data)
        for name, make in MODELS.items():
            pred = make().fit(train["text"], train["label"]).predict(test["text"])
            rows.append({"task": task, "model": name,
                         "accuracy": 100 * accuracy_score(test["label"], pred),
                         "macro_f1": 100 * f1_score(test["label"], pred, average="macro")})
            print(f"{task:8} {name:46} acc {rows[-1]['accuracy']:.1f}  F1 {rows[-1]['macro_f1']:.1f}")
    pd.DataFrame(rows).round(2).to_csv(OUT / "classifiers.csv", index=False)


if __name__ == "__main__":
    main()
