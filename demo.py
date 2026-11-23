"""Interactive demo: type Roman Urdu, see how each tokenizer cuts it and what sentiment it predicts.

Usage: python demo.py   (run python run.py first), then open http://localhost:8000
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer

from run import OUT, SEED, VOCAB_SIZES, clean, load_data

PAGE = Path(__file__).with_name("demo.html").read_bytes()
NAMES = [f"{algo}_{size}" for algo in ["bpe", "wordpiece"] for size in VOCAB_SIZES]


def identity(tokens):
    return tokens


def build_models():
    """Load the saved tokenizers and train one sentiment classifier per tokenizer (same split as run.py)."""
    df = load_data()
    train, _ = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    texts = train["text"].tolist()
    models = {}
    for name in NAMES:
        tok = Tokenizer.from_file(str(OUT / "tokenizers" / f"{name}.json"))
        vec = TfidfVectorizer(analyzer=identity)
        X = vec.fit_transform([e.tokens for e in tok.encode_batch(texts)])
        clf = LogisticRegression(max_iter=1000).fit(X, train["label"])
        models[name] = (tok, vec, clf)
        print(f"ready {name}")
    return models


def analyze(models, text):
    text = clean(text)
    results = []
    for name, (tok, vec, clf) in models.items():
        enc = tok.encode(text)
        words = {}
        for token, word_id in zip(enc.tokens, enc.word_ids):
            words.setdefault(word_id, []).append(token)
        probs = clf.predict_proba(vec.transform([enc.tokens]))[0]
        algo, size = name.split("_")
        results.append({
            "algorithm": "BPE" if algo == "bpe" else "WordPiece",
            "vocab_size": int(size),
            "words": list(words.values()),
            "pieces": len(enc.tokens),
            "sentiment": {label: round(float(p), 3) for label, p in zip(clf.classes_, probs)},
        })
    return {"text": text, "n_words": len(results[0]["words"]) if text else 0, "results": results}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply(200, "text/html; charset=utf-8", PAGE)

    def do_POST(self):
        if self.path != "/api":
            return self.reply(404, "text/plain", b"not found")
        length = min(int(self.headers.get("Content-Length", 0)), 10_000)
        try:
            text = str(json.loads(self.rfile.read(length) or b"{}").get("text", ""))[:500]
        except json.JSONDecodeError:
            return self.reply(400, "text/plain", b"bad json")
        body = json.dumps(analyze(self.server.models, text)).encode()
        self.reply(200, "application/json", body)

    def reply(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    server.models = build_models()
    print("Demo running at http://localhost:8000  (Ctrl+C to stop)")
    server.serve_forever()
