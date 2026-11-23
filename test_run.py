"""Small sanity checks for the metric functions in run.py. Run: python test_run.py"""
from run import clean, intrinsic, jaccard, train_tokenizer, variant_consistency

texts = ["kya baat hai yaar", "nahi yaar acha hai", "bohat acha kia"] * 20

assert clean("Kya BAAT hai 😀 https://x.com @ali") == "kya baat hai"
assert jaccard(["a", "b"], ["a", "b"]) == 1.0
assert jaccard(["a"], ["b"]) == 0.0

for algo in ["BPE", "WordPiece"]:
    tok = train_tokenizer(algo, 200, texts)
    m = intrinsic(tok, ["kya baat hai"])
    assert m["fertility"] >= 1.0, m
    assert m["unk_rate"] == 0.0, m  # all characters seen in training
    assert intrinsic(tok, ["kya zzz"])["unk_rate"] > 0  # unseen character -> [UNK]
    assert variant_consistency(tok, ["kya", "kya"]) == 1.0

print("all checks passed")
