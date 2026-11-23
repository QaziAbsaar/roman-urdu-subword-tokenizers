<p align="center">
  <img src="logo_full.png" alt="PAF-IAST logo" width="600">
</p>

<h1 align="center">Roman Urdu Subword Tokenizers</h1>

<p align="center">
  <b>Benchmarking Byte-Pair Encoding (BPE) vs. WordPiece on Non-Standardized Romanized Scripts: A Roman Urdu Case Study</b><br>
  Natural Language Processing course project, Pak-Austria Fachhochschule: Institute of Applied Sciences and Technology
</p>

---

## Overview

Roman Urdu is Urdu written in the Latin script. It is widely used on social media and in chat apps, but it has no standard spelling. The same word appears in many forms:

| Meaning | Spelling variants |
|---------|-------------------|
| what    | `kya`, `kia`, `kyaa` |
| no      | `nahi`, `nahin`, `nai` |
| is      | `hai`, `hay`, `he` |

This variation makes the vocabulary large and sparse, so word-level models see many out-of-vocabulary (OOV) words. Subword tokenizers split rare words into smaller, frequent pieces. This project compares the two most common subword algorithms, **BPE** and **WordPiece**, on Roman Urdu text.

## Research Questions

1. Which algorithm, BPE or WordPiece, gives lower OOV rate and fertility on Roman Urdu?
2. How does vocabulary size change the results for each algorithm?
3. Which algorithm splits spelling variants of the same word more consistently?
4. Do differences in tokenization quality affect downstream sentiment classification?

## Dataset

[Roman Urdu Data Set](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set) from the UCI Machine Learning Repository. It has about 20,000 social media sentences, each labeled positive, negative or neutral.

## Method

1. **Preprocess:** lowercase, remove URLs, mentions, emojis and duplicates. Split 80% train and 20% test.
2. **Train tokenizers:** BPE and WordPiece at vocabulary sizes 1k, 4k, 8k and 16k (8 tokenizers in total).
3. **Intrinsic evaluation** on the test split:
   - Unknown-token (`[UNK]`) rate
   - Fertility (subword tokens per word)
   - Proportion of words split into two or more tokens
   - Compression (characters per token)
4. **Spelling-variant consistency:** Jaccard similarity between the token sequences of spelling variants.
5. **Downstream evaluation:** TF-IDF and Logistic Regression sentiment classifier on each tokenizer's output. Report accuracy and macro F1.

## Tech Stack

- Python 3
- [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers)
- `pandas`, `scikit-learn`, `matplotlib`
- Jupyter Notebook (no GPU needed)

## Key Results

| Tokenizer | Fertility | Words split (%) | Variant consistency | Macro F1 (%) |
|-----------|-----------|-----------------|---------------------|--------------|
| BPE 1k | 1.685 | 43.33 | 0.10 | 58.95 |
| BPE 4k | 1.319 | 23.59 | 0.07 | 63.01 |
| BPE 16k | 1.127 | 10.31 | 0.05 | 63.26 |
| WordPiece 1k | 1.807 | 47.44 | 0.14 | 58.98 |
| WordPiece 4k | 1.365 | 25.97 | 0.07 | 63.37 |
| WordPiece 16k | 1.142 | 11.41 | 0.06 | 62.63 |
| Word-level baseline | – | – | – | 63.13 |

- BPE fragments words slightly less than WordPiece at every vocabulary size.
- Vocabulary size matters much more than the algorithm.
- Neither algorithm groups spelling variants. Frequent variants (`nahi`, `nahin`, `nai`, `nhi`) each become separate whole tokens.
- From 4k upward, all tokenizers match the word-level baseline on sentiment classification.

Full tables, 8k results and plots are in `results/`, `results.ipynb` and `report.pdf`.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_run.py     # quick checks of the metric functions
python run.py          # downloads data, trains 8 tokenizers, writes results/ (about 2 minutes on CPU)
jupyter notebook results.ipynb
```

Build the documents with `pdflatex proposal.tex` and `pdflatex report.tex` (run twice).

Note: WordPiece training in the `tokenizers` library is not fully deterministic, so WordPiece numbers can change slightly between runs. BPE results are identical on every run.

## Repository Structure

```
.
├── run.py            # Full pipeline: data, tokenizers, metrics, plots
├── test_run.py       # Sanity checks for the metric functions
├── results.ipynb     # Results tables, plots and example tokenizations
├── requirements.txt
├── results/          # metrics.csv, variants.csv, downstream.csv, plots, trained tokenizers
├── proposal.tex/pdf  # Project proposal
├── report.tex/pdf    # Final report
├── logo_full.png     # University logo
└── data/             # Dataset, downloaded by run.py (not committed)
```

## Project Status

- [x] Project proposal
- [x] Data download and preprocessing
- [x] Tokenizer training
- [x] Intrinsic evaluation
- [x] Spelling-variant analysis
- [x] Downstream classification
- [x] Final report

## Team

| Name | Registration No. |
|------|------------------|
| Qazi Absaar Baseer Ud Din | B23F0054CS069 |
| Rania Khan | B23F0082CS073 |
| Memmona Shafiq | B23F0555CS086 |

## References

- R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.
- M. Schuster, K. Nakajima. *Japanese and Korean Voice Search.* ICASSP 2012.
- J. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- P. Rust et al. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
- Z. Sharf. *Roman Urdu Data Set.* UCI Machine Learning Repository, 2018.
